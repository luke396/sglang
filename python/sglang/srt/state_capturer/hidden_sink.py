"""Export thread and sinks for hidden-state capture.

Consumes jobs enqueued at request finish, waits for the finalize barrier,
validates row identity against the host sidecar, and hands one
SpecForge-``OfflineEagle3Dataset``-compatible record per sample to the
configured sink::

    {
        "input_ids":        LongTensor [T],
        "loss_mask":        LongTensor [T],   # all-ones placeholder
        "aux_hidden_state": bf16 [1, T, K*H], # packed aux, serving layout
        "hidden_state":     bf16 [1, T, H],   # post-final-norm, pre-LM-head
        "rid":              str,              # original request id
        "prompt_len":       int,              # prompt rows in [0, T)
    }

Sinks share ``put(sample_id, record) -> bool`` / ``write_fingerprint(dict)``:
``HiddenFileSink`` (default) writes per-sample ``.ckpt`` files;
``AsyncMooncakeHiddenSink`` (``SGLANG_HIDDEN_CAPTURE_SINK=mooncake``, in
``hidden_mooncake.py``) publishes self-describing keys into a Mooncake store
through a fail-soft background connection.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import msgspec
import torch

from sglang.srt.state_capturer.hidden_host import (
    HiddenCaptureBookkeeper,
    HiddenCaptureStats,
    HiddenFinalizeWorker,
    HiddenHostSidecar,
    HiddenWallSpanRecorder,
)
from sglang.srt.state_capturer.hidden_mooncake import SampleTooLargeError

logger = logging.getLogger(__name__)

# Generous bound on staging-ring drain; a healthy finalize thread settles a
# slot in ~ms, so hitting this means the pipeline is wedged -> fail closed.
_BARRIER_TIMEOUT_S = 30.0
_ORPHAN_SWEEP_INTERVAL_S = 60.0


def _union_ns(intervals, start_ns: int, end_ns: int) -> int:
    clipped = sorted(
        (max(start_ns, left), min(end_ns, right))
        for left, right in intervals
        if right > start_ns and left < end_ns
    )
    total = 0
    current_left = current_right = None
    for left, right in clipped:
        if right <= left:
            continue
        if current_left is None:
            current_left, current_right = left, right
        elif left <= current_right:
            current_right = max(current_right, right)
        else:
            total += current_right - current_left
            current_left, current_right = left, right
    if current_left is not None:
        total += current_right - current_left
    return total


class HiddenExportTrace:
    """One export job's cross-thread gather/put wall intervals.

    The trace retains intervals, not summed busy time. At completion a sweep
    produces disjoint queue, finalize, barrier, gather, put, overlap, and
    other buckets. Finalize spans are intersected only with the dequeued
    pre-barrier phase, so all components add up exactly to the
    finish-hook-to-export wall clock.
    """

    def __init__(self, start_ns: int, dequeued_ns: int) -> None:
        self.start_ns = int(start_ns)
        self.dequeued_ns = int(dequeued_ns)
        self.barrier_end_ns = int(start_ns)
        self._lock = threading.Lock()
        self._intervals = []

    @contextlib.contextmanager
    def span(self, kind: str):
        started_ns = time.monotonic_ns()
        try:
            yield
        finally:
            self.record(kind, started_ns, time.monotonic_ns())

    def record(self, kind: str, start_ns: int, end_ns: int) -> None:
        if kind not in ("gather", "put") or end_ns <= start_ns:
            return
        with self._lock:
            self._intervals.append((kind, int(start_ns), int(end_ns)))

    def attribute(
        self,
        *,
        barrier_end_ns: int,
        end_ns: int,
        span_recorder: HiddenWallSpanRecorder,
    ) -> Dict[str, int]:
        start_ns = min(self.start_ns, end_ns)
        dequeued_ns = min(max(start_ns, self.dequeued_ns), end_ns)
        barrier_end_ns = min(max(dequeued_ns, barrier_end_ns), end_ns)
        queue_ns = dequeued_ns - start_ns
        finalize_intervals = span_recorder.finalize_intervals(
            dequeued_ns, barrier_end_ns
        )
        finalize_ns = _union_ns(finalize_intervals, dequeued_ns, barrier_end_ns)
        barrier_ns = barrier_end_ns - dequeued_ns - finalize_ns

        with self._lock:
            intervals = list(self._intervals)
        events = {}
        for kind, left, right in intervals:
            left = max(barrier_end_ns, left)
            right = min(end_ns, right)
            if right <= left:
                continue
            events.setdefault(left, []).append((kind, 1))
            events.setdefault(right, []).append((kind, -1))

        buckets = {"gather": 0, "put": 0, "gather_put_overlap": 0}
        active = {"gather": 0, "put": 0}
        cursor = barrier_end_ns
        for point in sorted(events):
            duration = max(0, point - cursor)
            if active["gather"] and active["put"]:
                buckets["gather_put_overlap"] += duration
            elif active["gather"]:
                buckets["gather"] += duration
            elif active["put"]:
                buckets["put"] += duration
            for kind, delta in events[point]:
                active[kind] += delta
            cursor = point
        duration = max(0, end_ns - cursor)
        if active["gather"] and active["put"]:
            buckets["gather_put_overlap"] += duration
        elif active["gather"]:
            buckets["gather"] += duration
        elif active["put"]:
            buckets["put"] += duration

        post_ns = end_ns - barrier_end_ns
        post_accounted = sum(buckets.values())
        other_ns = max(0, post_ns - post_accounted)
        return {
            "wall_ns": end_ns - start_ns,
            "queue": queue_ns,
            "finalize": finalize_ns,
            "barrier": barrier_ns,
            **buckets,
            "other": other_ns,
        }


class HiddenExportJob(msgspec.Struct):
    rid: str
    sample_id: str
    tokens: torch.Tensor  # [T] long: forwarded token ids (prompt + committed)
    slots: torch.Tensor  # [T] long: kv token-slot per token
    ring_seq_barrier: int
    # Rows [0, prompt_len) are prompt; the rest are verify-committed decode
    # rows. Recorded for downstream loss-mask reconstruction.
    prompt_len: int = 0
    # Set by submit if the caller leaves it zero. It anchors the critical
    # path at finish-hook admission, including export-queue and barrier wait.
    enqueued_ns: int = 0


class HiddenFileSink:
    """Atomic per-sample ``torch.save``: tmp write + rename, no partial files."""

    def __init__(self, sink_dir: str) -> None:
        self.sink_dir = sink_dir
        os.makedirs(sink_dir, exist_ok=True)

    def put(self, sample_id: str, record: Dict[str, torch.Tensor]) -> bool:
        """Write one sample; False if the id was already exported.

        First write wins: rid is caller-suppliable, so a duplicate means two
        distinct requests mapped to one sample id — overwriting would silently
        drop the first sample.
        """
        final_path = os.path.join(self.sink_dir, f"{sample_id}.ckpt")
        if os.path.exists(final_path):
            return False
        tmp_path = final_path + ".tmp"
        torch.save(record, tmp_path)
        os.replace(tmp_path, final_path)
        return True

    def write_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        final_path = os.path.join(self.sink_dir, "_fingerprint.json")
        if os.path.exists(final_path):
            return  # idempotent across restarts and DP replicas
        tmp_path = f"{final_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(fingerprint, f, indent=2, sort_keys=True)
        os.replace(tmp_path, final_path)


class HiddenExportWorker:
    """Daemon thread: barrier-wait, validate, gather, sink.

    Bounded queue; a full queue is a capture-miss for that sample, never
    backpressure on the scheduler.
    """

    def __init__(
        self,
        *,
        sidecar: HiddenHostSidecar,
        bookkeeper: HiddenCaptureBookkeeper,
        finalize_worker: HiddenFinalizeWorker,
        stats: HiddenCaptureStats,
        sink: Any,
        queue_size: int,
        barrier_timeout_s: float = _BARRIER_TIMEOUT_S,
    ) -> None:
        self.sidecar = sidecar
        self.bookkeeper = bookkeeper
        self.finalize_worker = finalize_worker
        self.span_recorder = finalize_worker.span_recorder
        self.stats = stats
        self.sink = sink
        self._queue: queue.Queue[HiddenExportJob] = queue.Queue(maxsize=queue_size)
        self._barrier_timeout_s = barrier_timeout_s
        self._admission_lock = threading.Lock()
        self._accepting = True
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = 0
        self._last_sweep_s = time.monotonic()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="hidden-capture-export", daemon=True
        )
        self._thread.start()

    def stop_admission(self) -> None:
        with self._admission_lock:
            self._accepting = False

    def stop(self) -> None:
        """Stop admission; the worker drains the queue before exiting."""
        self.stop_admission()
        self._stop_requested.set()

    def join(self, timeout_s: Optional[float] = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def _settle_sidecar_rows(self, own_slot_gens: Dict[int, int]) -> None:
        # Prefix rows are immutable in Mooncake after export and can leave the
        # arena. The file sink retains the newest version as an evictable warm
        # cache, but its in-flight export lease must still be removed.
        self.sidecar.release_rows(
            own_slot_gens,
            discard=bool(getattr(self.sink, "prefix_enabled", False)),
        )

    def submit(self, job: HiddenExportJob) -> bool:
        with self._admission_lock:
            if not self._accepting:
                self.stats.bump("shutdown_admission_miss_ct")
                own_slot_gens = self.bookkeeper.pop(job.rid)
                self._settle_sidecar_rows(own_slot_gens)
                return False
            try:
                if job.enqueued_ns <= 0:
                    job.enqueued_ns = time.monotonic_ns()
                self._queue.put_nowait(job)
            except queue.Full:
                self.stats.bump("export_queue_full_miss_ct")
                own_slot_gens = self.bookkeeper.pop(job.rid)
                self._settle_sidecar_rows(own_slot_gens)
                return False
            self.stats.observe_max("export_queue_high_water_ct", self._queue.qsize())
            return True

    def _run(self) -> None:
        while True:
            if self._stop_requested.is_set() and (self._queue.empty()):
                break
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._maybe_sweep_orphans()
                continue
            try:
                self._active = 1
                self.export_one(job)
            except Exception:
                # Sink write failed (e.g. mooncake store out of space) after
                # identity validation passed -- a real dropped sample, not a
                # pipeline miss. Count it so coverage loss is attributable
                # without diffing export_ok_ct against completed requests.
                self.stats.bump("sink_put_failed_miss_ct")
                logger.exception("hidden capture export failed for rid %s", job.rid)
            finally:
                self._active = 0
            self._maybe_sweep_orphans()

    def _maybe_sweep_orphans(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep_s > _ORPHAN_SWEEP_INTERVAL_S:
            self._last_sweep_s = now
            orphan_rows = self.bookkeeper.take_orphans()
            swept = len(orphan_rows)
            for own_slot_gens in orphan_rows:
                self._settle_sidecar_rows(own_slot_gens)
            if swept:
                logger.info("hidden capture: swept %d orphaned records", swept)

    def _wait_barrier(self, ring_seq_barrier: int) -> bool:
        deadline = time.monotonic() + self._barrier_timeout_s
        while self.finalize_worker.last_finalized_seq < ring_seq_barrier:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    def export_one(self, job: HiddenExportJob) -> None:
        dequeued_ns = time.monotonic_ns()
        trace = HiddenExportTrace(
            job.enqueued_ns or dequeued_ns, dequeued_ns=dequeued_ns
        )
        try:
            self._export_one(job, trace)
        finally:
            ended_ns = time.monotonic_ns()
            components = trace.attribute(
                barrier_end_ns=trace.barrier_end_ns,
                end_ns=ended_ns,
                span_recorder=self.span_recorder,
            )
            self.stats.record_export_critical_path(components)

    def _export_one(self, job: HiddenExportJob, trace: HiddenExportTrace) -> None:
        barrier_started_ns = time.monotonic_ns()
        barrier_ready = self._wait_barrier(job.ring_seq_barrier)
        trace.barrier_end_ns = time.monotonic_ns()
        self.stats.bump(
            "export_barrier_wait_ns_ct", trace.barrier_end_ns - barrier_started_ns
        )
        if not barrier_ready:
            self.stats.bump("export_timeout_miss_ct")
            own_slot_gens = self.bookkeeper.pop(job.rid)
            self._settle_sidecar_rows(own_slot_gens)
            return

        missed = self.bookkeeper.is_missed(job.rid)
        own_slot_gens = self.bookkeeper.pop(job.rid)
        try:
            self._export_settled(job, trace, missed, own_slot_gens)
        finally:
            # Failures and duplicates settle leases too: the sample has
            # already become an attributable miss and must not pin capacity.
            self._settle_sidecar_rows(own_slot_gens)

    def _export_settled(
        self,
        job: HiddenExportJob,
        trace: HiddenExportTrace,
        missed: bool,
        own_slot_gens: Dict[int, int],
    ) -> None:
        if missed:
            # Staging already dropped rows for this request; whole-sample miss.
            return

        if getattr(self.sink, "prefix_enabled", False):
            try:
                sink_started_ns = time.monotonic_ns()
                try:
                    exported = self.sink.put_prefix_sample(
                        sample_id=job.sample_id,
                        rid=job.rid,
                        tokens=job.tokens,
                        slots=job.slots,
                        prompt_len=job.prompt_len,
                        own_slot_gens=own_slot_gens,
                        sidecar=self.sidecar,
                        trace=trace,
                    )
                finally:
                    self.stats.bump(
                        "sink_put_busy_ns_ct",
                        time.monotonic_ns() - sink_started_ns,
                    )
            except SampleTooLargeError:
                self.stats.bump("sample_too_large_miss_ct")
                return
            if exported is None:
                return
            self._record_export_result(job, exported)
            return

        # File sink: gather the sample's rows out of the sidecar and write
        # one SpecForge-compatible record.
        gather_started_ns = time.monotonic_ns()
        try:
            with trace.span("gather"):
                rows = self.sidecar.read_rows_validated(
                    slots=job.slots,
                    expected_tokens=job.tokens,
                    own_slot_gens=own_slot_gens,
                )
        finally:
            self.stats.bump(
                "export_gather_busy_ns_ct", time.monotonic_ns() - gather_started_ns
            )
        if rows is None:
            # Counter name is historical: this is the whole-sample export
            # path's row-identity validation failure, not prefix-specific.
            self.stats.bump("prefix_invalid_miss_ct")
            logger.debug("hidden capture: identity validation failed, rid=%s", job.rid)
            return
        aux_rows, last_rows = rows

        num_tokens = job.tokens.shape[0]
        record = {
            "input_ids": job.tokens.to(torch.long),
            # Placeholder: bypass capture cannot see chat-template role
            # boundaries; recompute the real mask offline before training
            # (also flagged in _fingerprint.json).
            "loss_mask": torch.ones(num_tokens, dtype=torch.long),
            "aux_hidden_state": aux_rows.unsqueeze(0),
            "hidden_state": last_rows.unsqueeze(0),
            "rid": job.rid,
            "prompt_len": job.prompt_len,
        }
        sink_started_ns = time.monotonic_ns()
        try:
            with trace.span("put"):
                exported = self.sink.put(job.sample_id, record)
        finally:
            self.stats.bump(
                "sink_put_busy_ns_ct", time.monotonic_ns() - sink_started_ns
            )
        self._record_export_result(job, exported)

    def _record_export_result(self, job: HiddenExportJob, exported: bool) -> None:
        if exported:
            self.stats.bump("export_ok_ct")
        else:
            self.stats.bump("duplicate_sample_miss_ct")
            logger.warning(
                "hidden capture: duplicate sample id %s (rid %s); keeping the "
                "first export",
                job.sample_id,
                job.rid,
            )
