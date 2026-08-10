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
    }

Sinks share ``put(sample_id, record) -> bool`` / ``write_fingerprint(dict)``:
``HiddenFileSink`` (default) writes per-sample ``.ckpt`` files;
``MooncakeHiddenSink`` (``SGLANG_HIDDEN_CAPTURE_SINK=mooncake``, in
``hidden_mooncake.py``) publishes self-describing keys into a Mooncake store.
"""

from __future__ import annotations

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
)
from sglang.srt.state_capturer.hidden_mooncake import SampleTooLargeError

logger = logging.getLogger(__name__)

# Generous bound on staging-ring drain; a healthy finalize thread settles a
# slot in ~ms, so hitting this means the pipeline is wedged -> fail closed.
_BARRIER_TIMEOUT_S = 30.0
_ORPHAN_SWEEP_INTERVAL_S = 60.0


class HiddenExportJob(msgspec.Struct):
    rid: str
    sample_id: str
    tokens: torch.Tensor  # [T] long: forwarded token ids (prompt + committed)
    slots: torch.Tensor  # [T] long: kv token-slot per token
    ring_seq_barrier: int
    # Rows [0, prompt_len) are prompt; the rest are verify-committed decode
    # rows. Recorded for downstream loss-mask reconstruction.
    prompt_len: int = 0


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
        self.stats = stats
        self.sink = sink
        self._queue: queue.Queue[HiddenExportJob] = queue.Queue(maxsize=queue_size)
        self._barrier_timeout_s = barrier_timeout_s
        self._running = True
        self._thread: Optional[threading.Thread] = None
        self._last_sweep_s = time.monotonic()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="hidden-capture-export", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def submit(self, job: HiddenExportJob) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self.stats.bump("export_queue_full_miss_ct")
            self.bookkeeper.pop(job.rid)
            return False

    def _run(self) -> None:
        while self._running:
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._maybe_sweep_orphans()
                continue
            try:
                self.export_one(job)
            except Exception:
                logger.exception("hidden capture export failed for rid %s", job.rid)
            self._maybe_sweep_orphans()

    def _maybe_sweep_orphans(self) -> None:
        now = time.monotonic()
        if now - self._last_sweep_s > _ORPHAN_SWEEP_INTERVAL_S:
            self._last_sweep_s = now
            swept = self.bookkeeper.sweep_orphans()
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
        if not self._wait_barrier(job.ring_seq_barrier):
            self.stats.bump("export_timeout_miss_ct")
            self.bookkeeper.pop(job.rid)
            return

        missed = self.bookkeeper.is_missed(job.rid)
        own_slot_gens = self.bookkeeper.pop(job.rid)
        if missed:
            # Staging already dropped rows for this request; whole-sample miss.
            return

        rows = self.sidecar.read_rows_validated(
            slots=job.slots,
            expected_tokens=job.tokens,
            own_slot_gens=own_slot_gens,
        )
        if rows is None:
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
        try:
            exported = self.sink.put(job.sample_id, record)
        except SampleTooLargeError:
            self.stats.bump("sample_too_large_miss_ct")
            return
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
