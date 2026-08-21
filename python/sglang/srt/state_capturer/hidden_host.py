"""Host-side capture pipeline: staging rings, sidecar, finalize worker.

Data path::

    prefill forward -> async D2H of newly computed rows into pinned slots
    verify forward  -> GPU pack of committed rows into a device twin blob
        -> one upper-bound-length D2H of the whole blob (single copy/event)
        -> HiddenFinalizeWorker (global-seq event gate; sole sidecar writer)
        -> HiddenHostSidecar    (bounded sparse payload cache)
        -> export thread        (segment reuse plus direct suffix gather)

Threads: scheduler (staging enqueue, finish snapshots), one finalize thread
(sole sidecar writer), one export thread (reader). The scheduler is the ONLY
submitter on the capture stream, so per-ring event completion order equals
enqueue order — reap/finalize polling exploits this (first pending event ends
the scan). Concurrency and identity rules live on the classes that own them:
``HiddenHostSidecar`` (pin/COW fence, generation identity),
``HiddenStagingRing`` (event-gated FIFO), ``HiddenFinalizeWorker`` (global
sequence order). Package glossary: ``state_capturer/__init__.py``.

Every inconsistency fails closed to a whole-sample capture miss; capture
never backpressures or corrupts serving.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

logger = logging.getLogger(__name__)

_GB = 1024**3


def _set_os_thread_name(name: str) -> None:
    """Give the calling thread a kernel-visible name (prctl PR_SET_NAME).

    Python ``Thread(name=...)`` never reaches the OS, so profilers (nsys,
    perf) attribute capture-worker CUDA API calls to the process comm.
    nsys attribution of capture host overhead (issue #10/#11) requires the
    workers to be separable by name; comm is capped at 15 chars.
    """
    try:
        import ctypes

        PR_SET_NAME = 15
        ctypes.CDLL("libc.so.6").prctl(PR_SET_NAME, name.encode()[:15], 0, 0, 0)
    except Exception:  # non-Linux or restricted environment: name is cosmetic
        pass

# Finalize-side stats log cadence, in seconds (time-based so short runs
# still get attribution before shutdown).
_STATS_LOG_INTERVAL_S = 30.0
# Orphaned bookkeeping entries (aborted requests) older than this are swept.
_BOOKKEEPING_TTL_S = 600.0


def _align16(nbytes: int) -> int:
    return (nbytes + 15) & ~15


class _VerifyBlobViews(msgspec.Struct):
    """Typed views into one verify step's contiguous byte blob."""

    commit_lens: torch.Tensor  # [bs] int32
    aux: torch.Tensor  # [n_rows, aux_width]
    last: torch.Tensor  # [n_rows, last_width]
    cache_loc: torch.Tensor  # [n_rows] int64
    tokens: torch.Tensor  # [n_rows] int64
    copy_bytes: int


def verify_blob_bytes(
    *, bs: int, n_rows: int, aux_width: int, last_width: int, dtype: torch.dtype
) -> int:
    """Byte length of the verify blob layout for one step's upper bound."""
    return (
        _align16(bs * torch.int32.itemsize)
        + _align16(n_rows * aux_width * dtype.itemsize)
        + _align16(n_rows * last_width * dtype.itemsize)
        + _align16(n_rows * torch.int64.itemsize)
        + _align16(n_rows * torch.int64.itemsize)
    )


def verify_blob_views(
    blob: torch.Tensor,
    *,
    bs: int,
    stride: int,
    aux_width: int,
    last_width: int,
    dtype: torch.dtype,
) -> _VerifyBlobViews:
    """Carve one verify step's typed views out of a contiguous byte blob.

    Layout (16-byte-aligned sections, sized by this step's host-known upper
    bound ``n_rows = bs * stride``): commit_lens | aux | last | cache_loc |
    tokens. The device twin and the pinned staging slot use the same layout,
    so one length-bounded byte copy transfers every field — commit_lens is a
    plain field of the block, with no separate copy, event, or ordering.
    """
    n_rows = bs * stride
    offset = 0

    def _take(numel: int, section_dtype: torch.dtype) -> torch.Tensor:
        nonlocal offset
        nbytes = numel * section_dtype.itemsize
        view = blob[offset : offset + nbytes].view(section_dtype)
        offset = _align16(offset + nbytes)
        return view

    commit_lens = _take(bs, torch.int32)
    aux = _take(n_rows * aux_width, dtype).view(n_rows, aux_width)
    last = _take(n_rows * last_width, dtype).view(n_rows, last_width)
    cache_loc = _take(n_rows, torch.int64)
    tokens = _take(n_rows, torch.int64)
    return _VerifyBlobViews(
        commit_lens=commit_lens,
        aux=aux,
        last=last,
        cache_loc=cache_loc,
        tokens=tokens,
        copy_bytes=offset,
    )


class _NullEvent:
    """Event stand-in for CPU-only unit tests: enqueue copies are synchronous."""

    def record(self) -> None:
        pass

    def query(self) -> bool:
        return True


class HiddenCaptureStats:
    """Monotonic counters (``_ct``) for observability; single lock, tiny ops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rows_staged_ct = 0
        self.prefill_rows_staged_ct = 0
        self.verify_candidate_rows_staged_ct = 0
        self.prefill_d2h_bytes_ct = 0
        self.verify_d2h_bytes_ct = 0
        self.verify_pack_ns_ct = 0
        self.verify_twin_hold_ns_ct = 0
        self.verify_twin_high_water_ct = 0
        self.slots_finalized_ct = 0
        self.prefill_rows_finalized_ct = 0
        self.prefill_finalize_slow_gather_ct = 0
        self.finalize_prefill_busy_ns_ct = 0
        self.finalize_verify_busy_ns_ct = 0
        # Settlement backlog observability (#9): max enqueue-to-finalize seq
        # distance, and max event-completed-but-unsettled slots held behind
        # the strict global-order settle point.
        self.finalize_lag_high_water_ct = 0
        self.finalize_hol_blocked_slots_high_water_ct = 0
        self.prefill_ring_high_water_ct = 0
        self.verify_ring_high_water_ct = 0
        self.export_queue_high_water_ct = 0
        self.sidecar_write_rows_ct = 0
        self.sidecar_write_bytes_ct = 0
        self.sidecar_gather_rows_ct = 0
        self.sidecar_gather_bytes_ct = 0
        self.sidecar_write_lock_wait_ns_ct = 0
        self.sidecar_write_lock_hold_ns_ct = 0
        self.sidecar_write_copy_ns_ct = 0
        self.sidecar_gather_lock_wait_ns_ct = 0
        self.sidecar_gather_lock_hold_ns_ct = 0
        self.sidecar_gather_copy_ns_ct = 0
        self.sidecar_cow_rows_ct = 0
        self.sidecar_evicted_rows_ct = 0
        self.sidecar_released_rows_ct = 0
        self.sidecar_unleased_rows_ct = 0
        self.sidecar_capacity_miss_ct = 0
        self.sidecar_reader_conflict_miss_ct = 0
        self.sidecar_lease_conflict_miss_ct = 0
        self.sidecar_resident_rows_high_water_ct = 0
        self.sidecar_leased_rows_high_water_ct = 0
        self.sidecar_direct_gather_rows_ct = 0
        self.sidecar_direct_gather_bytes_ct = 0
        self.export_barrier_wait_ns_ct = 0
        self.export_gather_busy_ns_ct = 0
        self.sink_put_busy_ns_ct = 0
        # Critical-path counters are deliberately disjoint: for every export
        # job their components add up to export_critical_wall_ns_ct, with
        # async gather/put overlap in its own bucket (the legacy busy
        # counters above double-count overlap; kept for cross-version
        # comparison, removal tracked in PR #6).
        self.export_critical_jobs_ct = 0
        self.export_critical_wall_ns_ct = 0
        self.export_critical_queue_ns_ct = 0
        self.export_critical_finalize_ns_ct = 0
        self.export_critical_barrier_ns_ct = 0
        self.export_critical_gather_ns_ct = 0
        self.export_critical_put_ns_ct = 0
        self.export_critical_gather_put_overlap_ns_ct = 0
        self.export_critical_other_ns_ct = 0
        self.export_critical_max_ns_ct = 0
        self.export_critical_attribution_mismatch_ct = 0
        # Serving-thread spans identify the contention source without CUDA
        # synchronization and can be carried unchanged to RDMA/NUMA hosts.
        for name in (
            "serving_forward_hook",
            "serving_verify_hook",
            "serving_prefill_stage",
            "serving_verify_stage",
            "serving_finish_hook",
        ):
            setattr(self, f"{name}_calls_ct", 0)
            setattr(self, f"{name}_ns_ct", 0)
            setattr(self, f"{name}_max_ns_ct", 0)
        self.mooncake_batch_put_calls_ct = 0
        self.mooncake_batch_put_objects_attempted_ct = 0
        self.mooncake_batch_put_bytes_attempted_ct = 0
        self.mooncake_put_from_calls_ct = 0
        self.mooncake_put_from_bytes_attempted_ct = 0
        self.mooncake_payload_objects_ct = 0
        self.mooncake_payload_bytes_ct = 0
        self.mooncake_batch_partial_failure_miss_ct = 0
        self.mooncake_batch_is_exist_calls_ct = 0
        self.mooncake_batch_is_exist_keys_ct = 0
        self.mooncake_is_exist_calls_ct = 0
        self.mooncake_get_calls_ct = 0
        self.mooncake_batch_get_calls_ct = 0
        self.mooncake_batch_get_objects_ct = 0
        self.mooncake_batch_get_bytes_ct = 0
        self.mooncake_put_calls_ct = 0
        self.mooncake_put_bytes_ct = 0
        self.mooncake_registered_bytes_ct = 0
        self.prefix_lookup_rows_ct = 0
        self.prefix_reused_rows_ct = 0
        self.prefix_boundary_republished_rows_ct = 0
        self.suffix_published_rows_ct = 0
        self.prefix_index_stale_hit_ct = 0
        self.prefix_context_unsupported_miss_ct = 0
        self.segment_publish_failed_miss_ct = 0
        self.segment_exist_check_failed_miss_ct = 0
        self.segment_component_missing_miss_ct = 0
        self.sample_publish_failed_miss_ct = 0
        self.segment_orphan_ct = 0
        self.segment_object_count_ct = 0
        self.segment_bytes_written_ct = 0
        self.segment_boundary_read_bytes_ct = 0
        self.prefix_lane_wait_ns_ct = 0
        self.prefix_ready_queue_high_water_ct = 0
        self.prefix_gather_busy_ns_ct = 0
        self.prefix_put_busy_ns_ct = 0
        self.prefix_lane_gathering_ns_ct = 0
        self.prefix_lane_ready_ns_ct = 0
        self.prefix_lane_putting_ns_ct = 0
        self.prefix_lane_free_ns_ct = 0
        self.prefix_index_segments_ct = 0
        self.prefix_index_rows_ct = 0
        self.per_request_barrier_ct = 0
        self.fully_warm_global_barrier_ct = 0
        self.barrier_younger_seq_avoided_ct = 0
        self.stage_full_miss_ct = 0
        self.prefill_stage_full_miss_ct = 0
        self.verify_stage_full_miss_ct = 0
        self.attach_mismatch_miss_ct = 0
        self.aux_not_fresh_miss_ct = 0
        self.prefix_invalid_miss_ct = 0
        self.export_queue_full_miss_ct = 0
        self.export_timeout_miss_ct = 0
        self.duplicate_sample_miss_ct = 0
        self.sample_too_large_miss_ct = 0
        self.sink_put_failed_miss_ct = 0
        self.sink_unavailable_miss_ct = 0
        self.sink_reconnect_attempt_ct = 0
        self.sink_reconnect_success_ct = 0
        self.sink_probe_failed_ct = 0
        self.sink_initial_probe_timeout_ct = 0
        # Sample exported but its manifest entry failed: discoverable by
        # nothing, reclaimed by lease TTL (mooncake sink only).
        self.manifest_orphan_miss_ct = 0
        self.export_ok_ct = 0
        self.skipped_forward_ct = 0
        self.verify_rows_committed_ct = 0
        self.verify_twin_full_miss_ct = 0
        self.verify_oversize_miss_ct = 0
        self.finish_snapshot_batches_ct = 0
        self.finish_snapshot_requests_ct = 0
        self.finish_snapshot_rows_ct = 0
        self.finish_snapshot_d2h_bytes_ct = 0
        self.finish_snapshot_wait_ns_ct = 0
        self.finish_snapshot_grow_ct = 0
        self.finish_snapshot_batch_high_water_ct = 0
        self.finish_snapshot_failed_miss_ct = 0
        self.capture_degraded_ct = 0
        self.capture_recovered_ct = 0
        self.capture_degraded_drop_ct = 0
        self.capture_degraded_request_ct = 0
        self.capture_degraded_row_ct = 0
        self.capture_degraded_sample_miss_ct = 0
        self.capture_degraded_ns_ct = 0
        self.capture_degraded_sink_ct = 0
        self.capture_degraded_export_queue_ct = 0
        self.capture_degraded_prefill_ring_ct = 0
        self.capture_degraded_verify_ring_ct = 0
        self.capture_degraded_verify_twin_ct = 0
        self.capture_degraded_sidecar_ct = 0
        self.shutdown_admission_miss_ct = 0
        self.shutdown_finalize_timeout_miss_ct = 0
        self.shutdown_export_timeout_miss_ct = 0

    def bump(self, name: str, delta: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + delta)

    def observe_max(self, name: str, value: int) -> None:
        """Raise a monotonic high-water counter to value if needed."""
        with self._lock:
            setattr(self, name, max(getattr(self, name), value))

    def observe_duration(self, name: str, duration_ns: int) -> None:
        """Record one wall span with count/total/max in one lock section."""
        duration_ns = max(0, int(duration_ns))
        with self._lock:
            calls_name = f"{name}_calls_ct"
            total_name = f"{name}_ns_ct"
            max_name = f"{name}_max_ns_ct"
            setattr(self, calls_name, getattr(self, calls_name) + 1)
            setattr(self, total_name, getattr(self, total_name) + duration_ns)
            setattr(self, max_name, max(getattr(self, max_name), duration_ns))

    def record_export_critical_path(self, components: Dict[str, int]) -> None:
        """Accumulate one already-disjoint exporter critical-path attribution.

        Attribution is observational and must never fail an export. The trace
        normally closes exactly; defensively reconcile malformed/missing
        values while exposing a mismatch counter for the harness.
        """
        wall_ns = max(0, int(components.get("wall_ns", 0)))
        names = (
            "queue",
            "finalize",
            "barrier",
            "gather",
            "put",
            "gather_put_overlap",
            "other",
        )
        values = {name: max(0, int(components.get(name, 0))) for name in names}
        accounted = sum(values.values())
        mismatch = accounted != wall_ns
        if accounted != wall_ns:
            if accounted < wall_ns:
                values["other"] += wall_ns - accounted
            else:
                overflow = accounted - wall_ns
                # Prefer trimming the residual bucket, then the post-barrier
                # buckets. This branch is only a defensive guard against a
                # trace bug; the mismatch counter makes any use visible.
                for name in (
                    "other",
                    "gather_put_overlap",
                    "put",
                    "gather",
                    "barrier",
                    "finalize",
                    "queue",
                ):
                    removed = min(values[name], overflow)
                    values[name] -= removed
                    overflow -= removed
                    if overflow == 0:
                        break
        with self._lock:
            self.export_critical_jobs_ct += 1
            self.export_critical_attribution_mismatch_ct += int(mismatch)
            self.export_critical_wall_ns_ct += wall_ns
            self.export_critical_max_ns_ct = max(
                self.export_critical_max_ns_ct, wall_ns
            )
            for name in names:
                attr = f"export_critical_{name}_ns_ct"
                setattr(self, attr, getattr(self, attr) + values[name])

    def critical_path_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            wall_ns = self.export_critical_wall_ns_ct
            jobs = self.export_critical_jobs_ct
            max_ns = self.export_critical_max_ns_ct
            components = {
                name: getattr(self, f"export_critical_{name}_ns_ct")
                for name in (
                    "queue",
                    "finalize",
                    "barrier",
                    "gather",
                    "put",
                    "gather_put_overlap",
                    "other",
                )
            }
        denominator = max(1, wall_ns)
        return {
            "jobs": jobs,
            "wall_ns": wall_ns,
            "max_job_ns": max_ns,
            "components_ns": components,
            "components_fraction": {
                name: value / denominator for name, value in components.items()
            },
        }

    def serving_span_snapshot(self) -> Dict[str, Dict[str, int]]:
        names = (
            "serving_forward_hook",
            "serving_verify_hook",
            "serving_prefill_stage",
            "serving_verify_stage",
            "serving_finish_hook",
        )
        with self._lock:
            return {
                name: {
                    "calls": getattr(self, f"{name}_calls_ct"),
                    "wall_ns": getattr(self, f"{name}_ns_ct"),
                    "max_ns": getattr(self, f"{name}_max_ns_ct"),
                }
                for name in names
            }

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {k: v for k, v in vars(self).items() if k.endswith("_ct")}

    def log(self, prefix: str) -> None:
        logger.info("%s %s", prefix, self.snapshot())


class HiddenWallSpanRecorder:
    """Bounded global interval recorder used only for barrier attribution.

    Finalize runs on a different thread from export. A job waiting at its
    barrier can therefore distinguish time during which finalize was actually
    executing from residual queue/poll wait. Intervals are monotonic-clock
    values and reads never wait for a worker or synchronize CUDA.
    """

    def __init__(self, *, retention_s: float = 120.0, max_spans: int = 65536):
        self._lock = threading.Lock()
        self._retention_ns = int(retention_s * 1e9)
        self._spans: deque[Tuple[int, int]] = deque(maxlen=max_spans)

    def record_finalize(self, start_ns: int, end_ns: int) -> None:
        if end_ns <= start_ns:
            return
        cutoff = end_ns - self._retention_ns
        with self._lock:
            self._spans.append((start_ns, end_ns))
            while self._spans and self._spans[0][1] < cutoff:
                self._spans.popleft()

    def finalize_intervals(self, start_ns: int, end_ns: int) -> List[Tuple[int, int]]:
        if end_ns <= start_ns:
            return []
        with self._lock:
            spans = list(self._spans)
        return [
            (max(start_ns, left), min(end_ns, right))
            for left, right in spans
            if right > start_ns and left < end_ns
        ]


class HiddenCapturePressureController:
    """Hysteretic, non-blocking admission gate with auditable time windows."""

    _REASON_COUNTERS = {
        "sink": "capture_degraded_sink_ct",
        "export_queue": "capture_degraded_export_queue_ct",
        "prefill_ring": "capture_degraded_prefill_ring_ct",
        "verify_ring": "capture_degraded_verify_ring_ct",
        "verify_twin": "capture_degraded_verify_twin_ct",
        "sidecar": "capture_degraded_sidecar_ct",
    }

    def __init__(
        self,
        *,
        stats: HiddenCaptureStats,
        high_watermark: float,
        recover_watermark: float,
        min_degraded_s: float,
        max_windows: int = 64,
    ) -> None:
        if not 0.0 <= recover_watermark < high_watermark <= 1.0:
            raise ValueError(
                "capture pressure watermarks require 0 <= recover < high <= 1"
            )
        self.stats = stats
        self.high_watermark = float(high_watermark)
        self.recover_watermark = float(recover_watermark)
        self.min_degraded_ns = int(max(0.0, min_degraded_s) * 1e9)
        self._lock = threading.Lock()
        self._degraded = False
        self._started_mono_ns = 0
        self._current: Optional[Dict[str, Any]] = None
        self._windows: deque[Dict[str, Any]] = deque(maxlen=max_windows)

    def refresh(self, signals: Dict[str, float]) -> bool:
        now_mono_ns = time.monotonic_ns()
        now_wall_ns = time.time_ns()
        high_reasons = {
            name
            for name, value in signals.items()
            if float(value) >= self.high_watermark
        }
        recovered = all(
            float(value) <= self.recover_watermark for value in signals.values()
        )
        with self._lock:
            if not self._degraded and high_reasons:
                self._degraded = True
                self._started_mono_ns = now_mono_ns
                self._current = {
                    "start_wall_ns": now_wall_ns,
                    "end_wall_ns": None,
                    "reasons": sorted(high_reasons),
                    "drop_calls": 0,
                    "dropped_requests": 0,
                    "dropped_rows": 0,
                    "phase_drop_calls": {},
                }
                self.stats.bump("capture_degraded_ct")
                for reason in high_reasons:
                    counter = self._REASON_COUNTERS.get(reason)
                    if counter is not None:
                        self.stats.bump(counter)
                logger.warning(
                    "hidden capture entered degraded mode: reasons=%s",
                    sorted(high_reasons),
                )
            elif self._degraded:
                current_reasons = set(self._current["reasons"])
                for reason in high_reasons - current_reasons:
                    counter = self._REASON_COUNTERS.get(reason)
                    if counter is not None:
                        self.stats.bump(counter)
                current_reasons.update(high_reasons)
                self._current["reasons"] = sorted(current_reasons)
                if (
                    recovered
                    and now_mono_ns - self._started_mono_ns >= self.min_degraded_ns
                ):
                    duration_ns = now_mono_ns - self._started_mono_ns
                    self._current["end_wall_ns"] = now_wall_ns
                    self._current["duration_ns"] = duration_ns
                    self._windows.append(dict(self._current))
                    self._current = None
                    self._degraded = False
                    self.stats.bump("capture_recovered_ct")
                    self.stats.bump("capture_degraded_ns_ct", duration_ns)
                    logger.info(
                        "hidden capture recovered after %.3fs", duration_ns / 1e9
                    )
            return not self._degraded

    def record_drop(
        self, *, phase: str, requests: int, rows: int, sample_misses: int = 0
    ) -> None:
        requests = max(0, int(requests))
        rows = max(0, int(rows))
        with self._lock:
            if self._current is not None:
                self._current["drop_calls"] += 1
                self._current["dropped_requests"] += requests
                self._current["dropped_rows"] += rows
                phase_calls = self._current["phase_drop_calls"]
                phase_calls[phase] = phase_calls.get(phase, 0) + 1
        self.stats.bump("capture_degraded_drop_ct")
        self.stats.bump("capture_degraded_request_ct", requests)
        self.stats.bump("capture_degraded_row_ct", rows)
        if sample_misses:
            self.stats.bump("capture_degraded_sample_miss_ct", sample_misses)

    def snapshot(self) -> Dict[str, Any]:
        now_mono_ns = time.monotonic_ns()
        with self._lock:
            current = dict(self._current) if self._current is not None else None
            if current is not None:
                current["duration_ns"] = now_mono_ns - self._started_mono_ns
            return {
                "degraded": self._degraded,
                "high_watermark": self.high_watermark,
                "recover_watermark": self.recover_watermark,
                "min_degraded_ns": self.min_degraded_ns,
                "current_window": current,
                "recent_windows": list(self._windows),
            }

    def close(self) -> None:
        """Seal an open window for final stats without calling it recovery."""
        now_mono_ns = time.monotonic_ns()
        now_wall_ns = time.time_ns()
        with self._lock:
            if not self._degraded or self._current is None:
                return
            duration_ns = now_mono_ns - self._started_mono_ns
            self._current["end_wall_ns"] = now_wall_ns
            self._current["duration_ns"] = duration_ns
            self._current["closed_with_process"] = True
            self._windows.append(dict(self._current))
            self._current = None
            self._degraded = False
        self.stats.bump("capture_degraded_ns_ct", duration_ns)


class _StagingSlot(msgspec.Struct):
    index: int
    event: Any
    # Prefill slots own four pinned tensors. Verify slots own one pinned
    # byte blob instead; aux/last/cache_loc/tokens/commit_lens become
    # per-step views carved into it at enqueue (see verify_blob_views).
    aux: Optional[torch.Tensor] = None  # pinned [slot_tokens, aux_width]
    last: Optional[torch.Tensor] = None  # pinned [slot_tokens, last_width]
    cache_loc: Optional[torch.Tensor] = None  # pinned [slot_tokens] int64
    tokens: Optional[torch.Tensor] = None  # pinned [slot_tokens] int64
    blob: Optional[torch.Tensor] = None  # pinned [blob_bytes] uint8
    # True once event.query() returned True; a fired event never un-fires,
    # so it is queried at most once per enqueue.
    event_done: bool = False
    num_rows: int = 0
    ring_seq: int = -1
    # Prefill slots: [(rid, start_row, end_row)] within [0, num_rows).
    req_ranges: List[Tuple[str, int, int]] = []
    # Verify slots: request i owns rows [i*stride, (i+1)*stride); only the
    # first commit_lens[i] of them are committed (selected at finalize).
    kind: str = "prefill"
    commit_lens: Optional[torch.Tensor] = None  # view into blob [num_reqs] int32
    rids: List[str] = []
    stride: int = 0
    num_reqs: int = 0
    # Device twin borrowed for this verify enqueue; released at finalize.
    twin: Optional[_DeviceTwin] = None
    timing_accounted: bool = False


class _DeviceTwin(msgspec.Struct):
    """Capture-owned HBM blob for one verify step's strided window.

    The worker packs the (graph/persistent) verify outputs into a twin on the
    forward stream — same-stream ordering IS the overwrite fence: step t+1's
    replay queues behind the pack, so the persistent source can't be
    rewritten under the copy, and the forward stream never waits on D2H
    (which reads the twin, not the source).

    All packed fields (commit_lens header plus payload sections) live in one
    contiguous byte blob laid out per step by ``verify_blob_views``, so the
    whole step travels to the pinned slot in a single length-bounded copy.
    ``commit_offsets``/``verify_offsets`` are pack-kernel scratch only and
    never leave the device.
    """

    index: int
    blob: torch.Tensor  # [blob_bytes] uint8 device
    commit_offsets: torch.Tensor  # [max_reqs] int32 device (pack scratch)
    verify_offsets: torch.Tensor  # [max_reqs] int32 device (pack scratch)
    fence_event: Any = None  # recorded on the forward stream after the pack
    pack_start_event: Any = None
    borrowed_ns: int = 0


class DeviceTwinPool:
    """Small pool of verify twins; empty pool => whole-step capture miss."""

    def __init__(
        self,
        *,
        num_twins: int,
        twin_tokens: int,
        max_reqs: int,
        aux_width: int,
        last_width: int,
        dtype: torch.dtype,
        device: str = "cuda",
        use_cuda_events: bool = True,
        stats: Optional[HiddenCaptureStats] = None,
    ) -> None:
        self.twin_tokens = twin_tokens
        self.max_reqs = max_reqs
        self.num_twins = num_twins
        self.aux_width = aux_width
        self.last_width = last_width
        self.dtype = dtype
        self.blob_bytes = verify_blob_bytes(
            bs=max_reqs,
            n_rows=twin_tokens,
            aux_width=aux_width,
            last_width=last_width,
            dtype=dtype,
        )
        self._lock = threading.Lock()
        self._free: List[_DeviceTwin] = []
        self._in_use = 0
        self._stats = stats
        for i in range(num_twins):
            self._free.append(
                _DeviceTwin(
                    index=i,
                    blob=torch.empty(
                        (self.blob_bytes,), dtype=torch.uint8, device=device
                    ),
                    commit_offsets=torch.empty(
                        (max_reqs,), dtype=torch.int32, device=device
                    ),
                    verify_offsets=torch.empty(
                        (max_reqs,), dtype=torch.int32, device=device
                    ),
                    fence_event=(
                        torch.cuda.Event(enable_timing=True)
                        if use_cuda_events
                        else _NullEvent()
                    ),
                    pack_start_event=(
                        torch.cuda.Event(enable_timing=True)
                        if use_cuda_events
                        else _NullEvent()
                    ),
                )
            )
        self.allocated_bytes = sum(
            tensor.numel() * tensor.element_size()
            for twin in self._free
            for tensor in (twin.blob, twin.commit_offsets, twin.verify_offsets)
        )
        size_mb = (num_twins * self.blob_bytes) / (1024 * 1024)
        logger.info(
            "Hidden capture DeviceTwinPool allocated: %d twins x %d tokens, "
            "%.0f MB HBM",
            num_twins,
            twin_tokens,
            size_mb,
        )

    def views(self, twin: _DeviceTwin, *, bs: int, stride: int) -> _VerifyBlobViews:
        return verify_blob_views(
            twin.blob,
            bs=bs,
            stride=stride,
            aux_width=self.aux_width,
            last_width=self.last_width,
            dtype=self.dtype,
        )

    def try_acquire(self) -> Optional[_DeviceTwin]:
        with self._lock:
            if not self._free:
                return None
            twin = self._free.pop()
            self._in_use += 1
            twin.borrowed_ns = time.monotonic_ns()
            if self._stats is not None:
                self._stats.observe_max("verify_twin_high_water_ct", self._in_use)
            return twin

    def release(self, twin: _DeviceTwin) -> None:
        with self._lock:
            self._in_use -= 1
            twin.borrowed_ns = 0
            self._free.append(twin)

    @property
    def in_use_count(self) -> int:
        with self._lock:
            return self._in_use


class StagingSeqCounter:
    """Monotonic enqueue sequence shared across staging rings.

    Prefill and verify stage through separate rings (capacity isolation), but
    the export barrier must stay one number and finalize must settle slots in
    enqueue order. A shared counter gives every slot a global seq. All copies
    are submitted by the scheduler thread on one capture stream, so the seq
    order is also the stream submission order; the finalizer still gates on
    the seq (not on readiness) so the settlement order is explicit rather
    than inferred.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self.last_enqueued_seq = -1

    def next_seq(self) -> int:
        with self._lock:
            seq = self._next
            self._next += 1
            self.last_enqueued_seq = seq
            return seq


class HiddenStagingRing:
    """Fixed-slot pinned staging for async D2H, FIFO-finalized.

    ``try_acquire`` / ``release`` are lock-protected; enqueue records a
    per-slot CUDA event on the *current* stream (the scheduler's copy stream
    in overlap scheduling) so the finalize thread can poll completion without
    synchronizing.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        slot_tokens: int,
        aux_width: int,
        last_width: int,
        dtype: torch.dtype,
        pin_memory: bool = True,
        use_cuda_events: bool = True,
        seq_counter: Optional[StagingSeqCounter] = None,
        stats: Optional[HiddenCaptureStats] = None,
        stats_prefix: str = "prefill",
        max_verify_reqs: int = 0,
    ) -> None:
        self.slot_tokens = slot_tokens
        self.max_verify_reqs = max_verify_reqs
        self._aux_width = aux_width
        self._last_width = last_width
        self._dtype = dtype
        self._lock = threading.Lock()
        self._free: deque[_StagingSlot] = deque()
        self._inflight: deque[_StagingSlot] = deque()
        self._seq = seq_counter if seq_counter is not None else StagingSeqCounter()
        self._stats = stats
        self._stats_prefix = stats_prefix
        self.num_slots = num_slots

        def _event() -> Any:
            return torch.cuda.Event() if use_cuda_events else _NullEvent()

        if max_verify_reqs > 0:
            # Verify slots: one pinned byte blob per slot; typed views are
            # carved per step by enqueue_verify_compact.
            blob_bytes = verify_blob_bytes(
                bs=max_verify_reqs,
                n_rows=slot_tokens,
                aux_width=aux_width,
                last_width=last_width,
                dtype=dtype,
            )
            for i in range(num_slots):
                self._free.append(
                    _StagingSlot(
                        index=i,
                        event=_event(),
                        blob=torch.empty(
                            (blob_bytes,), dtype=torch.uint8, pin_memory=pin_memory
                        ),
                    )
                )
            self.allocated_bytes = num_slots * blob_bytes
        else:

            def _pinned(shape: Tuple[int, ...], dt: torch.dtype) -> torch.Tensor:
                return torch.empty(shape, dtype=dt, pin_memory=pin_memory)

            for i in range(num_slots):
                self._free.append(
                    _StagingSlot(
                        index=i,
                        event=_event(),
                        aux=_pinned((slot_tokens, aux_width), dtype),
                        last=_pinned((slot_tokens, last_width), dtype),
                        cache_loc=_pinned((slot_tokens,), torch.int64),
                        tokens=_pinned((slot_tokens,), torch.int64),
                    )
                )
            self.allocated_bytes = sum(
                tensor.numel() * tensor.element_size()
                for slot in self._free
                for tensor in (slot.aux, slot.last, slot.cache_loc, slot.tokens)
            )
        size_gb = self.allocated_bytes / _GB
        logger.info(
            "HiddenStagingRing allocated: %d slots x %d tokens, pinned %.2f GB",
            num_slots,
            slot_tokens,
            size_gb,
        )

    def try_acquire(self, num_slots: int) -> Optional[List[_StagingSlot]]:
        """Atomically take ``num_slots`` free slots, or None (=> caller misses)."""
        with self._lock:
            if len(self._free) < num_slots:
                return None
            return [self._free.popleft() for _ in range(num_slots)]

    def peek_inflight_seq(self) -> Optional[int]:
        """Sequence of the oldest in-flight slot (for cross-ring merging)."""
        with self._lock:
            return self._inflight[0].ring_seq if self._inflight else None

    def enqueue_verify_compact(
        self,
        slot: _StagingSlot,
        *,
        twin: _DeviceTwin,
        rids: List[str],
        stride: int,
        num_reqs: int,
    ) -> int:
        """Issue the single async D2H of a packed twin blob and commit the
        slot in-flight.

        Runs on the caller's current stream (the capture stream, already
        fenced behind the twin's pack). One upper-bound-length byte copy
        moves commit_lens and every payload section together; one event
        gates readiness. The committed row count is derived by the finalizer
        from the pinned commit_lens — nothing here reads device data.

        Failure contract: every operation that can raise precedes the ring
        registration, so on exception the slot is NOT in flight and the
        caller still owns both the slot and the twin (drop-sample cleanup).
        """
        views = verify_blob_views(
            slot.blob,
            bs=num_reqs,
            stride=stride,
            aux_width=self._aux_width,
            last_width=self._last_width,
            dtype=self._dtype,
        )
        slot.blob[: views.copy_bytes].copy_(
            twin.blob[: views.copy_bytes], non_blocking=True
        )
        slot.event.record()

        slot.kind = "verify_compact"
        slot.commit_lens = views.commit_lens
        slot.aux = views.aux
        slot.last = views.last
        slot.cache_loc = views.cache_loc
        slot.tokens = views.tokens
        slot.num_rows = 0  # derived from commit_lens at finalize
        slot.req_ranges = []
        slot.rids = rids
        slot.stride = stride
        slot.num_reqs = num_reqs
        slot.twin = twin
        slot.event_done = False
        slot.timing_accounted = False
        with self._lock:
            slot.ring_seq = self._seq.next_seq()
            self._inflight.append(slot)
            inflight = len(self._inflight)
        self._observe_high_water(inflight)
        return slot.ring_seq

    def enqueue_segment(
        self,
        slot: _StagingSlot,
        *,
        aux_rows: torch.Tensor,
        last_rows: torch.Tensor,
        cache_locs: torch.Tensor,
        tokens: torch.Tensor,
        req_ranges: List[Tuple[str, int, int]],
    ) -> int:
        """Issue the async D2H copies for one segment and commit it in-flight.

        Runs on the caller's current stream. Mirrors ``_async_d2h``:
        non-blocking copies into pinned memory plus ``record_stream`` on the
        sources so the caching allocator can't recycle them before the copy
        stream drains.
        """
        num_rows = aux_rows.shape[0]
        # Cast index/token tensors on-device first so every D2H below is a
        # same-dtype copy; a dtype-converting async D2H would route through an
        # allocator temp that record_stream can't see.
        cache_locs = cache_locs.to(torch.int64)
        tokens = tokens.to(torch.int64)
        slot.aux[:num_rows].copy_(aux_rows, non_blocking=True)
        slot.last[:num_rows].copy_(last_rows, non_blocking=True)
        slot.cache_loc[:num_rows].copy_(cache_locs, non_blocking=True)
        slot.tokens[:num_rows].copy_(tokens, non_blocking=True)
        if aux_rows.is_cuda:
            stream = torch.cuda.current_stream(aux_rows.device)
            for src in (aux_rows, last_rows, cache_locs, tokens):
                src.record_stream(stream)
        slot.event.record()

        slot.num_rows = num_rows
        slot.req_ranges = req_ranges
        slot.event_done = False
        with self._lock:
            slot.ring_seq = self._seq.next_seq()
            self._inflight.append(slot)
            inflight = len(self._inflight)
        self._observe_high_water(inflight)
        return slot.ring_seq

    def _observe_high_water(self, inflight: int) -> None:
        if self._stats is not None:
            self._stats.observe_max(
                f"{self._stats_prefix}_ring_high_water_ct", inflight
            )

    def _slot_ready_locked(self, slot: _StagingSlot) -> bool:
        """Query a slot's copy event at most once after it fires.

        A recorded CUDA event never un-fires, so the first True is memoized
        in ``event_done`` and later polls cost a flag read instead of a
        ``cudaEventQuery`` call.
        """
        if slot.event_done:
            return True
        if slot.event.query():
            slot.event_done = True
            return True
        return False

    def pop_ready(self) -> Optional[_StagingSlot]:
        """Oldest in-flight slot whose D2H completed, preserving FIFO order."""
        with self._lock:
            if not self._inflight:
                return None
            slot = self._inflight[0]
            if not self._slot_ready_locked(slot):
                return None
            self._account_verify_completion_locked(slot)
            return self._inflight.popleft()

    @property
    def last_enqueued_seq(self) -> int:
        return self._seq.last_enqueued_seq

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    def reap_ready_twins(self, twin_pool: DeviceTwinPool) -> int:
        """Release device twins whose D2H into pinned memory has completed.

        A twin's data is dead the moment its slot's copy event fires; waiting
        for the slot to reach the finalize queue head (and for finalize's
        sidecar memcpy — hundreds of MB for prefill slots) holds twins for
        tens of ms longer than needed and starves verify capture. The
        scheduler is the sole submitter on the capture stream, so event
        completion order equals enqueue order within this ring: the first
        pending slot ends the scan, and everything before it is complete.
        """
        reaped = 0
        with self._lock:
            for slot in self._inflight:
                if not self._slot_ready_locked(slot):
                    break
                if slot.twin is None:
                    continue
                self._account_verify_completion_locked(slot)
                twin_pool.release(slot.twin)
                slot.twin = None
                reaped += 1
        return reaped

    def count_ready_prefix(self) -> int:
        """Completed oldest-first in-flight slots in this ring.

        Single-submitter ordering makes readiness a prefix property, and
        ``event_done`` memoization means at most one new ``cudaEventQuery``
        per call. Used by the finalizer's head-of-line observability: slots
        counted here while the *other* ring holds the pending global head are
        completed work pinned behind strict global-order settlement.
        """
        with self._lock:
            ready = 0
            for slot in self._inflight:
                if not self._slot_ready_locked(slot):
                    break
                ready += 1
            return ready

    def _account_verify_completion_locked(self, slot: _StagingSlot) -> None:
        if (
            self._stats is None
            or not slot.kind.startswith("verify")
            or slot.timing_accounted
        ):
            return
        twin = slot.twin
        if twin is not None:
            if twin.borrowed_ns:
                self._stats.bump(
                    "verify_twin_hold_ns_ct",
                    max(0, time.monotonic_ns() - twin.borrowed_ns),
                )
            if hasattr(twin.pack_start_event, "elapsed_time"):
                try:
                    self._stats.bump(
                        "verify_pack_ns_ct",
                        int(twin.pack_start_event.elapsed_time(twin.fence_event) * 1e6),
                    )
                except Exception:
                    logger.debug("verify pack timing event unavailable", exc_info=True)
        slot.timing_accounted = True

    def release(self, slot: _StagingSlot) -> None:
        slot.num_rows = 0
        slot.ring_seq = -1
        slot.req_ranges = []
        slot.kind = "prefill"
        slot.rids = []
        slot.stride = 0
        slot.num_reqs = 0
        slot.twin = None
        slot.event_done = False
        slot.timing_accounted = False
        if slot.blob is not None:
            # Verify slot: drop the per-step views into the blob.
            slot.aux = None
            slot.last = None
            slot.cache_loc = None
            slot.tokens = None
            slot.commit_lens = None
        with self._lock:
            self._free.append(slot)


class SidecarCapacityError(RuntimeError):
    """The bounded sparse sidecar had no unleased/non-reader row to reuse."""


class HiddenHostSidecar:
    """Capacity-bounded sparse payload cache keyed by logical KV slot.

    Generation/token identity maps are kept at KV-pool cardinality (small),
    but payloads live in a compact physical arena of bounded token budget —
    never one row per KV slot. The mutex protects only logical->physical
    metadata and reader pins:

    * a reader validates and pins physical rows under the mutex, copies hidden
      payloads after releasing it, then unpins;
    * the sole finalize writer reuses an unpinned/unleased row, or
      copy-on-writes when a reader or an export lease owns the current row. It
      never waits for either;
    * exact ``(slot, generation)`` versions remain leased until export settles,
      so serving may reuse a KV slot without corrupting a queued sample;
    * full capacity with every candidate pinned/leased is an immediate
      whole-slot capture miss, never serving backpressure.

    Pinned physical rows are immutable, which is the synchronization fence
    required on weakly ordered hosts (ARM: GH200/GB200). ``slot_gen`` is the
    ABA identity generation (even settled, odd in-place write window, zero
    never written) — it validates identity, it is not torn-read protection.

    Identity model: rows are keyed by KV token-slot index. Requests hold
    their slots until ``release_kv_cache`` (after the finish hook) and
    warm-prefix slots are radix-locked while alive, so the race window is
    only in-flight staging rows and slot reuse after finish. Own rows are
    validated against the exact generation recorded at finalize; warm rows
    written by another request are validated by token-id match plus an even,
    unchanged generation across the read (probabilistic if the same token
    rewrites a slot — the prefix protocol therefore reads warm boundaries
    from already-published immutable segments instead).
    """

    def __init__(
        self,
        *,
        num_slots: int,
        aux_width: int,
        last_width: int,
        dtype: torch.dtype,
        stats: Optional[HiddenCaptureStats] = None,
        capacity_tokens: Optional[int] = None,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("sidecar num_slots must be positive")
        capacity_tokens = num_slots if capacity_tokens is None else int(capacity_tokens)
        if capacity_tokens <= 0 or capacity_tokens > num_slots:
            raise ValueError(
                "sidecar capacity_tokens must be in [1, num_slots], got "
                f"{capacity_tokens} for {num_slots}"
            )
        self._lock = threading.Lock()
        self._stats = stats
        self.num_slots = int(num_slots)
        self.capacity_tokens = capacity_tokens
        self.aux_buf = torch.empty((capacity_tokens, aux_width), dtype=dtype)
        self.last_buf = torch.empty((capacity_tokens, last_width), dtype=dtype)
        self.token_id_map = torch.full((num_slots,), -1, dtype=torch.int32)
        self.slot_gen = torch.zeros((num_slots,), dtype=torch.int64)
        # ``_slot_to_row`` is a dense logical-slot index for the newest
        # version used by warm-prefix validation.  Keeping current versions in
        # a Python ``dict[(slot, generation)]`` cost one tuple allocation and
        # two hash-table operations per captured token, and showed up as the
        # dominant finalizer GIL holder under nsys. ``_version_to_row`` now
        # contains only older COW generations that still need exact lookup.
        self._slot_to_row = [-1] * num_slots
        self._version_to_row: Dict[Tuple[int, int], int] = {}
        self._row_slot = [-1] * capacity_tokens
        self._row_gen = [0] * capacity_tokens
        self._row_leases = [0] * capacity_tokens
        self._row_readers = [0] * capacity_tokens
        self._row_writing = [False] * capacity_tokens
        self._free_rows = set(range(capacity_tokens))
        self._evict_cursor = 0
        self._resident_rows = 0
        self._leased_rows = 0
        self._reader_pins = 0
        self._reader_rows = 0
        self._writing_rows = 0
        # Exact union of rows blocked from eviction by a lease, reader, or
        # in-progress write.  The old sum of the three populations counted a
        # leased row again while Mooncake gathered it, causing false 100%
        # pressure and long degraded-capture windows.
        self._unavailable_rows = 0
        self.allocated_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.aux_buf,
                self.last_buf,
                self.token_id_map,
                self.slot_gen,
            )
        )
        payload_gb = (
            self.aux_buf.numel() * self.aux_buf.element_size()
            + self.last_buf.numel() * self.last_buf.element_size()
        ) / _GB
        logger.info(
            "HiddenHostSidecar allocated: logical_slots=%d, capacity_tokens=%d, "
            "aux_width=%d, last_width=%d, %.2f GB pageable payload",
            num_slots,
            capacity_tokens,
            aux_width,
            last_width,
            payload_gb,
        )

    def write_rows(
        self,
        *,
        slots: torch.Tensor,
        aux_rows: torch.Tensor,
        last_rows: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Copy rows outside the metadata lock and publish atomically.

        The finalize worker is the sole caller. A same-slot reader causes COW;
        if the bounded arena cannot supply a row, ``SidecarCapacityError`` is
        raised immediately and the finalizer marks the affected requests miss.
        """
        rows = int(slots.numel())
        if rows == 0:
            return torch.empty((0,), dtype=torch.int64)
        if (
            slots.device.type != "cpu"
            or aux_rows.device.type != "cpu"
            or last_rows.device.type != "cpu"
            or tokens.device.type != "cpu"
        ):
            raise ValueError("sidecar write tensors must be on CPU")
        if (
            aux_rows.shape != (rows, self.aux_buf.shape[1])
            or last_rows.shape != (rows, self.last_buf.shape[1])
            or tokens.numel() != rows
        ):
            raise ValueError("sidecar write tensors have incompatible shapes")
        slot_values = [int(slot) for slot in slots.tolist()]
        if len(set(slot_values)) != rows:
            raise ValueError("sidecar write slots must be unique")
        if any(slot < 0 or slot >= self.num_slots for slot in slot_values):
            raise IndexError("sidecar logical slot out of range")
        tokens_i32 = tokens.to(dtype=torch.int32).contiguous()

        wait_started_ns = time.monotonic_ns()
        self._lock.acquire()
        lock_started_ns = time.monotonic_ns()
        plans = []
        evicted_count = 0
        try:
            current_rows = [self._slot_to_row[slot] for slot in slot_values]
            target_rows = {row for row in current_rows if row >= 0}
            physical_values = list(current_rows)
            allocation_indices = []
            for index, current in enumerate(current_rows):
                if (
                    current >= 0
                    and self._row_leases[current] == 0
                    and self._row_readers[current] == 0
                    and not self._row_writing[current]
                ):
                    continue
                allocation_indices.append(index)
            reserved = self._reserve_rows_locked(
                len(allocation_indices), protected_rows=target_rows
            )
            evicted_slots = [None] * rows
            for index, (new_row, evicted_slot) in zip(allocation_indices, reserved):
                physical_values[index] = new_row
                evicted_slots[index] = evicted_slot
                if evicted_slot is not None:
                    evicted_count += 1

            # One tensor gather replaces a per-row Python tensor-scalar
            # conversion — per-row conversions dominate finalizer GIL time at
            # scale (measurements: PR #3).
            old_gens = self.slot_gen[slots].tolist()
            new_gens = []
            in_place_slots = []
            in_place_odd_gens = []
            for slot, physical_row, old_row, evicted_slot, old_gen in zip(
                slot_values,
                physical_values,
                current_rows,
                evicted_slots,
                old_gens,
            ):
                new_gen = old_gen + (1 if old_gen % 2 else 2)
                self._row_writing[physical_row] = True
                if physical_row == old_row:
                    in_place_slots.append(slot)
                    in_place_odd_gens.append(new_gen - 1)
                plans.append(
                    (slot, physical_row, old_row, evicted_slot, old_gen, new_gen)
                )
                new_gens.append(new_gen)
            self._writing_rows += rows
            self._unavailable_rows += rows
            if in_place_slots:
                self.slot_gen[torch.tensor(in_place_slots, dtype=torch.long)] = (
                    torch.tensor(in_place_odd_gens, dtype=torch.int64)
                )
        finally:
            reserve_hold_ns = time.monotonic_ns() - lock_started_ns
            self._lock.release()

        physical_rows = torch.tensor(physical_values, dtype=torch.long)
        new_gens_tensor = torch.tensor(new_gens, dtype=torch.int64)
        copy_started_ns = time.monotonic_ns()
        try:
            self.aux_buf.index_copy_(0, physical_rows, aux_rows)
            self.last_buf.index_copy_(0, physical_rows, last_rows)
        except BaseException:
            self._abort_write(plans)
            raise
        copy_ns = time.monotonic_ns() - copy_started_ns

        publish_wait_ns = time.monotonic_ns()
        with self._lock:
            publish_started_ns = time.monotonic_ns()
            for (
                slot,
                physical_row,
                old_row,
                _evicted_slot,
                _old_gen,
                _new_gen,
            ) in plans:
                if physical_row != old_row:
                    # Keep an older leased/current-reader generation resident;
                    # exact-generation export may still need it after this KV
                    # slot has been reused by serving.
                    if (
                        old_row >= 0
                        and self._row_slot[old_row] == slot
                        and self._row_gen[old_row] == _old_gen
                    ):
                        self._version_to_row[(slot, _old_gen)] = old_row
                    self._resident_rows += 1
                self._slot_to_row[slot] = physical_row
                self._row_slot[physical_row] = slot
                self._row_gen[physical_row] = _new_gen
                self._row_writing[physical_row] = False
            self._writing_rows -= rows
            self._unavailable_rows -= rows
            self.token_id_map[slots] = tokens_i32
            self.slot_gen[slots] = new_gens_tensor
            resident = self._resident_rows
            publish_hold_ns = time.monotonic_ns() - publish_started_ns

        if self._stats is not None:
            payload_bytes = rows * self._payload_row_bytes
            self._stats.bump(
                "sidecar_write_lock_wait_ns_ct",
                (lock_started_ns - wait_started_ns)
                + (publish_started_ns - publish_wait_ns),
            )
            self._stats.bump(
                "sidecar_write_lock_hold_ns_ct", reserve_hold_ns + publish_hold_ns
            )
            self._stats.bump("sidecar_write_copy_ns_ct", copy_ns)
            self._stats.bump("sidecar_write_rows_ct", rows)
            self._stats.bump("sidecar_write_bytes_ct", payload_bytes)
            self._stats.bump(
                "sidecar_cow_rows_ct",
                sum(old >= 0 and physical != old for _, physical, old, *_ in plans),
            )
            self._stats.bump("sidecar_evicted_rows_ct", evicted_count)
            self._stats.observe_max("sidecar_resident_rows_high_water_ct", resident)
        return new_gens_tensor

    @property
    def _payload_row_bytes(self) -> int:
        return (
            self.aux_buf.shape[1] * self.aux_buf.element_size()
            + self.last_buf.shape[1] * self.last_buf.element_size()
        )

    def _reserve_rows_locked(
        self, count: int, *, protected_rows: set
    ) -> List[Tuple[int, Optional[int]]]:
        if count <= 0:
            return []
        free_candidates = []
        for row in self._free_rows:
            free_candidates.append(row)
            if len(free_candidates) == count:
                break
        candidates = [(row, None) for row in free_candidates]
        selected = set(protected_rows)
        selected.update(free_candidates)
        scanned = 0
        while len(candidates) < count and scanned < self.capacity_tokens:
            row = self._evict_cursor
            self._evict_cursor = (self._evict_cursor + 1) % self.capacity_tokens
            scanned += 1
            if (
                row in selected
                or self._row_leases[row]
                or self._row_readers[row]
                or self._row_writing[row]
            ):
                continue
            old_slot = self._row_slot[row]
            candidates.append((row, old_slot if old_slot >= 0 else None))
            selected.add(row)
        if len(candidates) < count:
            if self._stats is not None:
                self._stats.bump("sidecar_capacity_miss_ct", count)
                if any(self._row_readers):
                    self._stats.bump("sidecar_reader_conflict_miss_ct", count)
                if any(self._row_leases):
                    self._stats.bump("sidecar_lease_conflict_miss_ct", count)
            raise SidecarCapacityError(
                f"need {count} rows, only {len(candidates)} reusable"
            )

        for row, old_slot in candidates:
            if old_slot is None:
                self._free_rows.discard(row)
                continue
            self._remove_row_locked(row, add_to_free=False)
        return candidates

    def _row_for_version_locked(self, slot: int, generation: int) -> int:
        """Resolve a current or retained COW generation under ``_lock``."""
        current = self._slot_to_row[slot]
        if current >= 0 and self._row_gen[current] == generation:
            return current
        return self._version_to_row.get((slot, generation), -1)

    def _remove_row_locked(self, row: int, *, add_to_free: bool = True) -> None:
        slot = self._row_slot[row]
        generation = self._row_gen[row]
        if slot >= 0:
            if self._version_to_row.get((slot, generation)) == row:
                del self._version_to_row[(slot, generation)]
            if self._slot_to_row[slot] == row:
                self._slot_to_row[slot] = -1
            self._resident_rows -= 1
        if self._row_leases[row]:
            self._leased_rows -= 1
            if self._row_readers[row] == 0 and not self._row_writing[row]:
                self._unavailable_rows -= 1
        self._row_slot[row] = -1
        self._row_gen[row] = 0
        self._row_leases[row] = 0
        if add_to_free and self._row_readers[row] == 0 and not self._row_writing[row]:
            self._free_rows.add(row)

    def _abort_write(self, plans) -> None:
        with self._lock:
            for slot, physical_row, old_row, _evicted, old_gen, _new_gen in plans:
                self._row_writing[physical_row] = False
                if physical_row == old_row:
                    self._remove_row_locked(physical_row)
                    self.slot_gen[slot] = old_gen + (2 if old_gen % 2 == 0 else 1)
                else:
                    # Reserved rows are unpublished. An evicted resident was
                    # already removed by ``_reserve_rows_locked``.
                    self._row_slot[physical_row] = -1
                    self._row_gen[physical_row] = 0
                    if self._row_readers[physical_row] == 0:
                        self._free_rows.add(physical_row)
            self._writing_rows -= len(plans)
            self._unavailable_rows -= len(plans)

    def read_rows_validated(
        self,
        *,
        slots: torch.Tensor,
        expected_tokens: torch.Tensor,
        own_slot_gens: Dict[int, int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        result = self._read_rows_validated(
            slots=slots,
            expected_tokens=expected_tokens,
            own_slot_gens=own_slot_gens,
            aux_dst=None,
            last_dst=None,
            direct=False,
        )
        return result

    def read_rows_validated_into(
        self,
        *,
        slots: torch.Tensor,
        expected_tokens: torch.Tensor,
        own_slot_gens: Dict[int, int],
        aux_dst: torch.Tensor,
        last_dst: torch.Tensor,
    ) -> bool:
        self._validate_destinations(slots, aux_dst, last_dst)
        return (
            self._read_rows_validated(
                slots=slots,
                expected_tokens=expected_tokens,
                own_slot_gens=own_slot_gens,
                aux_dst=aux_dst,
                last_dst=last_dst,
                direct=True,
            )
            is not None
        )

    def _read_rows_validated(
        self,
        *,
        slots: torch.Tensor,
        expected_tokens: torch.Tensor,
        own_slot_gens: Dict[int, int],
        aux_dst: Optional[torch.Tensor],
        last_dst: Optional[torch.Tensor],
        direct: bool,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if slots.device.type != "cpu" or expected_tokens.device.type != "cpu":
            raise ValueError("sidecar read identity tensors must be on CPU")
        rows = int(slots.numel())
        if expected_tokens.numel() != rows:
            raise ValueError("sidecar expected_tokens must match slots")
        if aux_dst is None:
            aux_dst = torch.empty(
                (rows, self.aux_buf.shape[1]), dtype=self.aux_buf.dtype
            )
        if last_dst is None:
            last_dst = torch.empty(
                (rows, self.last_buf.shape[1]), dtype=self.last_buf.dtype
            )
        slot_values = [int(slot) for slot in slots.tolist()]
        if any(slot < 0 or slot >= self.num_slots for slot in slot_values):
            return None
        expected_token_values = expected_tokens.to(torch.int32).tolist()

        wait_started_ns = time.monotonic_ns()
        self._lock.acquire()
        lock_started_ns = time.monotonic_ns()
        current_gens = self.slot_gen[slots].tolist()
        stored_tokens = self.token_id_map[slots].tolist()
        physical_values = []
        valid = True
        for index, slot in enumerate(slot_values):
            own_gen = own_slot_gens.get(slot)
            if own_gen is not None:
                generation = int(own_gen)
                physical = self._row_for_version_locked(slot, generation)
            else:
                generation = int(current_gens[index])
                physical = self._slot_to_row[slot]
                if (
                    generation <= 0
                    or generation % 2
                    or int(stored_tokens[index]) != int(expected_token_values[index])
                ):
                    valid = False
            physical_values.append(physical)
            if (
                physical < 0
                or self._row_slot[physical] != slot
                or self._row_gen[physical] != generation
                or self._row_writing[physical]
            ):
                valid = False
        if valid:
            for physical in physical_values:
                if self._row_readers[physical] == 0:
                    self._reader_rows += 1
                    if (
                        self._row_leases[physical] == 0
                        and not self._row_writing[physical]
                    ):
                        self._unavailable_rows += 1
                self._row_readers[physical] += 1
            self._reader_pins += len(physical_values)
        snapshot_hold_ns = time.monotonic_ns() - lock_started_ns
        self._lock.release()
        if not valid:
            if self._stats is not None:
                self._stats.bump(
                    "sidecar_gather_lock_wait_ns_ct", lock_started_ns - wait_started_ns
                )
                self._stats.bump("sidecar_gather_lock_hold_ns_ct", snapshot_hold_ns)
            return None

        physical_rows = torch.tensor(physical_values, dtype=torch.long)
        copy_started_ns = time.monotonic_ns()
        gathered = False
        try:
            torch.index_select(self.aux_buf, 0, physical_rows, out=aux_dst)
            torch.index_select(self.last_buf, 0, physical_rows, out=last_dst)
            gathered = True
            return aux_dst, last_dst
        finally:
            copy_ns = time.monotonic_ns() - copy_started_ns
            unpin_wait_ns = time.monotonic_ns()
            with self._lock:
                unpin_started_ns = time.monotonic_ns()
                for physical in physical_values:
                    self._row_readers[physical] -= 1
                    if self._row_readers[physical] == 0:
                        self._reader_rows -= 1
                        if (
                            self._row_leases[physical] == 0
                            and not self._row_writing[physical]
                        ):
                            self._unavailable_rows -= 1
                    if (
                        self._row_readers[physical] == 0
                        and self._row_slot[physical] < 0
                        and not self._row_writing[physical]
                    ):
                        self._free_rows.add(physical)
                self._reader_pins -= len(physical_values)
                unpin_hold_ns = time.monotonic_ns() - unpin_started_ns
            if self._stats is not None:
                self._stats.bump(
                    "sidecar_gather_lock_wait_ns_ct",
                    (lock_started_ns - wait_started_ns)
                    + (unpin_started_ns - unpin_wait_ns),
                )
                self._stats.bump(
                    "sidecar_gather_lock_hold_ns_ct",
                    snapshot_hold_ns + unpin_hold_ns,
                )
                self._stats.bump("sidecar_gather_copy_ns_ct", copy_ns)
                if gathered:
                    payload_bytes = rows * self._payload_row_bytes
                    self._stats.bump("sidecar_gather_rows_ct", rows)
                    self._stats.bump("sidecar_gather_bytes_ct", payload_bytes)
                    if direct:
                        self._stats.bump("sidecar_direct_gather_rows_ct", rows)
                        self._stats.bump(
                            "sidecar_direct_gather_bytes_ct", payload_bytes
                        )

    def lease_rows(self, own_slot_gens: Dict[int, int]) -> int:
        """Protect exact generations until their request export settles."""
        leased = 0
        with self._lock:
            for slot, generation in own_slot_gens.items():
                row = self._row_for_version_locked(int(slot), int(generation))
                if row < 0 or self._row_leases[row]:
                    continue
                if self._row_readers[row] == 0 and not self._row_writing[row]:
                    self._unavailable_rows += 1
                self._row_leases[row] = 1
                self._leased_rows += 1
                leased += 1
            leased_rows = self._leased_rows
        if leased and self._stats is not None:
            self._stats.observe_max("sidecar_leased_rows_high_water_ct", leased_rows)
        return leased

    def release_rows(
        self, own_slot_gens: Dict[int, int], *, discard: bool = True
    ) -> int:
        """Settle exact generations after export.

        Immutable-prefix Mooncake can discard them immediately. The legacy
        file sink merely removes their lease, retaining newest versions as an
        evictable warm-prefix cache.
        """
        released = 0
        unleased = 0
        with self._lock:
            for slot, generation in own_slot_gens.items():
                row = self._row_for_version_locked(int(slot), int(generation))
                if row < 0:
                    continue
                if self._row_leases[row]:
                    self._row_leases[row] = 0
                    self._leased_rows -= 1
                    if self._row_readers[row] == 0 and not self._row_writing[row]:
                        self._unavailable_rows -= 1
                    unleased += 1
                if discard:
                    self._remove_row_locked(row)
                released += 1
        if self._stats is not None:
            if discard and released:
                self._stats.bump("sidecar_released_rows_ct", released)
            if unleased:
                self._stats.bump("sidecar_unleased_rows_ct", unleased)
        return released

    def state_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            readers = self._reader_pins
            writing = self._writing_rows
            unavailable = self._unavailable_rows
            pressure_ratio = unavailable / self.capacity_tokens
            return {
                "logical_slots": self.num_slots,
                "capacity_tokens": self.capacity_tokens,
                "resident_rows": self._resident_rows,
                "leased_rows": self._leased_rows,
                "free_rows": len(self._free_rows),
                "reader_pins": readers,
                "reader_rows": self._reader_rows,
                "writing_rows": writing,
                "unavailable_rows": unavailable,
                "pressure_ratio": pressure_ratio,
            }

    @property
    def pressure_ratio(self) -> float:
        # Exact-union, lock-free read for serving admission. The integer is
        # updated under the metadata lock/GIL; a slightly stale observation
        # only delays one hysteresis transition and cannot affect correctness.
        return self._unavailable_rows / self.capacity_tokens

    def _validate_destinations(
        self,
        slots: torch.Tensor,
        aux_dst: torch.Tensor,
        last_dst: torch.Tensor,
    ) -> None:
        rows = int(slots.numel())
        expected = (
            (aux_dst, (rows, self.aux_buf.shape[1]), self.aux_buf.dtype, "aux"),
            (last_dst, (rows, self.last_buf.shape[1]), self.last_buf.dtype, "last"),
        )
        for tensor, shape, dtype, name in expected:
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"{name}_dst must be a contiguous CPU tensor")
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(
                    f"{name}_dst must have shape={shape}, dtype={dtype}; "
                    f"got shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                )


class HiddenCaptureBookkeeper:
    """rid-keyed finalize records, miss propagation, and orphan sweep."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # rid -> {kv slot: settled gen at finalize time}
        self._finalize_records: Dict[str, Dict[int, int]] = {}
        # rid -> highest staging seq containing rows owned by that request.
        self._max_enqueued_seq: Dict[str, int] = {}
        self._missed_rids: set = set()
        self._touch_s: Dict[str, float] = {}
        # Retraction may race an already-enqueued D2H slot. Keep its sequence
        # cutoff so late finalization cannot recreate stale exact leases after
        # ``invalidate`` has cleared the request.
        self._invalidated_through_seq: Dict[str, int] = {}

    def record_rows(
        self,
        rid: str,
        slots: Sequence[int],
        gens: Sequence[int],
        *,
        ring_seq: Optional[int] = None,
    ) -> bool:
        with self._lock:
            if ring_seq is not None and ring_seq <= self._invalidated_through_seq.get(
                rid, -1
            ):
                return False
            self._finalize_records.setdefault(rid, {}).update(zip(slots, gens))
            self._touch_s[rid] = time.monotonic()
            return True

    def record_enqueued(self, rids: Sequence[str], ring_seq: int) -> None:
        with self._lock:
            now = time.monotonic()
            for rid in rids:
                self._max_enqueued_seq[rid] = max(
                    ring_seq, self._max_enqueued_seq.get(rid, -1)
                )
                self._touch_s[rid] = now

    def export_barrier(self, rid: str, global_seq: int) -> Tuple[int, bool]:
        """Return (seq, has_own_rows) for a finish snapshot.

        A fully-warm request has no enqueue record and must conservatively
        retain the global barrier until prefix V1 can rebuild it entirely
        from existence-checked immutable Mooncake segments.
        """
        with self._lock:
            own_seq = self._max_enqueued_seq.get(rid)
            return (own_seq, True) if own_seq is not None else (global_seq, False)

    def mark_miss(self, rids: Sequence[str]) -> int:
        """Mark requests fail-closed and return the number newly missed."""
        with self._lock:
            now = time.monotonic()
            new_misses = 0
            for rid in rids:
                if rid not in self._missed_rids:
                    new_misses += 1
                self._missed_rids.add(rid)
                self._touch_s[rid] = now
            return new_misses

    def is_missed(self, rid: str) -> bool:
        with self._lock:
            return rid in self._missed_rids

    def invalidate(self, rid: str) -> Dict[int, int]:
        """Retract/abort: drop stale finalize records so a re-scheduled request
        starts clean (its slots may be re-assigned on the next prefill)."""
        with self._lock:
            rows = self._finalize_records.pop(rid, {})
            cutoff = self._max_enqueued_seq.get(rid, -1)
            self._invalidated_through_seq[rid] = max(
                cutoff, self._invalidated_through_seq.get(rid, -1)
            )
            self._max_enqueued_seq.pop(rid, None)
            self._touch_s[rid] = time.monotonic()
            self._missed_rids.discard(rid)
            return rows

    def pop(self, rid: str) -> Dict[int, int]:
        """Take (and clear) all bookkeeping for a finished rid."""
        with self._lock:
            self._missed_rids.discard(rid)
            self._max_enqueued_seq.pop(rid, None)
            self._touch_s.pop(rid, None)
            self._invalidated_through_seq.pop(rid, None)
            return self._finalize_records.pop(rid, {})

    def sweep_orphans(self, ttl_s: float = _BOOKKEEPING_TTL_S) -> int:
        return len(self.take_orphans(ttl_s=ttl_s))

    def take_orphans(self, ttl_s: float = _BOOKKEEPING_TTL_S) -> List[Dict[int, int]]:
        """Pop stale exact generations so their sidecar leases can settle."""
        with self._lock:
            now = time.monotonic()
            stale = [rid for rid, t in self._touch_s.items() if now - t > ttl_s]
            rows = []
            for rid in stale:
                rows.append(self._finalize_records.pop(rid, {}))
                self._max_enqueued_seq.pop(rid, None)
                self._missed_rids.discard(rid)
                self._touch_s.pop(rid, None)
                self._invalidated_through_seq.pop(rid, None)
            return rows

    def state_snapshot(self) -> Dict[str, int]:
        """Small, lock-consistent backlog summary for characterization."""
        with self._lock:
            return {
                "finalize_record_rids": len(self._finalize_records),
                "enqueued_rids": len(self._max_enqueued_seq),
                "missed_rids": len(self._missed_rids),
                "invalidated_rids": len(self._invalidated_through_seq),
                "touched_rids": len(self._touch_s),
            }


class HiddenFinalizeWorker:
    """Daemon thread: drains the staging rings into the sidecar in enqueue
    (global sequence) order.

    Sole writer of the sidecar arrays. Advances ``last_finalized_seq`` so the
    export thread's ring barrier (``last_finalized_seq >= snapshot``) implies
    every slot enqueued at or before the snapshot has settled. Prefill and
    verify rings share one seq counter; the finalizer gates on the globally
    oldest in-flight sequence so settlement never exposes a hole.
    """

    def __init__(
        self,
        *,
        ring: HiddenStagingRing,
        sidecar: HiddenHostSidecar,
        bookkeeper: HiddenCaptureBookkeeper,
        stats: HiddenCaptureStats,
        verify_ring: Optional[HiddenStagingRing] = None,
        twin_pool: Optional[DeviceTwinPool] = None,
        span_recorder: Optional[HiddenWallSpanRecorder] = None,
        poll_interval_s: float = 0.001,
        stats_log_interval_s: float = _STATS_LOG_INTERVAL_S,
    ) -> None:
        self.ring = ring
        self.verify_ring = verify_ring
        self.sidecar = sidecar
        self.bookkeeper = bookkeeper
        self.stats = stats
        self.twin_pool = twin_pool
        self.span_recorder = span_recorder or HiddenWallSpanRecorder()
        self.last_finalized_seq = -1
        self._poll_interval_s = poll_interval_s
        self._stats_log_interval_s = stats_log_interval_s
        self._last_stats_log_s = time.monotonic()
        self._last_stats_snapshot: Optional[Dict[str, int]] = None
        # The head-of-line probe issues up to one event query per ring; keep
        # it off the 1ms idle cadence so observability never rivals the
        # settle path's own polling cost.
        self._backlog_probe_interval_s = 0.05
        self._last_backlog_probe_s = 0.0
        self._stop_requested = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="hidden-capture-finalize", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Request shutdown; the worker drains both rings before exiting."""
        self._stop_requested.set()

    def join(self, timeout_s: Optional[float] = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    @property
    def pending_count(self) -> int:
        count = self.ring.inflight_count
        if self.verify_ring is not None:
            count += self.verify_ring.inflight_count
        return count

    def _pop_next_ready(self) -> Optional[Tuple[_StagingSlot, HiddenStagingRing]]:
        """Ready head of whichever ring holds the globally oldest slot.

        Strict seq order across rings is a correctness barrier: exporting a
        younger slot first would let a request barrier observe a hole in
        global settlement. Within one ring, the single capture-stream
        submitter makes readiness follow enqueue order; across rings the seq
        decides which head must settle next.
        """
        if self.verify_ring is None:
            slot = self.ring.pop_ready()
            return (slot, self.ring) if slot is not None else None
        prefill_seq = self.ring.peek_inflight_seq()
        verify_seq = self.verify_ring.peek_inflight_seq()
        if prefill_seq is None and verify_seq is None:
            return None
        if verify_seq is None or (prefill_seq is not None and prefill_seq < verify_seq):
            source = self.ring
        else:
            source = self.verify_ring
        slot = source.pop_ready()
        return (slot, source) if slot is not None else None

    def _run(self) -> None:
        _set_os_thread_name("hcap-finalize")
        while True:
            if self._stop_requested.is_set() and self.pending_count == 0:
                break
            self._maybe_log_stats()
            # Recycle twins as soon as their D2H completes, independent of
            # this thread's (much slower) sidecar-memcpy progress; the
            # forward thread also reaps opportunistically on pool miss.
            if self.twin_pool is not None and self.verify_ring is not None:
                self.verify_ring.reap_ready_twins(self.twin_pool)
            popped = self._pop_next_ready()
            if popped is None:
                self._observe_backlog()
                time.sleep(self._poll_interval_s)
                continue
            slot, source_ring = popped
            try:
                self._active = 1
                self.finalize_slot(slot)
            except SidecarCapacityError as error:
                logger.warning(
                    "hidden capture sidecar pressure miss; dropping slot: %s",
                    error,
                )
                missed = (
                    slot.rids
                    if slot.kind.startswith("verify")
                    else [rid for rid, _, _ in slot.req_ranges]
                )
                self.bookkeeper.mark_miss(missed)
            except Exception:
                # Capture must never take serving down; drop the slot's rows.
                logger.exception("hidden capture finalize failed; dropping slot")
                missed = (
                    slot.rids
                    if slot.kind.startswith("verify")
                    else [rid for rid, _, _ in slot.req_ranges]
                )
                self.bookkeeper.mark_miss(missed)
            finally:
                self._active = 0
                seq = slot.ring_seq
                if slot.twin is not None and self.twin_pool is not None:
                    # Safe: pop_ready() already confirmed the D2H out of the
                    # twin completed.
                    self.twin_pool.release(slot.twin)
                source_ring.release(slot)
                self.last_finalized_seq = seq

    def _observe_backlog(self) -> None:
        """Record settlement-backlog high-water marks while blocked.

        ``finalize_lag_high_water_ct`` is the max enqueue-to-finalize seq
        distance; ``finalize_hol_blocked_slots_high_water_ct`` is the max
        number of copy-complete slots pinned behind a pending global head by
        strict in-order settlement (#9's core structural judgment input).
        Only sampled on idle iterations and rate-limited, so the counters'
        own event queries never rival the settle path's polling cost.
        """
        now = time.monotonic()
        if now - self._last_backlog_probe_s < self._backlog_probe_interval_s:
            return
        self._last_backlog_probe_s = now
        lag = self.ring.last_enqueued_seq - self.last_finalized_seq
        if lag > 0:
            self.stats.observe_max("finalize_lag_high_water_ct", lag)
        if self.verify_ring is None:
            return
        # _pop_next_ready returned None with work in flight => the globally
        # oldest slot's copy is pending. Every completed slot in either ring
        # is settlement-blocked behind it.
        blocked = self.ring.count_ready_prefix() + self.verify_ring.count_ready_prefix()
        if blocked > 0:
            self.stats.observe_max(
                "finalize_hol_blocked_slots_high_water_ct", blocked
            )

    def _maybe_log_stats(self) -> None:
        """Time-based periodic stats emission (finalize thread, off the hot
        slot path); skipped while counters are unchanged. The old slot-count
        trigger (slots_finalized_ct % N) went silent exactly when it
        mattered: a short saturation burst finalizes fewer than N slots, so
        runs whose misses needed attribution logged nothing."""
        now = time.monotonic()
        if now - self._last_stats_log_s < self._stats_log_interval_s:
            return
        self._last_stats_log_s = now
        snapshot = self.stats.snapshot()
        if snapshot != self._last_stats_snapshot:
            self._last_stats_snapshot = snapshot
            logger.info("hidden capture stats: %s", snapshot)

    def finalize_slot(self, slot: _StagingSlot) -> None:
        started_ns = time.monotonic_ns()
        kind = slot.kind
        stats_kind = "verify" if kind.startswith("verify") else "prefill"
        try:
            if kind.startswith("verify"):
                self._finalize_verify_slot(slot)
            else:
                finalized_rows = self._finalize_prefill_slot(slot)
                self.stats.bump("prefill_rows_finalized_ct", finalized_rows)
            self.stats.bump("slots_finalized_ct")
        finally:
            ended_ns = time.monotonic_ns()
            self.stats.bump(
                f"finalize_{stats_kind}_busy_ns_ct",
                ended_ns - started_ns,
            )
            self.span_recorder.record_finalize(started_ns, ended_ns)

    def _finalize_prefill_slot(self, slot: _StagingSlot) -> int:
        num_rows = slot.num_rows
        keep = []
        compact_ranges = []
        for rid, start, end in slot.req_ranges:
            if end <= start:
                continue
            compact_start = len(keep)
            keep.extend(range(start, end))
            compact_ranges.append((rid, compact_start, len(keep)))
        if not keep:
            return 0
        if len(keep) == num_rows and keep[0] == 0 and keep[-1] == num_rows - 1:
            slots = slot.cache_loc[:num_rows]
            aux_rows = slot.aux[:num_rows]
            last_rows = slot.last[:num_rows]
            tokens = slot.tokens[:num_rows]
        else:
            keep_idx = torch.tensor(keep, dtype=torch.long)
            self.stats.bump("prefill_finalize_slow_gather_ct")
            slots = slot.cache_loc[keep_idx]
            aux_rows = slot.aux[keep_idx]
            last_rows = slot.last[keep_idx]
            tokens = slot.tokens[keep_idx]
        gens = self.sidecar.write_rows(
            slots=slots,
            aux_rows=aux_rows,
            last_rows=last_rows,
            tokens=tokens,
        )
        slots_list = slots.tolist()
        gens_list = gens.tolist()
        leased_rows = {}
        for rid, start, end in compact_ranges:
            rid_slots = slots_list[start:end]
            rid_gens = gens_list[start:end]
            if self.bookkeeper.record_rows(
                rid, rid_slots, rid_gens, ring_seq=slot.ring_seq
            ):
                leased_rows.update(zip(rid_slots, rid_gens))
        leased = self.sidecar.lease_rows(leased_rows)
        if leased != len(leased_rows):
            raise RuntimeError(
                f"sidecar leased {leased}/{len(leased_rows)} finalized prefill rows"
            )
        return len(keep)

    def _finalize_verify_slot(self, slot: _StagingSlot) -> None:
        """Commit compact verify rows: the pack kernel placed exactly the
        committed prefix of every request (request i's rows are contiguous at
        [sum(commit_lens[:i]), sum(commit_lens[:i+1]))), so the committed
        prefix of the payload enters the sidecar. The row count is derived
        from the pinned commit_lens — it arrived in the same copy as the
        payload, so it is trustworthy exactly when the payload is."""
        commit_lens = slot.commit_lens[: slot.num_reqs].tolist()
        if any(length < 0 or length > slot.stride for length in commit_lens):
            raise RuntimeError(
                f"invalid compact verify commit_lens: {commit_lens}, "
                f"stride={slot.stride}"
            )
        # Per-request bound length <= stride caps the sum at the payload
        # views' n_rows = num_reqs * stride, so no separate capacity check.
        num_rows = sum(commit_lens)
        slot.num_rows = num_rows
        if num_rows:
            slots = slot.cache_loc[:num_rows]
            gens = self.sidecar.write_rows(
                slots=slots,
                aux_rows=slot.aux[:num_rows],
                last_rows=slot.last[:num_rows],
                tokens=slot.tokens[:num_rows],
            )
            slots_list = slots.tolist()
            gens_list = gens.tolist()
            row = 0
            leased_rows = {}
            for rid, commit_len in zip(slot.rids, commit_lens):
                rid_slots = slots_list[row : row + commit_len]
                rid_gens = gens_list[row : row + commit_len]
                if self.bookkeeper.record_rows(
                    rid,
                    rid_slots,
                    rid_gens,
                    ring_seq=slot.ring_seq,
                ):
                    leased_rows.update(zip(rid_slots, rid_gens))
                row += commit_len
            leased = self.sidecar.lease_rows(leased_rows)
            if leased != len(leased_rows):
                raise RuntimeError(
                    f"sidecar leased {leased}/{len(leased_rows)} compact verify rows"
                )
        self.stats.bump("verify_rows_committed_ct", num_rows)
