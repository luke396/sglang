"""Host-side staging ring, sidecar, and finalize worker for hidden-state capture.

Data path (M1, prefill only)::

    forward (GPU aux/last refs)
        -> HiddenStagingRing.enqueue      (async D2H into pinned slots, copy stream)
        -> HiddenFinalizeWorker           (event-gated seqlock writes into the sidecar)
        -> HiddenHostSidecar              (pageable per-token-slot payload arrays)
        -> export thread (hidden_sink.py) (validated ordered gather at request finish)

Concurrency contract
--------------------
Three threads touch this module: the scheduler thread (staging enqueue, finish
snapshots), one finalize thread (the *sole writer* of the sidecar payload /
generation / token arrays), and one export thread (reader). Sidecar access is
serialized by a mutex held across ``write_rows`` / ``read_rows_validated``
(both run on background threads; the hold time is the row memcpy, so serving
is unaffected). A mutex — not a seqlock — is load-bearing here: torch CPU
kernels release the GIL, and on weakly-ordered hosts (ARM: GH200/GB200) a
seqlock's unfenced payload stores could become visible after the "settled"
generation store, letting a reader return wrong-but-validated data. The
generation counters remain for *identity* (ABA) validation, not for torn-read
protection.

Identity model: sidecar rows are keyed by KV token-slot index (same numbering
as the KV cache). Requests hold their slots until ``release_kv_cache`` (called
*after* the finish hook), and warm-prefix slots are protected by radix
``lock_ref`` while the request is alive, so the race window is only
(a) rows still in flight in the staging ring and (b) slot reuse after finish.
Own rows are validated against the exact generation recorded at finalize time;
warm-prefix rows (written by another request) are validated by token-id match
plus an even, unchanged generation across the payload read. The prefix check
is probabilistic: a slot rewritten with the *same* token id between the finish
snapshot and the export read would pass validation. Accepted M1 residual risk
(see the capture plan); every other inconsistency fails closed to a
whole-sample capture miss — capture never blocks or corrupts serving.
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

# Finalize-side log cadence, in finalized slots.
_STATS_LOG_EVERY_SLOTS = 256
# Orphaned bookkeeping entries (aborted requests) older than this are swept.
_BOOKKEEPING_TTL_S = 600.0


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
        self.slots_finalized_ct = 0
        self.stage_full_miss_ct = 0
        self.attach_mismatch_miss_ct = 0
        self.gen_mismatch_miss_ct = 0
        self.prefix_invalid_miss_ct = 0
        self.export_queue_full_miss_ct = 0
        self.export_timeout_miss_ct = 0
        self.duplicate_sample_miss_ct = 0
        self.sample_too_large_miss_ct = 0
        self.sink_put_failed_miss_ct = 0
        # Sample exported but its manifest entry failed: discoverable by
        # nothing, reclaimed by lease TTL (mooncake sink only).
        self.manifest_orphan_miss_ct = 0
        self.export_ok_ct = 0
        self.skipped_forward_ct = 0
        self.verify_rows_committed_ct = 0
        self.verify_twin_full_miss_ct = 0
        self.verify_oversize_miss_ct = 0

    def bump(self, name: str, delta: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + delta)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {k: v for k, v in vars(self).items() if k.endswith("_ct")}

    def log(self, prefix: str) -> None:
        logger.info("%s %s", prefix, self.snapshot())


class _StagingSlot(msgspec.Struct):
    index: int
    aux: torch.Tensor  # pinned [slot_tokens, aux_width]
    last: torch.Tensor  # pinned [slot_tokens, last_width]
    cache_loc: torch.Tensor  # pinned [slot_tokens] int64
    tokens: torch.Tensor  # pinned [slot_tokens] int64
    event: Any
    num_rows: int = 0
    ring_seq: int = -1
    # Prefill slots: [(rid, start_row, end_row)] within [0, num_rows).
    req_ranges: List[Tuple[str, int, int]] = []
    # Verify slots: request i owns rows [i*stride, (i+1)*stride); only the
    # first commit_lens[i] of them are committed (selected at finalize).
    kind: str = "prefill"
    commit_lens: Optional[torch.Tensor] = None  # pinned [num_reqs] int32
    rids: List[str] = []
    stride: int = 0
    num_reqs: int = 0
    # Device twin borrowed for this verify enqueue; released at finalize.
    twin: Optional[_DeviceTwin] = None


class _DeviceTwin(msgspec.Struct):
    """Capture-owned HBM buffers for one verify step's strided window.

    The worker packs the (graph/persistent) verify outputs into a twin on the
    forward stream — same-stream ordering IS the overwrite fence required by
    the capture plan's invariant #6: step t+1's replay queues behind the pack,
    so the persistent source can't be rewritten under the copy, and the
    forward stream never waits on D2H (which reads the twin, not the source).
    """

    index: int
    aux: torch.Tensor  # [twin_tokens, aux_width] device
    last: torch.Tensor  # [twin_tokens, last_width] device
    cache_loc: torch.Tensor  # [twin_tokens] int64 device
    tokens: torch.Tensor  # [twin_tokens] int64 device
    commit_lens: torch.Tensor  # [max_reqs] int32 device
    fence_event: Any = None  # recorded on the forward stream after the pack


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
    ) -> None:
        self.twin_tokens = twin_tokens
        self.max_reqs = max_reqs
        self._lock = threading.Lock()
        self._free: List[_DeviceTwin] = []
        for i in range(num_twins):
            self._free.append(
                _DeviceTwin(
                    index=i,
                    aux=torch.empty(
                        (twin_tokens, aux_width), dtype=dtype, device=device
                    ),
                    last=torch.empty(
                        (twin_tokens, last_width), dtype=dtype, device=device
                    ),
                    cache_loc=torch.empty(
                        (twin_tokens,), dtype=torch.int64, device=device
                    ),
                    tokens=torch.empty(
                        (twin_tokens,), dtype=torch.int64, device=device
                    ),
                    commit_lens=torch.empty(
                        (max_reqs,), dtype=torch.int32, device=device
                    ),
                    fence_event=(
                        torch.cuda.Event() if use_cuda_events else _NullEvent()
                    ),
                )
            )
        size_mb = (
            num_twins * twin_tokens * (aux_width + last_width) * dtype.itemsize
        ) / (1024 * 1024)
        logger.info(
            "Hidden capture DeviceTwinPool allocated: %d twins x %d tokens, "
            "%.0f MB HBM",
            num_twins,
            twin_tokens,
            size_mb,
        )

    def try_acquire(self) -> Optional[_DeviceTwin]:
        with self._lock:
            return self._free.pop() if self._free else None

    def release(self, twin: _DeviceTwin) -> None:
        with self._lock:
            self._free.append(twin)


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
    ) -> None:
        self.slot_tokens = slot_tokens
        self._lock = threading.Lock()
        self._free: deque[_StagingSlot] = deque()
        self._inflight: deque[_StagingSlot] = deque()
        self._next_seq = 0
        self.last_enqueued_seq = -1

        def _pinned(shape: Tuple[int, ...], dt: torch.dtype) -> torch.Tensor:
            return torch.empty(shape, dtype=dt, pin_memory=pin_memory)

        for i in range(num_slots):
            self._free.append(
                _StagingSlot(
                    index=i,
                    aux=_pinned((slot_tokens, aux_width), dtype),
                    last=_pinned((slot_tokens, last_width), dtype),
                    cache_loc=_pinned((slot_tokens,), torch.int64),
                    tokens=_pinned((slot_tokens,), torch.int64),
                    event=(torch.cuda.Event() if use_cuda_events else _NullEvent()),
                )
            )
        size_gb = (
            num_slots * slot_tokens * (aux_width + last_width) * dtype.itemsize
        ) / _GB
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

    def enqueue_verify_segment(
        self,
        slot: _StagingSlot,
        *,
        twin: _DeviceTwin,
        rids: List[str],
        stride: int,
        num_reqs: int,
    ) -> None:
        """D2H one verify step's strided window from its device twin into
        pinned memory. Caller's current stream must wait on ``twin.fence_event``
        first (the twin was packed on the forward stream). Committed-row
        selection happens at finalize on CPU via the commit lens.
        """
        num_rows = num_reqs * stride
        slot.aux[:num_rows].copy_(twin.aux[:num_rows], non_blocking=True)
        slot.last[:num_rows].copy_(twin.last[:num_rows], non_blocking=True)
        slot.cache_loc[:num_rows].copy_(twin.cache_loc[:num_rows], non_blocking=True)
        slot.tokens[:num_rows].copy_(twin.tokens[:num_rows], non_blocking=True)
        if slot.commit_lens is None or slot.commit_lens.shape[0] < num_reqs:
            slot.commit_lens = torch.empty(
                (max(num_reqs, 256),),
                dtype=torch.int32,
                pin_memory=slot.aux.is_pinned(),
            )
        slot.commit_lens[:num_reqs].copy_(
            twin.commit_lens[:num_reqs], non_blocking=True
        )
        slot.event.record()

        slot.kind = "verify"
        slot.num_rows = num_rows
        slot.req_ranges = []
        slot.rids = rids
        slot.stride = stride
        slot.num_reqs = num_reqs
        slot.twin = twin
        with self._lock:
            slot.ring_seq = self._next_seq
            self._next_seq += 1
            self.last_enqueued_seq = slot.ring_seq
            self._inflight.append(slot)

    def enqueue_segment(
        self,
        slot: _StagingSlot,
        *,
        aux_rows: torch.Tensor,
        last_rows: torch.Tensor,
        cache_locs: torch.Tensor,
        tokens: torch.Tensor,
        req_ranges: List[Tuple[str, int, int]],
    ) -> None:
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
        with self._lock:
            slot.ring_seq = self._next_seq
            self._next_seq += 1
            self.last_enqueued_seq = slot.ring_seq
            self._inflight.append(slot)

    def pop_ready(self) -> Optional[_StagingSlot]:
        """Oldest in-flight slot whose D2H completed, preserving FIFO order."""
        with self._lock:
            if not self._inflight or not self._inflight[0].event.query():
                return None
            return self._inflight.popleft()

    def reap_ready_twins(self, twin_pool: DeviceTwinPool) -> int:
        """Release device twins whose D2H into pinned memory has completed.

        A twin's data is dead the moment its slot's copy event fires; waiting
        for the slot to reach the finalize queue head (and for finalize's
        sidecar memcpy — hundreds of MB for prefill slots) holds twins for
        tens of ms longer than needed and starves verify capture. The capture
        stream is FIFO so this scan usually stops at the first pending event.
        """
        reaped = 0
        with self._lock:
            for slot in self._inflight:
                if slot.twin is None:
                    continue
                if not slot.event.query():
                    break  # FIFO stream: later slots can't be done either
                twin_pool.release(slot.twin)
                slot.twin = None
                reaped += 1
        return reaped

    def release(self, slot: _StagingSlot) -> None:
        slot.num_rows = 0
        slot.ring_seq = -1
        slot.req_ranges = []
        slot.kind = "prefill"
        slot.rids = []
        slot.stride = 0
        slot.num_reqs = 0
        slot.twin = None
        with self._lock:
            self._free.append(slot)


class HiddenHostSidecar:
    """Pageable per-token-slot payload arrays plus the identity maps.

    Payload buffers are ``torch.empty`` (validity is gated by ``slot_gen`` /
    ``token_id_map``, so pre-zeroing terabytes is wasted work). Only the
    finalize thread writes; writes and validated reads are serialized by
    ``_lock`` (see the module docstring for why a mutex, not a seqlock).
    ``slot_gen`` still increments by 2 per settled write — the even/odd shape
    is kept so generations recorded before this locking change stay valid —
    but its role is identity (ABA) validation only.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        aux_width: int,
        last_width: int,
        dtype: torch.dtype,
    ) -> None:
        self._lock = threading.Lock()
        self.aux_buf = torch.empty((num_slots, aux_width), dtype=dtype)
        self.last_buf = torch.empty((num_slots, last_width), dtype=dtype)
        self.token_id_map = torch.full((num_slots,), -1, dtype=torch.int32)
        # Even = settled, odd = write in flight; 0 = never written.
        self.slot_gen = torch.zeros((num_slots,), dtype=torch.int64)
        size_gb = (
            self.aux_buf.numel() * self.aux_buf.element_size()
            + self.last_buf.numel() * self.last_buf.element_size()
        ) / _GB
        logger.info(
            "HiddenHostSidecar allocated: %d slots, aux_width=%d, last_width=%d, "
            "%.2f GB pageable host memory",
            num_slots,
            aux_width,
            last_width,
            size_gb,
        )

    def write_rows(
        self,
        *,
        slots: torch.Tensor,
        aux_rows: torch.Tensor,
        last_rows: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Write rows under the sidecar lock (finalize thread only).

        Returns the settled generations for identity bookkeeping.
        """
        with self._lock:
            gens = self.slot_gen[slots]
            new_gens = gens + 2
            self.aux_buf[slots] = aux_rows
            self.last_buf[slots] = last_rows
            self.token_id_map[slots] = tokens.to(torch.int32)
            self.slot_gen[slots] = new_gens
            return new_gens

    def read_rows_validated(
        self,
        *,
        slots: torch.Tensor,
        expected_tokens: torch.Tensor,
        own_slot_gens: Dict[int, int],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Read ``slots`` under the sidecar lock; None on any identity failure.

        Own rows (present in ``own_slot_gens``) must match the exact settled
        generation recorded at finalize time. Warm-prefix rows must carry the
        expected token id at a nonzero generation.
        """
        with self._lock:
            gens = self.slot_gen[slots].clone()
            aux_rows = self.aux_buf[slots].clone()
            last_rows = self.last_buf[slots].clone()
            stored_tokens = self.token_id_map[slots].clone()

        if not bool(((gens > 0) & (gens % 2 == 0)).all()):
            # gen 0 = never written; odd = corrupt (writes advance by 2).
            return None

        slots_list = slots.tolist()
        gens_list = gens.tolist()
        own_ok = True
        prefix_mask = torch.ones(len(slots_list), dtype=torch.bool)
        for i, (slot, gen) in enumerate(zip(slots_list, gens_list)):
            expected_gen = own_slot_gens.get(slot)
            if expected_gen is not None:
                prefix_mask[i] = False
                if gen != expected_gen:
                    own_ok = False
                    break
        if not own_ok:
            return None
        # Warm-prefix rows: token identity check.
        if bool(prefix_mask.any()):
            if not torch.equal(
                stored_tokens[prefix_mask],
                expected_tokens[prefix_mask].to(torch.int32),
            ):
                return None
        return aux_rows, last_rows


class HiddenCaptureBookkeeper:
    """rid-keyed finalize records, miss propagation, and orphan sweep."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # rid -> {kv slot: settled gen at finalize time}
        self._finalize_records: Dict[str, Dict[int, int]] = {}
        self._missed_rids: set = set()
        self._touch_s: Dict[str, float] = {}

    def record_rows(self, rid: str, slots: Sequence[int], gens: Sequence[int]) -> None:
        with self._lock:
            self._finalize_records.setdefault(rid, {}).update(zip(slots, gens))
            self._touch_s[rid] = time.monotonic()

    def mark_miss(self, rids: Sequence[str]) -> None:
        with self._lock:
            now = time.monotonic()
            for rid in rids:
                self._missed_rids.add(rid)
                self._touch_s[rid] = now

    def is_missed(self, rid: str) -> bool:
        with self._lock:
            return rid in self._missed_rids

    def invalidate(self, rid: str) -> None:
        """Retract/abort: drop stale finalize records so a re-scheduled request
        starts clean (its slots may be re-assigned on the next prefill)."""
        with self._lock:
            self._finalize_records.pop(rid, None)
            self._touch_s.pop(rid, None)

    def pop(self, rid: str) -> Dict[int, int]:
        """Take (and clear) all bookkeeping for a finished rid."""
        with self._lock:
            self._missed_rids.discard(rid)
            self._touch_s.pop(rid, None)
            return self._finalize_records.pop(rid, {})

    def sweep_orphans(self, ttl_s: float = _BOOKKEEPING_TTL_S) -> int:
        with self._lock:
            now = time.monotonic()
            stale = [rid for rid, t in self._touch_s.items() if now - t > ttl_s]
            for rid in stale:
                self._finalize_records.pop(rid, None)
                self._missed_rids.discard(rid)
                self._touch_s.pop(rid, None)
            return len(stale)


class HiddenFinalizeWorker:
    """Daemon thread: drains the staging ring into the sidecar in FIFO order.

    Sole writer of the sidecar arrays. Advances ``last_finalized_seq`` so the
    export thread's ring barrier (``last_finalized_seq >= snapshot``) implies
    every slot enqueued at or before the snapshot has settled.
    """

    def __init__(
        self,
        *,
        ring: HiddenStagingRing,
        sidecar: HiddenHostSidecar,
        bookkeeper: HiddenCaptureBookkeeper,
        stats: HiddenCaptureStats,
        twin_pool: Optional[DeviceTwinPool] = None,
        poll_interval_s: float = 0.001,
    ) -> None:
        self.ring = ring
        self.sidecar = sidecar
        self.bookkeeper = bookkeeper
        self.stats = stats
        self.twin_pool = twin_pool
        self.last_finalized_seq = -1
        self._poll_interval_s = poll_interval_s
        self._running = True
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="hidden-capture-finalize", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            # Recycle twins as soon as their D2H completes, independent of
            # this thread's (much slower) sidecar-memcpy progress; the
            # forward thread also reaps opportunistically on pool miss.
            if self.twin_pool is not None:
                self.ring.reap_ready_twins(self.twin_pool)
            slot = self.ring.pop_ready()
            if slot is None:
                time.sleep(self._poll_interval_s)
                continue
            try:
                self.finalize_slot(slot)
            except Exception:
                # Capture must never take serving down; drop the slot's rows.
                logger.exception("hidden capture finalize failed; dropping slot")
                missed = (
                    slot.rids
                    if slot.kind == "verify"
                    else [rid for rid, _, _ in slot.req_ranges]
                )
                self.bookkeeper.mark_miss(missed)
            finally:
                seq = slot.ring_seq
                if slot.twin is not None and self.twin_pool is not None:
                    # Safe: pop_ready() already confirmed the D2H out of the
                    # twin completed.
                    self.twin_pool.release(slot.twin)
                self.ring.release(slot)
                self.last_finalized_seq = seq
            if (
                self.stats.slots_finalized_ct % _STATS_LOG_EVERY_SLOTS
                == _STATS_LOG_EVERY_SLOTS - 1
            ):
                self.stats.log("hidden capture stats:")

    def finalize_slot(self, slot: _StagingSlot) -> None:
        if slot.kind == "verify":
            self._finalize_verify_slot(slot)
        else:
            self._finalize_prefill_slot(slot)
        self.stats.bump("slots_finalized_ct")

    def _finalize_prefill_slot(self, slot: _StagingSlot) -> None:
        num_rows = slot.num_rows
        slots = slot.cache_loc[:num_rows]
        gens = self.sidecar.write_rows(
            slots=slots,
            aux_rows=slot.aux[:num_rows],
            last_rows=slot.last[:num_rows],
            tokens=slot.tokens[:num_rows],
        )
        slots_list = slots.tolist()
        gens_list = gens.tolist()
        for rid, start, end in slot.req_ranges:
            self.bookkeeper.record_rows(
                rid, slots_list[start:end], gens_list[start:end]
            )

    def _finalize_verify_slot(self, slot: _StagingSlot) -> None:
        """Committed-prefix selection on CPU: request i's rows live at
        [i*stride, i*stride + commit_lens[i]); the rest of its stride window
        is rejected drafts and must not enter the sidecar."""
        stride = slot.stride
        commit_lens = slot.commit_lens[: slot.num_reqs].tolist()
        keep = []
        for i, commit_len in enumerate(commit_lens):
            keep.extend(range(i * stride, i * stride + commit_len))
        if keep:
            keep_idx = torch.tensor(keep, dtype=torch.long)
            slots = slot.cache_loc[keep_idx]
            gens = self.sidecar.write_rows(
                slots=slots,
                aux_rows=slot.aux[keep_idx],
                last_rows=slot.last[keep_idx],
                tokens=slot.tokens[keep_idx],
            )
            slots_list = slots.tolist()
            gens_list = gens.tolist()
            row = 0
            for rid, commit_len in zip(slot.rids, commit_lens):
                self.bookkeeper.record_rows(
                    rid,
                    slots_list[row : row + commit_len],
                    gens_list[row : row + commit_len],
                )
                row += commit_len
        self.stats.bump("verify_rows_committed_ct", len(keep))
