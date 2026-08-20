"""Online capture of target aux/last hidden states for draft-model training.

``HiddenStatesCapturer`` is the orchestration seam between serving and the
host-side capture pipeline (see ``hidden_host.py`` / ``hidden_sink.py``):

- ``on_forward_end``   (model runner, forward thread): grab GPU references to
  the packed aux hidden states ``[T, K*H]`` and the post-final-norm last
  hidden states ``[T, H]`` plus row->request attribution. No copies.
- ``capture_verify_window`` (DSpark worker, forward stream, post-acceptance):
  pack one verify step's strided window into a capture-owned device twin.
  The pack runs on the forward stream, so the next graph replay queues
  behind it — same-stream ordering IS the overwrite fence.
- ``HiddenCaptureOutput.stage`` / ``HiddenVerifyCaptureOutput.stage``
  (scheduler, copy stream): async D2H into the pinned staging ring on a
  dedicated capture stream.
- ``collect_batch_at_finish`` (scheduler, before ``release_kv_cache``): repeat
  the deterministic sampling gate, snapshot KV slots over prompt + committed
  decode rows, and enqueue the export job.

Coverage is every forwarded token: prompt rows (prefill capture) plus
verify-committed decode rows (the final sampled token has no hidden row).
Mooncake sink and DP-attention multi-writer status: see the fail-closed
matrix in ``create``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import random
import re
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.hidden_host import (
    DeviceTwinPool,
    HiddenCaptureBookkeeper,
    HiddenCapturePressureController,
    HiddenCaptureStats,
    HiddenFinalizeWorker,
    HiddenHostSidecar,
    HiddenStagingRing,
    StagingSeqCounter,
    verify_blob_bytes,
)
from sglang.srt.state_capturer.hidden_sink import (
    HiddenExportJob,
    HiddenExportWorker,
    HiddenFileSink,
)

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
        SpecAuxHiddenStateConfig,
    )
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# rids matching this are used verbatim as sample ids (uuid4().hex shape);
# anything else (caller-supplied) is hashed to a safe unique token.
_SAFE_RID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")

# HBM token-slot budget for the verify twin pool when the geometry is not
# pinned via env vars: slots = budget / verify_window_tokens (min 2). Must
# cover the worst measured in-flight burst with margin below the degrade
# watermark: large prefill D2H can delay twin recycling, so peak occupancy
# exceeds steady state. 16384 tokens = 32 slots at a 512-token window
# (~800MB at Qwen3-8B widths). Sizing measurements: PR #3/#6.
_VERIFY_TWIN_BUDGET_TOKENS = 16384

# Pinned-host token-slot budget for the staging ring when its geometry is not
# pinned via env vars: slots = budget / slot_tokens (min 2). 32768 total slots
# matches the old fixed 4x8192 default (~1.6GB pinned at Qwen3-8B widths).
_STAGING_RING_BUDGET_TOKENS = 32768

# Pinned-host token-slot budget for the verify staging ring (slot size = the
# verify window). Verify must not share slots with prefill: a prefill
# finalize memcpy takes tens of ms, and a shared ring going transiently full
# during that drain drops whole verify batches. Depth must absorb a full
# prefill-finalize stall. Sizing measurements: PR #3.
_VERIFY_STAGING_BUDGET_TOKENS = 16384


def _resolve_verify_window_tokens(
    *, server_args: ServerArgs, max_running_requests: int
) -> int:
    """Upper bound on one verify step's rows: bs x verify width, rounded up
    to 256 so near-boundary bs growth doesn't force misses. DSpark's verify
    width (anchor + gamma drafts) equals speculative_num_draft_tokens
    (dspark_config: gamma = num_draft_tokens - 1, width = gamma + 1)."""
    verify_width = server_args.speculative_num_draft_tokens or 1
    window = max(1, max_running_requests) * verify_width
    return (window + 255) // 256 * 256


def _resolve_staging_slot_tokens(*, server_args: ServerArgs) -> int:
    """Upper bound on one prefill forward's staged rows, rounded up to 256.

    The PrefillAdder caps a batch's extend rows at chunked_prefill_size
    (rem_chunk_tokens is a batch-level budget); with chunking disabled (-1)
    the bound becomes max_prefill_tokens. A slot sized to that bound never
    needs the multi-segment path in the common case (the split remains as a
    safety net; real bound can exceed max_prefill_tokens for models with a
    longer context, per its help text). Verify windows stage through their
    own ring and don't constrain this size.
    """
    chunk = server_args.chunked_prefill_size
    if chunk is None or chunk <= 0:
        prefill_bound = server_args.max_prefill_tokens
    else:
        prefill_bound = chunk
    bound = max(prefill_bound, 256)
    return (bound + 255) // 256 * 256


def _resolve_sidecar_capacity_tokens(
    *,
    num_tokens: int,
    staging_slot_tokens: int,
    verify_window_tokens: int,
) -> int:
    """Compact sidecar capacity in tokens.

    Sized for full-rate capture (inside a sampling window every request is
    captured). A floor keeps one maximum export or staging/verify window
    representable; saturation beyond the budget is an attributable miss and
    pressure-degraded window, never a reason to allocate the whole KV pool.
    """
    budget = envs.SGLANG_HIDDEN_CAPTURE_SIDECAR_TOKEN_BUDGET.get()
    floor = max(
        1,
        envs.SGLANG_HIDDEN_CAPTURE_MAX_EXPORT_TOKENS.get(),
        staging_slot_tokens,
        verify_window_tokens,
    )
    return min(num_tokens, max(floor, budget))


class _WindowSampler:
    """Time-window request sampling: capture everything inside an open
    window of ``window_s`` seconds once per ``period_s``-second cycle.

    The in/out decision for a request is made the first time any gate sees
    its rid and memoized for the request's whole lifetime, so a request never
    straddles a window edge with half its rows captured. Each cycle draws a
    fresh uniform window offset (unless ``phase_s`` pins it), so no
    time-of-cycle is systematically over- or under-sampled.

    Called from the scheduler thread only (all three gates run there); the
    memo needs no lock. ``window_s == period_s`` captures every request.
    """

    def __init__(
        self,
        *,
        window_s: float,
        period_s: float,
        phase_s: Optional[float] = None,
        clock=time.monotonic,
        rng=None,
    ) -> None:
        self.window_s = float(window_s)
        self.period_s = float(period_s)
        self._fixed_phase_s = None if phase_s is None else float(phase_s)
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()
        self._epoch_s = self._clock()
        self._cycle_index = -1
        self._cycle_offset_s = 0.0
        self._decisions: Dict[str, bool] = {}

    def _offset_for_cycle(self) -> float:
        slack = self.period_s - self.window_s
        if slack <= 0.0:
            return 0.0
        if self._fixed_phase_s is not None:
            return min(self._fixed_phase_s, slack)
        return self._rng.uniform(0.0, slack)

    def window_open(self) -> bool:
        elapsed = self._clock() - self._epoch_s
        cycle = int(elapsed // self.period_s)
        if cycle != self._cycle_index:
            self._cycle_index = cycle
            self._cycle_offset_s = self._offset_for_cycle()
        in_cycle = elapsed - cycle * self.period_s
        return self._cycle_offset_s <= in_cycle < self._cycle_offset_s + self.window_s

    def sampled(self, rid: str) -> bool:
        decision = self._decisions.get(rid)
        if decision is None:
            decision = self.window_open()
            self._decisions[rid] = decision
        return decision

    def forget(self, rid: str) -> None:
        self._decisions.pop(rid, None)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "window_open": self.window_open(),
            "window_s": self.window_s,
            "period_s": self.period_s,
            "cycle_offset_s": self._cycle_offset_s,
            "pending_decisions": len(self._decisions),
        }


def _serving_span(name: str):
    """Record one serving-side capture hook without changing its returns."""

    def decorate(method):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            started_ns = time.monotonic_ns()
            try:
                return method(self, *args, **kwargs)
            finally:
                stats = getattr(self, "stats", None)
                if stats is not None:
                    stats.observe_duration(name, time.monotonic_ns() - started_ns)

        return wrapped

    return decorate


def _disabled(reason: str) -> None:
    logger.warning("hidden state capture disabled: %s", reason)


def _resolve_aux_layer_ids(spec_aux_config: SpecAuxHiddenStateConfig):
    if spec_aux_config.dflash_use_aux_hidden_state:
        return spec_aux_config.dflash_target_layer_ids
    if spec_aux_config.eagle_use_aux_hidden_state:
        return spec_aux_config.eagle_aux_hidden_state_layer_ids
    return None


class HiddenCaptureOutput(msgspec.Struct):
    """Per-forward GPU references pending async D2H (overlap scheduling).

    ``stage()`` runs on the scheduler's copy stream inside
    ``GenerationBatchResult.copy_to_cpu`` and either commits every row into
    the staging ring or marks every affected request as capture-miss.
    """

    aux_hidden_states: torch.Tensor  # [T, K*H] packed aux (GPU ref)
    last_hidden_states: torch.Tensor  # [T, H] post-final-norm (GPU ref)
    out_cache_loc: torch.Tensor  # [T] kv token slots (GPU ref)
    input_tokens: torch.Tensor  # [T] forwarded token ids (GPU ref)
    req_ranges: List[Tuple[str, int, int]]  # (rid, start_row, end_row)
    capturer: HiddenStatesCapturer

    def stage(self) -> None:
        self.capturer.stage(self)


class HiddenVerifyCaptureOutput(msgspec.Struct):
    """One verify step's window, already packed into a device twin (fenced).

    Unlike ``HiddenCaptureOutput`` this holds no references into graph or
    persistent buffers — the twin is capture-owned, so the next replay can't
    overwrite it. ``stage()`` D2Hs twin -> pinned ring off-stream.
    """

    twin: Any
    rids: List[str]
    admitted_rids: List[str]
    stride: int
    num_reqs: int
    capturer: HiddenStatesCapturer

    def stage(self) -> None:
        self.capturer.stage_verify(self)


class HiddenStatesCapturer:
    """Owns the staging ring, host sidecar, and worker threads for capture."""

    @staticmethod
    def create(
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        spec_aux_config: SpecAuxHiddenStateConfig,
        num_tokens: int,
        max_running_requests: int,
        dp_rank: Optional[int] = None,
    ) -> Optional[HiddenStatesCapturer]:
        if not server_args.enable_hidden_state_capture:
            return None

        sink_kind = envs.SGLANG_HIDDEN_CAPTURE_SINK.get()
        sink_dir = envs.SGLANG_HIDDEN_CAPTURE_DIR.get()
        if sink_kind == "file" and not sink_dir:
            _disabled("SGLANG_HIDDEN_CAPTURE_DIR is not set (file sink)")
            return None
        if sink_kind == "mooncake" and not envs.MOONCAKE_MASTER.get():
            _disabled("MOONCAKE_MASTER is not set (mooncake sink)")
            return None
        if sink_kind not in ("file", "mooncake"):
            _disabled(f"unknown SGLANG_HIDDEN_CAPTURE_SINK={sink_kind!r}")
            return None
        sample_window_s = envs.SGLANG_HIDDEN_CAPTURE_WINDOW_S.get()
        sample_period_s = envs.SGLANG_HIDDEN_CAPTURE_PERIOD_S.get()
        if not 0.0 < sample_window_s <= sample_period_s:
            _disabled(
                "sampling window requires 0 < WINDOW_S <= PERIOD_S, got "
                f"window={sample_window_s}, period={sample_period_s}"
            )
            return None
        pressure_high = envs.SGLANG_HIDDEN_CAPTURE_DEGRADE_HIGH_WATERMARK.get()
        pressure_recover = envs.SGLANG_HIDDEN_CAPTURE_DEGRADE_RECOVER_WATERMARK.get()
        if not 0.0 <= pressure_recover < pressure_high <= 1.0:
            _disabled(
                "capture pressure watermarks require 0 <= recover < high <= 1, "
                f"got recover={pressure_recover}, high={pressure_high}"
            )
            return None
        aux_layer_ids = _resolve_aux_layer_ids(spec_aux_config)
        # Current support matrix; anything outside fails closed (no partial data).
        gates = [
            (
                not aux_layer_ids,
                "no aux hidden state layers configured "
                "(requires a DFlash/DSpark or EAGLE3 draft with aux capture)",
            ),
            (
                server_args.enable_dp_attention,
                "DP attention would capture per-rank shards",
            ),
            (server_args.attn_cp_size > 1, "context parallelism is unsupported"),
            (server_args.pp_size > 1, "pipeline parallelism is unsupported"),
            (
                server_args.disaggregation_mode != "null",
                "PD disaggregation is unsupported",
            ),
            (
                server_args.enable_hierarchical_cache,
                "HiCache relocation invalidates kv-slot identity (M2+)",
            ),
            (
                server_args.disable_overlap_schedule,
                "capture rides the overlap result-copy stream",
            ),
            (server_args.dllm_algorithm is not None, "dLLM is unsupported"),
            (
                server_args.enable_mixed_chunk,
                "mixed prefill/decode batches are unsupported",
            ),
        ]
        for failed, reason in gates:
            if failed:
                _disabled(reason)
                return None

        # TP replicates the target hidden states across attention-TP ranks;
        # a single writer per (DP-replica) scheduler is correct and avoids
        # duplicate files. Non-writer ranks simply run without a capturer.
        if get_parallel().attn_tp_rank != 0:
            return None

        if getattr(server_args, "disable_cuda_graph", False):
            logger.warning(
                "hidden state capture with CUDA graph disabled is unsupported/"
                "experimental; production capture must use graph-on"
            )

        verify_window_tokens = _resolve_verify_window_tokens(
            server_args=server_args, max_running_requests=max_running_requests
        )
        staging_slot_tokens = _resolve_staging_slot_tokens(server_args=server_args)
        sidecar_capacity_tokens = _resolve_sidecar_capacity_tokens(
            num_tokens=num_tokens,
            staging_slot_tokens=staging_slot_tokens,
            verify_window_tokens=verify_window_tokens,
        )
        # Host memory gate: hidden payload follows the compact sampled/inflight
        # budget. Only the small generation/token identity maps retain KV-pool
        # cardinality.
        hidden_size = model_config.hf_text_config.hidden_size
        row_bytes = (len(aux_layer_ids) + 1) * hidden_size * model_config.dtype.itemsize
        sidecar_bytes = sidecar_capacity_tokens * row_bytes + num_tokens * (
            torch.int32.itemsize + torch.int64.itemsize
        )
        sidecar_gb = sidecar_bytes / 1024**3
        max_gb = envs.SGLANG_HIDDEN_CAPTURE_MAX_HOST_GB.get()
        if sidecar_gb > max_gb:
            _disabled(
                f"host sidecar would need {sidecar_gb:.1f} GB "
                f"(compact_capacity={sidecar_capacity_tokens} rows, "
                f"logical_slots={num_tokens}, payload={row_bytes} B/row) > "
                f"SGLANG_HIDDEN_CAPTURE_MAX_HOST_GB={max_gb}. Raise the limit "
                "or reduce SGLANG_HIDDEN_CAPTURE_SIDECAR_TOKEN_BUDGET"
            )
            return None

        try:
            return HiddenStatesCapturer(
                model_config=model_config,
                num_aux_layers=len(aux_layer_ids),
                num_tokens=num_tokens,
                sidecar_capacity_tokens=sidecar_capacity_tokens,
                verify_window_tokens=verify_window_tokens,
                staging_slot_tokens=staging_slot_tokens,
                sink_kind=sink_kind,
                sink_dir=sink_dir,
                dp_rank=dp_rank if dp_rank is not None else 0,
                aux_layer_ids=list(aux_layer_ids),
                model_path=server_args.model_path,
                model_revision=server_args.revision,
            )
        except Exception as error:
            # Capture is a best-effort side channel. A missing/unhealthy sink
            # at startup must disable capture, not make the serving worker
            # unavailable. The constructor starts workers only after sink and
            # fingerprint initialization succeed, and each failure path
            # releases any sink-owned registered/client resources.
            logger.exception("hidden state capture initialization failed")
            _disabled(f"initialization failed: {error}")
            return None

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        num_aux_layers: int,
        num_tokens: int,
        sidecar_capacity_tokens: int,
        verify_window_tokens: int,
        staging_slot_tokens: int,
        sink_kind: str,
        sink_dir: Optional[str],
        dp_rank: int,
        aux_layer_ids: List[int],
        model_path: str,
        model_revision: Optional[str],
    ) -> None:
        hidden_size = model_config.hf_text_config.hidden_size
        self.aux_width = num_aux_layers * hidden_size
        self.last_width = hidden_size
        self.dtype = model_config.dtype
        self.sampler = _WindowSampler(
            window_s=envs.SGLANG_HIDDEN_CAPTURE_WINDOW_S.get(),
            period_s=envs.SGLANG_HIDDEN_CAPTURE_PERIOD_S.get(),
            phase_s=envs.SGLANG_HIDDEN_CAPTURE_PHASE_S.get(),
        )
        # Contract check state: weakref to the previous graph forward's aux
        # (aux must be a fresh tensor every forward; see on_forward_end).
        self._last_graph_aux: Optional[weakref.ref] = None
        self._accepting = threading.Event()
        self._accepting.set()
        self._close_lock = threading.Lock()
        self._closed = False

        self.stats = HiddenCaptureStats()
        self.bookkeeper = HiddenCaptureBookkeeper()
        # Dedicated stream for capture D2H. The scheduler's copy stream is
        # FIFO: queueing capture's large copies there would make the NEXT
        # step's (tiny) result copies — and thus its copy_done — wait behind
        # them, leaking capture cost into serving tail latency. The scheduler
        # thread is the ONLY submitter on this stream (PyTorch stream
        # submission is thread-local), so event completion order on it equals
        # enqueue order — the rings' ordered polling depends on this.
        self.capture_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        # Finish-hook kv-slot snapshots: a dedicated stream + reusable pinned
        # buffer. A plain .cpu() on the scheduler thread is a synchronous
        # PAGEABLE D2H that serializes behind whatever the copy engine is
        # doing — including this capturer's own ~50MB staging transfers —
        # blocking the scheduler for ms at a time (measured: the entire
        # capture-on p99 TPOT tail, 6.5 -> 7.9ms, came from this one copy).
        self.snapshot_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._snapshot_buf = (
            torch.empty((num_tokens,), dtype=torch.int32, pin_memory=True)
            if torch.cuda.is_available()
            else None
        )
        self._snapshot_event = torch.cuda.Event() if torch.cuda.is_available() else None
        # Staging ring geometry adapts like the twin pool's: slot size = one
        # forward's staged-row bound (chunked_prefill_size, or
        # max_prefill_tokens when chunking is disabled; floored to the verify
        # window), slot count = pinned budget / slot size. A chunk-128
        # deployment thus gets 64 slots of 512 instead of
        # 4x8192 (98% dead capacity, concurrency capped at 4); the 8192-chunk
        # default reproduces the old 4x8192. Env vars pin either dimension.
        ring_slot_tokens = (
            envs.SGLANG_HIDDEN_CAPTURE_STAGING_SLOT_TOKENS.get()
            if envs.SGLANG_HIDDEN_CAPTURE_STAGING_SLOT_TOKENS.is_set()
            else staging_slot_tokens
        )
        ring_slots = (
            envs.SGLANG_HIDDEN_CAPTURE_STAGING_SLOTS.get()
            if envs.SGLANG_HIDDEN_CAPTURE_STAGING_SLOTS.is_set()
            else max(2, _STAGING_RING_BUDGET_TOKENS // max(1, ring_slot_tokens))
        )
        # Prefill and verify rings share one enqueue-sequence counter (and
        # one capture stream), so the export barrier stays a single number
        # and the finalize thread can merge the rings in enqueue order.
        self.staging_seq = StagingSeqCounter()
        self.ring = HiddenStagingRing(
            num_slots=ring_slots,
            slot_tokens=ring_slot_tokens,
            aux_width=self.aux_width,
            last_width=self.last_width,
            dtype=self.dtype,
            seq_counter=self.staging_seq,
            stats=self.stats,
            stats_prefix="prefill",
        )
        self.sidecar = HiddenHostSidecar(
            num_slots=num_tokens,
            capacity_tokens=sidecar_capacity_tokens,
            aux_width=self.aux_width,
            last_width=self.last_width,
            dtype=self.dtype,
            stats=self.stats,
        )
        # Verify capture: device twins are the overwrite fence for graph/
        # persistent buffers (packed on the forward stream; see _DeviceTwin).
        # Geometry adapts to the actual verify window (bs x draft tokens):
        # slots = HBM budget / window, so small deployments get deep pools
        # (fast decode steps need many in-flight twins) and large-batch ones
        # still fit their window. Explicit env vars override both.
        twin_tokens = (
            envs.SGLANG_HIDDEN_CAPTURE_VERIFY_RING_TOKENS.get()
            if envs.SGLANG_HIDDEN_CAPTURE_VERIFY_RING_TOKENS.is_set()
            else verify_window_tokens
        )
        num_twins = (
            envs.SGLANG_HIDDEN_CAPTURE_VERIFY_RING_SLOTS.get()
            if envs.SGLANG_HIDDEN_CAPTURE_VERIFY_RING_SLOTS.is_set()
            else max(
                2,
                _VERIFY_TWIN_BUDGET_TOKENS // max(1, twin_tokens),
            )
        )
        self.twin_pool = (
            DeviceTwinPool(
                num_twins=num_twins,
                twin_tokens=twin_tokens,
                max_reqs=512,
                aux_width=self.aux_width,
                last_width=self.last_width,
                dtype=self.dtype,
                stats=self.stats,
            )
            if torch.cuda.is_available()
            else None
        )
        # Verify windows stage through their own ring, sized to the window.
        # Sharing the prefill ring gave verify an effective depth of 4 slots
        # each 95% empty (a 384-row window burned an 8192-token slot), and a
        # prefill finalize (hundreds-of-MB memcpy) stalls draining long
        # enough for decode steps to fill that: at saturation 2 transient
        # full events dropped 75/300 samples, a whole decode batch per event.
        verify_staging_slots = (
            envs.SGLANG_HIDDEN_CAPTURE_VERIFY_STAGING_SLOTS.get()
            if envs.SGLANG_HIDDEN_CAPTURE_VERIFY_STAGING_SLOTS.is_set()
            else max(2, _VERIFY_STAGING_BUDGET_TOKENS // max(1, twin_tokens))
        )
        self.verify_ring = HiddenStagingRing(
            num_slots=verify_staging_slots,
            slot_tokens=twin_tokens,
            aux_width=self.aux_width,
            last_width=self.last_width,
            dtype=self.dtype,
            seq_counter=self.staging_seq,
            stats=self.stats,
            stats_prefix="verify",
            max_verify_reqs=512,
        )
        self.finalize_worker = HiddenFinalizeWorker(
            ring=self.ring,
            verify_ring=self.verify_ring,
            sidecar=self.sidecar,
            bookkeeper=self.bookkeeper,
            stats=self.stats,
            twin_pool=self.twin_pool,
        )
        self.sink = self._build_sink(sink_kind, sink_dir, dp_rank)
        fingerprint = {
            "model_path": model_path,
            "model_revision": model_revision,
            "dtype": str(self.dtype),
            "aux_layer_ids": aux_layer_ids,
            "hidden_size": hidden_size,
            "num_aux_layers": num_aux_layers,
            "norm_contract": "post_final_norm_pre_lm_head",
            "aux_layout": "packed_last_dim",  # [T, K*H], serving layout
            # Rows cover the prompt region plus verify-committed decode tokens
            # (the final sampled token has no hidden row).
            "coverage": "prefill_and_verify_commit",
            "coverage_semantics": (
                "full_below_pressure_threshold; fail_closed_degraded_windows"
            ),
            "degradation_observability": (
                "/server_info internal_states[].hidden_capture.degradation"
            ),
            # Recompute the real role-aware mask offline before training.
            "loss_mask": "all_ones_placeholder",
            "storage_schema": (
                "hidden-sample-view-v1"
                if getattr(self.sink, "prefix_enabled", False)
                else "hidden-whole-sample-v0"
            ),
        }
        try:
            self.sink.write_fingerprint(fingerprint)
        except BaseException:
            close_sink = getattr(self.sink, "close", None)
            if callable(close_sink):
                try:
                    close_sink()
                except Exception:
                    logger.exception(
                        "hidden capture sink cleanup failed after fingerprint error"
                    )
            raise
        self.export_worker = HiddenExportWorker(
            sidecar=self.sidecar,
            bookkeeper=self.bookkeeper,
            finalize_worker=self.finalize_worker,
            stats=self.stats,
            sink=self.sink,
            queue_size=envs.SGLANG_HIDDEN_CAPTURE_EXPORT_QUEUE_SIZE.get(),
        )
        self.pressure_controller = HiddenCapturePressureController(
            stats=self.stats,
            high_watermark=(envs.SGLANG_HIDDEN_CAPTURE_DEGRADE_HIGH_WATERMARK.get()),
            recover_watermark=(
                envs.SGLANG_HIDDEN_CAPTURE_DEGRADE_RECOVER_WATERMARK.get()
            ),
            min_degraded_s=envs.SGLANG_HIDDEN_CAPTURE_MIN_DEGRADED_S.get(),
        )
        # Start an unavailable-Mooncake window during initialization rather
        # than waiting for the first request hook to notice it.
        self.pressure_controller.refresh(self._pressure_signals())
        self.finalize_worker.start()
        self.export_worker.start()
        logger.info(
            "hidden state capture enabled: sink=%s (dir=%s), aux_layer_ids=%s, "
            "sampling window=%.0fs/%.0fs",
            sink_kind,
            sink_dir,
            aux_layer_ids,
            self.sampler.window_s,
            self.sampler.period_s,
        )

    def _build_sink(self, sink_kind: str, sink_dir: Optional[str], dp_rank: int):
        if sink_kind == "mooncake":
            from sglang.srt.state_capturer.hidden_mooncake import (
                AsyncMooncakeHiddenSink,
            )

            # +8 bytes/row: input_ids ride the same registered staging buffer.
            return AsyncMooncakeHiddenSink(
                initial_probe_timeout_s=(
                    envs.SGLANG_HIDDEN_CAPTURE_MOONCAKE_PROBE_TIMEOUT_S.get()
                ),
                reconnect_interval_s=(
                    envs.SGLANG_HIDDEN_CAPTURE_MOONCAKE_RECONNECT_INTERVAL_S.get()
                ),
                store_id=envs.SGLANG_HIDDEN_CAPTURE_STORE_ID.get(),
                max_export_tokens=envs.SGLANG_HIDDEN_CAPTURE_MAX_EXPORT_TOKENS.get(),
                dp_rank=dp_rank,
                stats=self.stats,
                max_segment_rows=(
                    envs.SGLANG_HIDDEN_CAPTURE_PREFIX_MAX_SEGMENT_ROWS.get()
                ),
                prefix_lanes=envs.SGLANG_HIDDEN_CAPTURE_PREFIX_LANES.get(),
                aux_width=self.aux_width,
                last_width=self.last_width,
                dtype=self.dtype,
            )
        return HiddenFileSink(sink_dir)

    def close(self, timeout_s: Optional[float] = None) -> bool:
        """Stop admission, drain both workers, then release sink resources.

        The registered Mooncake arena is unregistered only after the export
        thread has exited. A timeout leaves the sink registered and returns
        False, keeping shutdown bounded without a use-after-unregister.
        """
        if timeout_s is None:
            timeout_s = envs.SGLANG_HIDDEN_CAPTURE_SHUTDOWN_TIMEOUT_S.get()
        with self._close_lock:
            if self._closed:
                return True

            self._accepting.clear()
            self.export_worker.stop_admission()
            self.finalize_worker.stop()
            self.export_worker.stop()
            deadline = time.monotonic() + max(0.0, timeout_s)

            finalize_done = self.finalize_worker.join(
                max(0.0, deadline - time.monotonic())
            )
            if not finalize_done:
                self.stats.bump(
                    "shutdown_finalize_timeout_miss_ct",
                    max(1, self.finalize_worker.pending_count),
                )

            export_done = self.export_worker.join(max(0.0, deadline - time.monotonic()))
            if not export_done:
                self.stats.bump(
                    "shutdown_export_timeout_miss_ct",
                    max(1, self.export_worker.pending_count),
                )

            if finalize_done and export_done:
                close_sink = getattr(self.sink, "close", None)
                if callable(close_sink):
                    close_sink()
                self._closed = True
            pressure_controller = getattr(self, "pressure_controller", None)
            if pressure_controller is not None:
                pressure_controller.close()
            self.stats.log("hidden capture final stats:")
            return self._closed

    def _pressure_signals(self) -> Dict[str, float]:
        sink_ready = bool(getattr(self.sink, "ready", True))
        return {
            "sink": 0.0 if sink_ready else 1.0,
            "export_queue": self.export_worker.pending_count
            / max(1, self.export_worker.capacity),
            "prefill_ring": self.ring.inflight_count / max(1, self.ring.num_slots),
            "verify_ring": self.verify_ring.inflight_count
            / max(1, self.verify_ring.num_slots),
            "verify_twin": (
                self.twin_pool.in_use_count / max(1, self.twin_pool.num_twins)
                if self.twin_pool is not None
                else 0.0
            ),
            "sidecar": self.sidecar.pressure_ratio,
        }

    def _capture_allowed(
        self,
        *,
        rids: Sequence[str],
        rows: int,
        phase: str,
    ) -> bool:
        controller = getattr(self, "pressure_controller", None)
        if controller is None or controller.refresh(self._pressure_signals()):
            return True
        sample_misses = self.bookkeeper.mark_miss(rids)
        controller.record_drop(
            phase=phase,
            requests=len(rids),
            rows=rows,
            sample_misses=sample_misses,
        )
        return False

    def _settle_captured_rows(self, own_slot_gens: Dict[int, int]) -> None:
        self.sidecar.release_rows(
            own_slot_gens,
            discard=bool(getattr(self.sink, "prefix_enabled", False)),
        )

    def _submit_cleanup_job(self, rid: str, global_barrier: int) -> None:
        barrier, _has_own_rows = self.bookkeeper.export_barrier(rid, global_barrier)
        self.export_worker.submit(
            HiddenExportJob(
                rid=rid,
                sample_id=self._sample_id_for(rid),
                tokens=torch.empty((0,), dtype=torch.long),
                slots=torch.empty((0,), dtype=torch.long),
                ring_seq_barrier=barrier,
            )
        )

    def observability_snapshot(self) -> Dict[str, Any]:
        """Return a read-only, point-in-time capture/resource snapshot.

        This is called only by the internal-state endpoint.  It deliberately
        avoids synchronizing CUDA or waiting for workers, so characterization
        polling cannot become serving backpressure.
        """

        def _alive(worker: Any) -> bool:
            thread = getattr(worker, "_thread", None)
            return bool(thread is not None and thread.is_alive())

        # Polling is also a non-blocking pressure refresh, so an idle server's
        # startup-reconnect window closes at observation time instead of being
        # left open until the next request.
        self.pressure_controller.refresh(self._pressure_signals())
        sink_state = getattr(self.sink, "state_snapshot", None)
        snapshot_buf_bytes = (
            self._snapshot_buf.numel() * self._snapshot_buf.element_size()
            if self._snapshot_buf is not None
            else 0
        )
        return {
            "stats": self.stats.snapshot(),
            "critical_path": self.stats.critical_path_snapshot(),
            "serving_spans": self.stats.serving_span_snapshot(),
            "degradation": self.pressure_controller.snapshot(),
            "sampling": self.sampler.snapshot(),
            "state": {
                "accepting": self._accepting.is_set(),
                "closed": self._closed,
                "prefill_ring_inflight": self.ring.inflight_count,
                "verify_ring_inflight": self.verify_ring.inflight_count,
                "finalize_worker_active": self.finalize_worker._active,
                "export_queue_pending": self.export_worker.pending_count,
                "export_worker_active": self.export_worker._active,
                "last_enqueued_seq": self.ring.last_enqueued_seq,
                "last_finalized_seq": self.finalize_worker.last_finalized_seq,
                "finalize_worker_alive": _alive(self.finalize_worker),
                "export_worker_alive": _alive(self.export_worker),
                "bookkeeper": self.bookkeeper.state_snapshot(),
                "sidecar": self.sidecar.state_snapshot(),
                "sink": sink_state() if callable(sink_state) else None,
            },
            "resources": {
                "prefill_ring_pinned_bytes": self.ring.allocated_bytes,
                "verify_ring_pinned_bytes": self.verify_ring.allocated_bytes,
                "finish_snapshot_pinned_bytes": snapshot_buf_bytes,
                "sidecar_pageable_bytes": self.sidecar.allocated_bytes,
                "verify_twin_hbm_bytes": (
                    self.twin_pool.allocated_bytes if self.twin_pool is not None else 0
                ),
                "mooncake_registered_bytes": int(
                    getattr(self.sink, "registered_bytes", 0)
                ),
            },
        }

    # ---------------------------------------------------------------- forward

    @_serving_span("serving_forward_hook")
    def on_forward_end(
        self,
        *,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        can_run_graph: bool,
    ) -> Optional[HiddenCaptureOutput]:
        """Build the pending-D2H holder for an extend forward, or None.

        Runs on the forward thread before DSpark's worker consumes and clears
        ``logits_output.hidden_states``; only references are taken here.
        """
        if not self._accepting.is_set():
            rids = forward_batch.rids or []
            self.stats.bump("shutdown_admission_miss_ct", len(rids) or 1)
            self.bookkeeper.mark_miss(rids)
            return None
        if not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            return None
        aux = logits_output.hidden_states
        last = logits_output.last_hidden_states
        if aux is None or last is None:
            return None
        rids = forward_batch.rids
        extend_lens = forward_batch.extend_seq_lens_cpu
        if rids is None or extend_lens is None:
            return None
        if forward_batch.forward_mode.is_split_prefill():
            # forward_batch_split_prefill builds its GenerationBatchResult
            # without the capture holder, so rows would silently never stage.
            # Fail closed with attribution instead.
            self.stats.bump("skipped_forward_ct")
            self.bookkeeper.mark_miss(rids)
            return None
        num_rows = aux.shape[0]
        if sum(extend_lens) != num_rows or last.shape[0] != num_rows:
            # Row attribution would be wrong; drop the whole forward.
            self.stats.bump("attach_mismatch_miss_ct")
            self.bookkeeper.mark_miss(rids)
            return None

        req_ranges = []
        row = 0
        for rid, extend_len in zip(rids, extend_lens):
            if not rid.startswith(HEALTH_CHECK_RID_PREFIX) and self._sampled(rid):
                req_ranges.append((rid, row, row + extend_len))
            row += extend_len
        if not req_ranges:
            return None
        admitted_rows = sum(end - start for _, start, end in req_ranges)
        if not self._capture_allowed(
            rids=[rid for rid, _, _ in req_ranges],
            rows=admitted_rows,
            phase="forward",
        ):
            return None
        if can_run_graph:
            # Prefill CUDA graph (BCG/Full): the graph captures only the
            # transformer body; the LM-head/logits tail runs eagerly. The
            # packed aux is therefore a FRESH tensor every forward (eager
            # pack_aux_hidden_states cat; Qwen3 returns an aux list, not a
            # pre-packed static buffer) and record_stream protection applies.
            # The post-norm last hidden, however, is a view of the captured
            # body's shared static output buffer, overwritten in place by the
            # next replay — clone it on the forward stream: same-stream
            # ordering is the overwrite fence (the next replay queues behind
            # the clone), identical in principle to the verify device twin.
            #
            # Runtime contract check: the clone asymmetry above is only
            # correct while aux really is a fresh allocation per forward. If
            # the model side ever returns a persistent buffer, a replay would
            # overwrite rows we still reference — fail closed with
            # attribution instead of exporting corrupted training data.
            # A bare pointer compare is NOT enough: the caching allocator
            # legitimately reuses a freed aux's address for the next
            # forward's fresh tensor. Only "previous aux still alive at the
            # same address" proves aliasing (live allocations never share an
            # address with new ones), so the check holds a weakref.
            prev_aux = (
                self._last_graph_aux() if self._last_graph_aux is not None else None
            )
            if prev_aux is not None and prev_aux.data_ptr() == aux.data_ptr():
                self.stats.bump("aux_not_fresh_miss_ct")
                self.bookkeeper.mark_miss([rid for rid, _, _ in req_ranges])
                return None
            self._last_graph_aux = weakref.ref(aux)
            last = last.clone()
        return HiddenCaptureOutput(
            aux_hidden_states=aux,
            last_hidden_states=last,
            out_cache_loc=forward_batch.out_cache_loc,
            input_tokens=forward_batch.input_ids,
            req_ranges=req_ranges,
            capturer=self,
        )

    # ------------------------------------------------------------------ verify

    @_serving_span("serving_verify_hook")
    def capture_verify_window(
        self,
        *,
        rids: List[str],
        aux_strided: torch.Tensor,  # [bs*stride, K*H] (may be graph-persistent)
        verify_cache_loc: torch.Tensor,  # [bs*stride] kv slots
        verify_tokens: torch.Tensor,  # [bs*stride] token ids forwarded
        commit_lens: torch.Tensor,  # [bs] accepted-prefix lens (incl. anchor)
        bs: int,
        stride: int,
        last_strided: Optional[torch.Tensor] = None,  # [bs*stride, H]
        last_compact: Optional[torch.Tensor] = None,  # [total_verify_tokens, H]
        verify_lens: Optional[torch.Tensor] = None,  # [bs], for compact scatter
    ) -> Optional[HiddenVerifyCaptureOutput]:
        """Verify-step capture hook (DSpark worker, forward stream, after
        acceptance and before the worker drops its hidden reference).

        The sources may be CUDA-graph static or warmup-persistent buffers that
        step t+1's replay overwrites in place. The fence is same-stream
        ordering: this method packs the window into a capture-owned device
        twin ON THE CURRENT (forward) STREAM, so the next step's kernels queue
        behind the pack, without stalling forward on any D2H. The twin is
        then D2H'd off-stream by stage().

        The post-norm last hidden arrives strided on the dense verify path or
        compact (ragged) on the compact path. Committed-row mode reads the
        ragged source through verify-lens offsets directly into the compact
        twin; the full-window fallback scatters it to the strided layout.
        """
        if not self._accepting.is_set():
            self.stats.bump("shutdown_admission_miss_ct", len(rids) or 1)
            self.bookkeeper.mark_miss(rids)
            return None
        if self.twin_pool is None:
            return None
        num_rows = bs * stride
        admitted_mask = [
            not rid.startswith(HEALTH_CHECK_RID_PREFIX) and self._sampled(rid)
            for rid in rids
        ]
        admitted_rids = [rid for rid, admitted in zip(rids, admitted_mask) if admitted]
        if not admitted_rids:
            return None
        if not self._capture_allowed(
            rids=admitted_rids,
            rows=len(admitted_rids) * stride,
            phase="verify",
        ):
            return None
        if all(admitted_mask):
            capture_commit_lens = commit_lens
        else:
            mask = torch.tensor(
                admitted_mask,
                dtype=commit_lens.dtype,
                device=commit_lens.device,
            )
            capture_commit_lens = commit_lens * mask
        if num_rows > self.twin_pool.twin_tokens or bs > self.twin_pool.max_reqs:
            self.stats.bump("verify_oversize_miss_ct")
            self.bookkeeper.mark_miss(admitted_rids)
            return None
        if last_strided is None and last_compact is None:
            # Post-norm hidden unavailable (e.g. a graph captured before the
            # capturer was installed): fail closed with attribution.
            self.stats.bump("skipped_forward_ct")
            self.bookkeeper.mark_miss(admitted_rids)
            return None
        # Twins whose D2H already finished are recyclable right now — reap
        # them on this thread instead of waiting for the finalize thread to
        # chew through the queue (its sidecar memcpy holds twins for tens of
        # ms and starves fast decode steps).
        twin = self.twin_pool.try_acquire()
        if twin is None:
            self.verify_ring.reap_ready_twins(self.twin_pool)
            twin = self.twin_pool.try_acquire()
        if twin is None:
            self.stats.bump("verify_twin_full_miss_ct")
            self.bookkeeper.mark_miss(admitted_rids)
            return None

        try:
            # Pack on the caller's (forward) stream: this is the overwrite
            # fence. Timing events are consumed only after fence completion by
            # the background/reap path, never synchronized here.
            twin.pack_start_event.record()
            from sglang.srt.state_capturer.hidden_pack import (
                pack_committed_verify_rows_into,
            )

            views = self.twin_pool.views(twin, bs=bs, stride=stride)
            pack_committed_verify_rows_into(
                aux_strided=aux_strided,
                last_strided=last_strided,
                last_compact=last_compact,
                verify_lens=verify_lens,
                verify_cache_loc=verify_cache_loc,
                verify_tokens=verify_tokens,
                commit_lens=capture_commit_lens,
                bs=bs,
                stride=stride,
                out_aux=views.aux,
                out_last=views.last,
                out_cache_loc=views.cache_loc,
                out_tokens=views.tokens,
                out_commit_lens=views.commit_lens,
                out_commit_offsets=twin.commit_offsets,
                out_verify_offsets=twin.verify_offsets,
            )
            twin.fence_event.record()
        except Exception:
            # Capture is best-effort and must never take down serving. Return
            # the borrowed twin immediately: all attempted work was submitted
            # on the current stream, so same-stream ordering keeps reuse safe.
            logger.exception("hidden verify pack failed; dropping capture step")
            self.stats.bump("skipped_forward_ct")
            self.bookkeeper.mark_miss(admitted_rids)
            self.twin_pool.release(twin)
            return None

        return HiddenVerifyCaptureOutput(
            twin=twin,
            rids=list(rids),
            admitted_rids=admitted_rids,
            stride=stride,
            num_reqs=bs,
            capturer=self,
        )

    @_serving_span("serving_verify_stage")
    def stage_verify(self, output: HiddenVerifyCaptureOutput) -> None:
        """D2H a packed verify twin into the pinned verify ring (copy-stream
        context). Verify has its own ring: see the verify_ring construction
        comment for why sharing the prefill ring dropped whole decode batches
        at saturation."""
        if not self._accepting.is_set():
            self.stats.bump(
                "shutdown_admission_miss_ct", len(output.admitted_rids) or 1
            )
            self.bookkeeper.mark_miss(output.admitted_rids)
            self.twin_pool.release(output.twin)
            return
        if not self._capture_allowed(
            rids=output.admitted_rids,
            rows=len(output.admitted_rids) * output.stride,
            phase="verify_stage",
        ):
            self.twin_pool.release(output.twin)
            return
        slots = self.verify_ring.try_acquire(1)
        if slots is None:
            self.stats.bump("stage_full_miss_ct")
            self.stats.bump("verify_stage_full_miss_ct")
            self.bookkeeper.mark_miss(output.admitted_rids)
            self.twin_pool.release(output.twin)
            return
        (slot,) = slots

        num_rows = output.num_reqs * output.stride
        self.stats.bump("verify_candidate_rows_staged_ct", num_rows)
        if self.capture_stream is not None:
            self.capture_stream.wait_event(output.twin.fence_event)
            stream_ctx = torch.cuda.stream(self.capture_stream)
        else:
            stream_ctx = contextlib.nullcontext()
        with stream_ctx:
            ring_seq = self.verify_ring.enqueue_verify_compact(
                slot,
                twin=output.twin,
                rids=output.rids,
                stride=output.stride,
                num_reqs=output.num_reqs,
            )
        self.bookkeeper.record_enqueued(output.admitted_rids, ring_seq)
        # Upper-bound transfer: the whole bs x stride window plus the
        # commit_lens header moves in one copy (accepted cost; issue #10
        # red line 4 — precise-length transfer is the retired two-phase
        # design). rows_staged tracks copied rows; committed rows are
        # counted at finalize from the arrived commit_lens.
        self.stats.bump("rows_staged_ct", num_rows)
        self.stats.bump(
            "verify_d2h_bytes_ct",
            verify_blob_bytes(
                bs=output.num_reqs,
                n_rows=num_rows,
                aux_width=self.aux_width,
                last_width=self.last_width,
                dtype=self.dtype,
            ),
        )

    # ---------------------------------------------------------------- staging

    @_serving_span("serving_prefill_stage")
    def stage(self, output: HiddenCaptureOutput) -> None:
        """Enqueue one forward's rows into the ring (copy-stream context).

        All-or-nothing per forward: if the ring can't hold every row, every
        affected request becomes a capture-miss (never backpressure).

        The large copies are issued on the capturer's dedicated stream, which
        waits on the caller's stream (the result-copy stream, itself already
        past forward). The scheduler's copy stream stays free for the next
        step's small result copies; ring events gate the finalize thread.
        """
        affected_rids = [rid for rid, _, _ in output.req_ranges]
        if not self._accepting.is_set():
            self.stats.bump("shutdown_admission_miss_ct", len(affected_rids) or 1)
            self.bookkeeper.mark_miss(affected_rids)
            return
        num_rows = output.aux_hidden_states.shape[0]
        if num_rows == 0:
            return
        admitted_rows = sum(end - start for _, start, end in output.req_ranges)
        if not self._capture_allowed(
            rids=affected_rids,
            rows=admitted_rows,
            phase="prefill_stage",
        ):
            return
        slot_tokens = self.ring.slot_tokens
        segment_indexes = [
            segment
            for segment in range((num_rows + slot_tokens - 1) // slot_tokens)
            if any(
                start < min((segment + 1) * slot_tokens, num_rows)
                and end > segment * slot_tokens
                for _, start, end in output.req_ranges
            )
        ]
        slots = self.ring.try_acquire(len(segment_indexes))
        if slots is None:
            self.stats.bump("stage_full_miss_ct")
            self.stats.bump("prefill_stage_full_miss_ct")
            self.bookkeeper.mark_miss(affected_rids)
            return

        if self.capture_stream is not None:
            self.capture_stream.wait_stream(torch.cuda.current_stream())
            stream_ctx = torch.cuda.stream(self.capture_stream)
        else:
            stream_ctx = contextlib.nullcontext()
        with stream_ctx:
            staged_rows = 0
            for seg, slot in zip(segment_indexes, slots):
                seg_start = seg * slot_tokens
                seg_end = min(seg_start + slot_tokens, num_rows)
                staged_rows += seg_end - seg_start
                seg_ranges = [
                    (
                        rid,
                        max(r0, seg_start) - seg_start,
                        min(r1, seg_end) - seg_start,
                    )
                    for rid, r0, r1 in output.req_ranges
                    if r0 < seg_end and r1 > seg_start
                ]
                ring_seq = self.ring.enqueue_segment(
                    slot,
                    aux_rows=output.aux_hidden_states[seg_start:seg_end],
                    last_rows=output.last_hidden_states[seg_start:seg_end],
                    cache_locs=output.out_cache_loc[seg_start:seg_end],
                    tokens=output.input_tokens[seg_start:seg_end],
                    req_ranges=seg_ranges,
                )
                self.bookkeeper.record_enqueued(
                    [rid for rid, _, _ in seg_ranges], ring_seq
                )
        self.stats.bump("rows_staged_ct", staged_rows)
        self.stats.bump("prefill_rows_staged_ct", staged_rows)
        row_bytes = (self.aux_width + self.last_width) * self.dtype.itemsize + 16
        self.stats.bump("prefill_d2h_bytes_ct", staged_rows * row_bytes)

    # ----------------------------------------------------------------- finish

    def _sampled(self, rid: str) -> bool:
        return self.sampler.sampled(rid)

    @staticmethod
    def _sample_id_for(rid: str) -> str:
        """Filesystem/key-safe sample id. rid is caller-supplied (io_struct
        only generates a uuid when absent), so it can carry path separators or
        collide across requests; hash unless it's already a safe unique token.
        The original rid is preserved inside the exported record."""
        if _SAFE_RID_RE.fullmatch(rid):
            return rid
        return hashlib.sha1(rid.encode()).hexdigest()

    def _snapshot_kv_slots_batch(
        self, device_slot_ranges: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Snapshot one scheduler batch before any of its KV slots release.

        Every source range is copied into a disjoint region of one reusable
        same-dtype pinned buffer.  The dedicated stream is fenced once behind
        the producer stream and a single event is synchronized after all
        ranges, so a burst of N finishes pays one host wait instead of N.
        """
        if not device_slot_ranges:
            return []
        lengths = [int(slots.shape[0]) for slots in device_slot_ranges]
        total_rows = sum(lengths)
        stats = getattr(self, "stats", None)
        if stats is not None:
            stats.bump("finish_snapshot_batches_ct")
            stats.bump("finish_snapshot_requests_ct", len(lengths))
            stats.bump("finish_snapshot_rows_ct", total_rows)
            stats.observe_max("finish_snapshot_batch_high_water_ct", len(lengths))

        first = device_slot_ranges[0]
        if self.snapshot_stream is None or not first.is_cuda:
            return [slots.cpu().clone().to(torch.long) for slots in device_slot_ranges]
        if any(
            not slots.is_cuda or slots.device != first.device
            for slots in device_slot_ranges
        ):
            raise ValueError(
                "all batched finish slot ranges must share one CUDA device"
            )
        if any(slots.dtype != first.dtype for slots in device_slot_ranges):
            raise ValueError("all batched finish slot ranges must share one dtype")

        if self._snapshot_buf is None or self._snapshot_buf.dtype != first.dtype:
            capacity = max(1, total_rows)
            self._snapshot_buf = torch.empty(
                (capacity,), dtype=first.dtype, pin_memory=True
            )
            if stats is not None:
                stats.bump("finish_snapshot_grow_ct")
        elif total_rows > self._snapshot_buf.numel():
            capacity = 1 << (total_rows - 1).bit_length()
            self._snapshot_buf = torch.empty(
                (capacity,), dtype=first.dtype, pin_memory=True
            )
            if stats is not None:
                stats.bump("finish_snapshot_grow_ct")

        wait_started_ns = time.monotonic_ns()
        # req_to_token rows are produced on the scheduler's current stream.
        # This dependency is load-bearing: without it, the independent
        # snapshot stream can observe a stale block-table row.
        self.snapshot_stream.wait_stream(torch.cuda.current_stream(first.device))
        with torch.cuda.stream(self.snapshot_stream):
            offset = 0
            for slots, length in zip(device_slot_ranges, lengths):
                self._snapshot_buf[offset : offset + length].copy_(
                    slots, non_blocking=True
                )
                slots.record_stream(self.snapshot_stream)
                offset += length
            self._snapshot_event.record()
        self._snapshot_event.synchronize()
        if stats is not None:
            stats.bump(
                "finish_snapshot_wait_ns_ct", time.monotonic_ns() - wait_started_ns
            )
            stats.bump(
                "finish_snapshot_d2h_bytes_ct", total_rows * first.element_size()
            )

        snapshots = []
        offset = 0
        for length in lengths:
            # Convert only after D2H, on CPU, for tensor-indexing consumers.
            snapshots.append(
                self._snapshot_buf[offset : offset + length].clone().to(torch.long)
            )
            offset += length
        return snapshots

    def _snapshot_kv_slots(self, device_slots: torch.Tensor) -> torch.Tensor:
        """Single-request compatibility wrapper around the batched path."""
        return HiddenStatesCapturer._snapshot_kv_slots_batch(self, [device_slots])[0]

    @_serving_span("serving_finish_hook")
    def collect_batch_at_finish(
        self, reqs: Sequence[Req], req_to_token_pool: ReqToTokenPool
    ) -> None:
        """Batch finish hook (scheduler thread, before any KV release).

        Snapshots CPU-side state and enqueues the export job; the row copies,
        validation, and file write happen on the export thread. The kv-slot
        snapshots for every eligible request share one event wait.

        Coverage is every forwarded token: prompt rows (prefill capture) plus
        verify-committed decode rows. The final sampled token was never
        forwarded, so rows span ``[0, seqlen - 1)``.
        """
        prepared = []
        cleanup_rids = []
        for req in reqs:
            rid = req.rid
            if not self._accepting.is_set():
                self.stats.bump("shutdown_admission_miss_ct")
                self._settle_captured_rows(self.bookkeeper.pop(rid))
                self.sampler.forget(rid)
                continue
            if rid.startswith(HEALTH_CHECK_RID_PREFIX) or not self._sampled(rid):
                self._settle_captured_rows(self.bookkeeper.pop(rid))
                self.sampler.forget(rid)
                continue
            if getattr(
                self.sink, "prefix_enabled", False
            ) and not self._prefix_context_supported(req):
                self.stats.bump("prefix_context_unsupported_miss_ct")
                self.bookkeeper.mark_miss([rid])
                cleanup_rids.append(rid)
                continue

            prompt_len = len(req.origin_input_ids)
            seqlen = prompt_len + len(req.output_ids_through_stop)
            num_rows = max(prompt_len, seqlen - 1)
            tokens = list(req.origin_input_ids) + list(req.output_ids_through_stop)
            prepared.append((req, prompt_len, num_rows, tokens))

        for rid in cleanup_rids:
            self.sampler.forget(rid)
        if cleanup_rids:
            global_barrier = self.ring.last_enqueued_seq
            for rid in cleanup_rids:
                self._submit_cleanup_job(rid, global_barrier)
        if not prepared:
            return
        if not self._capture_allowed(
            rids=[req.rid for req, _, _, _ in prepared],
            rows=sum(num_rows for _, _, num_rows, _ in prepared),
            phase="finish",
        ):
            # Avoid the synchronous finish-hook slot snapshot under pressure.
            # Cleanup jobs still honor each request's finalize barrier before
            # popping generations, so late D2H rows cannot become orphans.
            global_barrier = self.ring.last_enqueued_seq
            for req, _prompt_len, _num_rows, _tokens in prepared:
                self._submit_cleanup_job(req.rid, global_barrier)
                self.sampler.forget(req.rid)
            return
        slot_ranges = [
            req_to_token_pool.req_to_token[req.req_pool_idx][:num_rows]
            for req, _, num_rows, _ in prepared
        ]
        try:
            snapshots = self._snapshot_kv_slots_batch(slot_ranges)
        except Exception:
            # Capture is best-effort. A snapshot failure must not prevent the
            # scheduler from releasing this batch's KV slots and completing
            # the serving requests. Keep the miss tombstones until the normal
            # orphan sweep so any already-enqueued rows cannot later be
            # mistaken for complete samples.
            rids = [req.rid for req, _, _, _ in prepared]
            logger.exception(
                "hidden finish slot snapshot failed; dropping %d samples",
                len(rids),
            )
            self.stats.bump("finish_snapshot_failed_miss_ct", len(rids))
            self.bookkeeper.mark_miss(rids)
            global_barrier = self.ring.last_enqueued_seq
            for rid in rids:
                self._submit_cleanup_job(rid, global_barrier)
                self.sampler.forget(rid)
            return
        global_barrier = self.ring.last_enqueued_seq
        for (req, prompt_len, num_rows, tokens), slots in zip(prepared, snapshots):
            rid = req.rid
            ring_seq_barrier, has_own_rows = self.bookkeeper.export_barrier(
                rid, global_barrier
            )
            if has_own_rows:
                self.stats.bump("per_request_barrier_ct")
                self.stats.bump(
                    "barrier_younger_seq_avoided_ct",
                    max(0, global_barrier - ring_seq_barrier),
                )
            else:
                self.stats.bump("fully_warm_global_barrier_ct")
            job = HiddenExportJob(
                rid=rid,
                sample_id=self._sample_id_for(rid),
                tokens=torch.tensor(tokens[:num_rows], dtype=torch.long),
                slots=slots,
                ring_seq_barrier=ring_seq_barrier,
                prompt_len=prompt_len,
            )
            self.export_worker.submit(job)
            self.sampler.forget(rid)

    @staticmethod
    def _prefix_context_supported(req: Req) -> bool:
        """V1 shares only fixed-target, pure-text, unsalted requests."""
        return (
            req.extra_key is None
            and req.lora_id is None
            and req.input_embeds is None
            and req.multimodal_inputs is None
            and req.positional_embed_overrides is None
        )

    def invalidate(self, rid: str) -> None:
        """Retract/abort: drop finalize records so a re-scheduled request's
        stale rows can't validate as its own."""
        self._settle_captured_rows(self.bookkeeper.invalidate(rid))
        self.sampler.forget(rid)


def get_global_hidden_capturer() -> Optional[HiddenStatesCapturer]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().hidden_capturer


def set_global_hidden_capturer(capturer: Optional[HiddenStatesCapturer]) -> None:
    from sglang.srt.runtime_context import get_resources

    get_resources().hidden_capturer = capturer
