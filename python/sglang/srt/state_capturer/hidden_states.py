"""Online capture of target aux/last hidden states for draft-model training.

``HiddenStatesCapturer`` is the orchestration seam between serving and the
host-side capture pipeline (see ``hidden_host.py`` / ``hidden_sink.py``):

- ``on_forward_end``   (model runner, forward thread): grab GPU references to
  the packed aux hidden states ``[T, K*H]`` and the post-final-norm last
  hidden states ``[T, H]`` plus row->request attribution. No copies.
- ``capture_verify_window`` (DSpark worker, forward stream, post-acceptance):
  pack one verify step's strided window into a capture-owned device twin.
  Same-stream ordering is the CUDA-graph overwrite fence (invariant #6 of the
  capture plan): the next replay queues behind the pack.
- ``HiddenCaptureOutput.stage`` / ``HiddenVerifyCaptureOutput.stage``
  (scheduler, copy stream): async D2H into the pinned staging ring on a
  dedicated capture stream.
- ``collect_at_finish`` (scheduler, before ``release_kv_cache``): sampling
  decision, kv-slot snapshot over prompt + committed decode rows, export job
  enqueue.

Coverage is every forwarded token: prompt rows (prefill capture) plus
verify-committed decode rows (the final sampled token has no hidden row).
Mooncake sink and DP-attention multi-writer status: see the fail-closed
matrix in ``create``.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

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
    HiddenCaptureStats,
    HiddenFinalizeWorker,
    HiddenHostSidecar,
    HiddenStagingRing,
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
# pinned via env vars: slots = budget / verify_window_tokens (min 2). At
# 4096 total slots and Qwen3-8B widths (K=5 aux + last, bf16) this is the
# same ~200MB the old fixed 2x2048 default used.
_VERIFY_TWIN_BUDGET_TOKENS = 4096


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
        device: str,
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

        aux_layer_ids = _resolve_aux_layer_ids(spec_aux_config)
        # M1 support matrix; anything outside fails closed (no partial data).
        gates = [
            (
                not aux_layer_ids,
                "no aux hidden state layers configured "
                "(requires a DFlash/DSpark or EAGLE3 draft with aux capture)",
            ),
            (
                server_args.enable_dp_attention,
                "DP attention captures per-rank shards (M2+)",
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

        # Host memory gate: the sidecar scales with the KV pool. Estimate
        # before allocating so an oversized config disables capture instead
        # of silently committing tens of GB per scheduler (x N under DP).
        hidden_size = model_config.hf_text_config.hidden_size
        row_bytes = (len(aux_layer_ids) + 1) * hidden_size * model_config.dtype.itemsize
        sidecar_gb = num_tokens * row_bytes / 1024**3
        max_gb = envs.SGLANG_HIDDEN_CAPTURE_MAX_HOST_GB.get()
        if sidecar_gb > max_gb:
            _disabled(
                f"host sidecar would need {sidecar_gb:.1f} GB "
                f"(max_total_num_tokens={num_tokens} x {row_bytes} B/row) > "
                f"SGLANG_HIDDEN_CAPTURE_MAX_HOST_GB={max_gb}. Raise the limit "
                "or reduce the KV pool"
            )
            return None

        return HiddenStatesCapturer(
            model_config=model_config,
            num_aux_layers=len(aux_layer_ids),
            num_tokens=num_tokens,
            verify_window_tokens=_resolve_verify_window_tokens(
                server_args=server_args, max_running_requests=max_running_requests
            ),
            sink_kind=sink_kind,
            sink_dir=sink_dir,
            aux_layer_ids=list(aux_layer_ids),
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        num_aux_layers: int,
        num_tokens: int,
        verify_window_tokens: int,
        sink_kind: str,
        sink_dir: Optional[str],
        aux_layer_ids: List[int],
        model_path: str,
        model_revision: Optional[str],
    ) -> None:
        hidden_size = model_config.hf_text_config.hidden_size
        self.aux_width = num_aux_layers * hidden_size
        self.last_width = hidden_size
        self.dtype = model_config.dtype
        self.sample_rate = envs.SGLANG_HIDDEN_CAPTURE_SAMPLE_RATE.get()

        self.stats = HiddenCaptureStats()
        self.bookkeeper = HiddenCaptureBookkeeper()
        # Dedicated stream for capture D2H. The scheduler's copy stream is
        # FIFO: queueing capture's large copies there would make the NEXT
        # step's (tiny) result copies — and thus its copy_done — wait behind
        # them, leaking capture cost into serving tail latency.
        self.capture_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.ring = HiddenStagingRing(
            num_slots=envs.SGLANG_HIDDEN_CAPTURE_STAGING_SLOTS.get(),
            slot_tokens=envs.SGLANG_HIDDEN_CAPTURE_STAGING_SLOT_TOKENS.get(),
            aux_width=self.aux_width,
            last_width=self.last_width,
            dtype=self.dtype,
        )
        self.sidecar = HiddenHostSidecar(
            num_slots=num_tokens,
            aux_width=self.aux_width,
            last_width=self.last_width,
            dtype=self.dtype,
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
            )
            if torch.cuda.is_available()
            else None
        )
        self.finalize_worker = HiddenFinalizeWorker(
            ring=self.ring,
            sidecar=self.sidecar,
            bookkeeper=self.bookkeeper,
            stats=self.stats,
            twin_pool=self.twin_pool,
        )
        self.sink = self._build_sink(sink_kind, sink_dir)
        self.sink.write_fingerprint(
            {
                "model_path": model_path,
                "model_revision": model_revision,
                "dtype": str(self.dtype),
                "aux_layer_ids": aux_layer_ids,
                "hidden_size": hidden_size,
                "num_aux_layers": num_aux_layers,
                "norm_contract": "post_final_norm_pre_lm_head",
                "aux_layout": "packed_last_dim",  # [T, K*H], serving layout
                # Rows cover the prompt region plus verify-committed decode
                # tokens (the final sampled token has no hidden row).
                # loss_mask is a placeholder — recompute offline before
                # feeding a training recipe that masks on role boundaries.
                "coverage": "prefill_and_verify_commit",
                "loss_mask": "all_ones_placeholder",
            }
        )
        self.export_worker = HiddenExportWorker(
            sidecar=self.sidecar,
            bookkeeper=self.bookkeeper,
            finalize_worker=self.finalize_worker,
            stats=self.stats,
            sink=self.sink,
            queue_size=envs.SGLANG_HIDDEN_CAPTURE_EXPORT_QUEUE_SIZE.get(),
        )
        self.finalize_worker.start()
        self.export_worker.start()
        logger.info(
            "hidden state capture enabled: sink=%s (dir=%s), aux_layer_ids=%s, "
            "sample_rate=%.3f",
            sink_kind,
            sink_dir,
            aux_layer_ids,
            self.sample_rate,
        )

    def _build_sink(self, sink_kind: str, sink_dir: Optional[str]):
        if sink_kind == "mooncake":
            from sglang.srt.state_capturer.hidden_mooncake import MooncakeHiddenSink

            # +8 bytes/row: input_ids ride the same registered staging buffer.
            row_bytes = (self.aux_width + self.last_width) * self.dtype.itemsize + 8
            return MooncakeHiddenSink(
                store_id=envs.SGLANG_HIDDEN_CAPTURE_STORE_ID.get(),
                row_bytes=row_bytes,
                max_export_tokens=envs.SGLANG_HIDDEN_CAPTURE_MAX_EXPORT_TOKENS.get(),
            )
        return HiddenFileSink(sink_dir)

    # ---------------------------------------------------------------- forward

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
            last = last.clone()

        num_rows = aux.shape[0]
        if sum(extend_lens) != num_rows or last.shape[0] != num_rows:
            # Row attribution would be wrong; drop the whole forward.
            self.stats.bump("attach_mismatch_miss_ct")
            self.bookkeeper.mark_miss(rids)
            return None

        req_ranges = []
        row = 0
        for rid, extend_len in zip(rids, extend_lens):
            req_ranges.append((rid, row, row + extend_len))
            row += extend_len
        return HiddenCaptureOutput(
            aux_hidden_states=aux,
            last_hidden_states=last,
            out_cache_loc=forward_batch.out_cache_loc,
            input_tokens=forward_batch.input_ids,
            req_ranges=req_ranges,
            capturer=self,
        )

    # ------------------------------------------------------------------ verify

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
        behind the pack — the plan's invariant #6 without stalling forward on
        any D2H. The twin is then D2H'd off-stream by stage().

        The post-norm last hidden arrives strided on the dense verify path or
        compact (ragged) on the compact path; the compact form is scattered
        straight into the twin with the same kernel the epilogue uses for aux.
        """
        if self.twin_pool is None:
            return None
        num_rows = bs * stride
        if num_rows > self.twin_pool.twin_tokens or bs > self.twin_pool.max_reqs:
            self.stats.bump("verify_oversize_miss_ct")
            self.bookkeeper.mark_miss(rids)
            return None
        if last_strided is None and last_compact is None:
            # Post-norm hidden unavailable (e.g. a graph captured before the
            # capturer was installed): fail closed with attribution.
            self.stats.bump("skipped_forward_ct")
            self.bookkeeper.mark_miss(rids)
            return None
        # Twins whose D2H already finished are recyclable right now — reap
        # them on this thread instead of waiting for the finalize thread to
        # chew through the queue (its sidecar memcpy holds twins for tens of
        # ms and starves fast decode steps).
        twin = self.twin_pool.try_acquire()
        if twin is None:
            self.ring.reap_ready_twins(self.twin_pool)
            twin = self.twin_pool.try_acquire()
        if twin is None:
            self.stats.bump("verify_twin_full_miss_ct")
            self.bookkeeper.mark_miss(rids)
            return None

        # Pack on the caller's (forward) stream: this is the fence.
        twin.aux[:num_rows].copy_(aux_strided[:num_rows], non_blocking=True)
        if last_strided is not None:
            twin.last[:num_rows].copy_(last_strided[:num_rows], non_blocking=True)
        else:
            from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
                scatter_compact_to_strided_into,
            )

            scatter_compact_to_strided_into(
                compact=last_compact.contiguous(),
                verify_lens=verify_lens,
                out=twin.last[:num_rows],
                stride=stride,
                fill_value=0.0,
            )
        twin.cache_loc[:num_rows].copy_(
            verify_cache_loc[:num_rows].to(torch.int64), non_blocking=True
        )
        twin.tokens[:num_rows].copy_(
            verify_tokens[:num_rows].to(torch.int64), non_blocking=True
        )
        twin.commit_lens[:bs].copy_(commit_lens.to(torch.int32), non_blocking=True)
        twin.fence_event.record()

        return HiddenVerifyCaptureOutput(
            twin=twin, rids=list(rids), stride=stride, num_reqs=bs, capturer=self
        )

    def stage_verify(self, output: HiddenVerifyCaptureOutput) -> None:
        """D2H a packed verify twin into the pinned ring (copy-stream context)."""
        slots = self.ring.try_acquire(1)
        if slots is None:
            self.stats.bump("stage_full_miss_ct")
            self.bookkeeper.mark_miss(output.rids)
            self.twin_pool.release(output.twin)
            return
        (slot,) = slots
        if output.num_reqs * output.stride > self.ring.slot_tokens:
            self.stats.bump("verify_oversize_miss_ct")
            self.bookkeeper.mark_miss(output.rids)
            self.twin_pool.release(output.twin)
            self.ring.release(slot)
            return

        if self.capture_stream is not None:
            self.capture_stream.wait_event(output.twin.fence_event)
            stream_ctx = torch.cuda.stream(self.capture_stream)
        else:
            stream_ctx = contextlib.nullcontext()
        with stream_ctx:
            self.ring.enqueue_verify_segment(
                slot,
                twin=output.twin,
                rids=output.rids,
                stride=output.stride,
                num_reqs=output.num_reqs,
            )
        self.stats.bump("rows_staged_ct", output.num_reqs * output.stride)

    # ---------------------------------------------------------------- staging

    def stage(self, output: HiddenCaptureOutput) -> None:
        """Enqueue one forward's rows into the ring (copy-stream context).

        All-or-nothing per forward: if the ring can't hold every row, every
        affected request becomes a capture-miss (never backpressure).

        The large copies are issued on the capturer's dedicated stream, which
        waits on the caller's stream (the result-copy stream, itself already
        past forward). The scheduler's copy stream stays free for the next
        step's small result copies; ring events gate the finalize thread.
        """
        num_rows = output.aux_hidden_states.shape[0]
        if num_rows == 0:
            return
        slot_tokens = self.ring.slot_tokens
        num_segments = (num_rows + slot_tokens - 1) // slot_tokens
        slots = self.ring.try_acquire(num_segments)
        if slots is None:
            self.stats.bump("stage_full_miss_ct")
            self.bookkeeper.mark_miss([rid for rid, _, _ in output.req_ranges])
            return

        if self.capture_stream is not None:
            self.capture_stream.wait_stream(torch.cuda.current_stream())
            stream_ctx = torch.cuda.stream(self.capture_stream)
        else:
            stream_ctx = contextlib.nullcontext()
        with stream_ctx:
            for seg, slot in enumerate(slots):
                seg_start = seg * slot_tokens
                seg_end = min(seg_start + slot_tokens, num_rows)
                seg_ranges = [
                    (rid, max(r0, seg_start) - seg_start, min(r1, seg_end) - seg_start)
                    for rid, r0, r1 in output.req_ranges
                    if r0 < seg_end and r1 > seg_start
                ]
                self.ring.enqueue_segment(
                    slot,
                    aux_rows=output.aux_hidden_states[seg_start:seg_end],
                    last_rows=output.last_hidden_states[seg_start:seg_end],
                    cache_locs=output.out_cache_loc[seg_start:seg_end],
                    tokens=output.input_tokens[seg_start:seg_end],
                    req_ranges=seg_ranges,
                )
        self.stats.bump("rows_staged_ct", num_rows)

    # ----------------------------------------------------------------- finish

    def _sampled(self, rid: str) -> bool:
        if self.sample_rate >= 1.0:
            return True
        digest = hashlib.md5(rid.encode()).digest()
        return int.from_bytes(digest[:8], "little") / 2**64 < self.sample_rate

    @staticmethod
    def _sample_id_for(rid: str) -> str:
        """Filesystem/key-safe sample id. rid is caller-supplied (io_struct
        only generates a uuid when absent), so it can carry path separators or
        collide across requests; hash unless it's already a safe unique token.
        The original rid is preserved inside the exported record."""
        if _SAFE_RID_RE.fullmatch(rid):
            return rid
        return hashlib.sha1(rid.encode()).hexdigest()

    def collect_at_finish(self, req: Req, req_to_token_pool: ReqToTokenPool) -> None:
        """Finish hook (scheduler thread, before ``release_kv_cache``).

        Snapshots CPU-side state and enqueues the export job; the row copies,
        validation, and file write happen on the export thread. The kv-slot
        snapshot is one small synchronous D2H (same pattern as the
        routed-experts finish hook).

        Coverage is every forwarded token: prompt rows (prefill capture) plus
        verify-committed decode rows. The final sampled token was never
        forwarded, so rows span ``[0, seqlen - 1)``.
        """
        rid = req.rid
        if rid.startswith(HEALTH_CHECK_RID_PREFIX):
            self.bookkeeper.pop(rid)
            return
        if not self._sampled(rid):
            self.bookkeeper.pop(rid)
            return

        prompt_len = len(req.origin_input_ids)
        seqlen = prompt_len + len(req.output_ids_through_stop)
        num_rows = max(prompt_len, seqlen - 1)
        slots = (
            req_to_token_pool.req_to_token[req.req_pool_idx][:num_rows]
            .cpu()
            .clone()
            .to(torch.long)
        )
        tokens = list(req.origin_input_ids) + list(req.output_ids_through_stop)
        job = HiddenExportJob(
            rid=rid,
            sample_id=self._sample_id_for(rid),
            tokens=torch.tensor(tokens[:num_rows], dtype=torch.long),
            slots=slots,
            ring_seq_barrier=self.ring.last_enqueued_seq,
            prompt_len=prompt_len,
        )
        self.export_worker.submit(job)

    def invalidate(self, rid: str) -> None:
        """Retract/abort: drop finalize records so a re-scheduled request's
        stale rows can't validate as its own."""
        self.bookkeeper.invalidate(rid)


def get_global_hidden_capturer() -> Optional[HiddenStatesCapturer]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().hidden_capturer


def set_global_hidden_capturer(capturer: Optional[HiddenStatesCapturer]) -> None:
    from sglang.srt.runtime_context import get_resources

    get_resources().hidden_capturer = capturer
