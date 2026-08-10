"""Online capture of target aux/last hidden states for draft-model training.

``HiddenStatesCapturer`` is the orchestration seam between serving and the
host-side capture pipeline (see ``hidden_host.py`` / ``hidden_sink.py``):

- ``on_forward_end``   (model runner, forward thread): grab GPU references to
  the packed aux hidden states ``[T, K*H]`` and the post-final-norm last
  hidden states ``[T, H]`` plus row->request attribution. No copies.
- ``HiddenCaptureOutput.stage`` (scheduler, copy stream): async D2H into the
  pinned staging ring, mirroring the result-copy path's timing.
- ``collect_at_finish`` (scheduler, before ``release_kv_cache``): sampling
  decision, kv-slot snapshot, export job enqueue. Never blocks.

M1 scope is prefill rows only (cold/chunked/warm-prefix); verify committed
rows, Mooncake sink, and DP-attention multi-writer are M2+ (the fail-closed
matrix in ``create`` reflects this).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.hidden_host import (
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


class HiddenStatesCapturer:
    """Owns the staging ring, host sidecar, and worker threads for capture."""

    @staticmethod
    def create(
        *,
        server_args: ServerArgs,
        model_config: ModelConfig,
        spec_aux_config: SpecAuxHiddenStateConfig,
        num_tokens: int,
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
            (
                server_args.cuda_graph_config.prefill.backend
                != CudaGraphBackend.DISABLED,
                "prefill CUDA graph replays overwrite static output buffers; "
                "every graph-run prefill would be a capture miss (zero yield). "
                "Launch with --disable-prefill-cuda-graph "
                "(or --cuda-graph-backend-prefill=disabled)",
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
        self.finalize_worker = HiddenFinalizeWorker(
            ring=self.ring,
            sidecar=self.sidecar,
            bookkeeper=self.bookkeeper,
            stats=self.stats,
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
                # M1 rows cover the prompt region only (no verify/decode rows),
                # and loss_mask is a placeholder — recompute offline before
                # feeding a training recipe that masks on role boundaries.
                "coverage": "prefill_only",
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
            # CUDA-graph replays overwrite static output buffers; staging from
            # them races the next replay. M1 fails closed (see plan: verify /
            # graph capture needs an explicit fence or device staging ring).
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
        """
        rid = req.rid
        if rid.startswith(HEALTH_CHECK_RID_PREFIX):
            self.bookkeeper.pop(rid)
            return
        if not self._sampled(rid):
            self.bookkeeper.pop(rid)
            return

        prompt_len = len(req.origin_input_ids)
        slots = (
            req_to_token_pool.req_to_token[req.req_pool_idx][:prompt_len]
            .cpu()
            .clone()
            .to(torch.long)
        )
        job = HiddenExportJob(
            rid=rid,
            sample_id=self._sample_id_for(rid),
            tokens=torch.tensor(req.origin_input_ids, dtype=torch.long),
            slots=slots,
            ring_seq_barrier=self.ring.last_enqueued_seq,
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
