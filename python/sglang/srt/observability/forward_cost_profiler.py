"""Per-forward cost samples for offline prefill cost-model fitting.

Each ModelRunner.forward call is one sample: the batch shape snapshotted at
scope entry plus the elapsed time between a start and an end CUDA event recorded
on the current stream at scope entry and exit. Harvesting is asynchronous; only
close(drain=True) synchronizes the device. The occupancy/FPM DeviceTimer and its
reporter payload are separate and unchanged.
"""

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, List, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device_timer import DeviceTimer

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class ForwardCostShape(msgspec.Struct, frozen=True):
    forward_mode: str
    batch_size: int
    # None outside extend batches, and on the gpu_only extend path.
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    # Per-DP-rank token counts from the DP gather; None without DP attention.
    global_num_tokens: Optional[List[int]]
    # First layer and layer count of a split prefill; (0, 1) otherwise.
    split_index: int
    split_forward_count: int
    # Scheduler admission snapshot; None when no queue state was captured.
    waiting_reqs_before: Optional[int]
    waiting_reqs_after: Optional[int]


class ForwardCostRecord(msgspec.Struct, frozen=True):
    # Host wall clock at scope entry; lets the fit drop warmup by time.
    launch_unix_s: float
    elapsed_s: float
    dp_rank: int
    # The forward raised: the interval covers partial work, not the shape.
    failed: bool
    shape: ForwardCostShape


class ForwardCostProfiler:
    def __init__(self, *, path: str, dp_rank: int):
        # "xb" refuses to append to an earlier run's samples.
        self._file = open(path, "xb")
        self._encoder = msgspec.json.Encoder()
        self._dp_rank = dp_rank
        self._timer = DeviceTimer(reporter=self._write)
        self._num_started = 0
        self._num_written = 0

    @classmethod
    def maybe_create(
        cls, *, is_draft_worker: bool, enable_elastic_ep: bool
    ) -> Optional["ForwardCostProfiler"]:
        out_dir = envs.SGLANG_FORWARD_COST_PROFILE_DIR.get()
        if out_dir is None or is_draft_worker:
            return None
        parallel = get_parallel()
        if parallel.pp_size > 1 or enable_elastic_ep:
            raise ValueError(
                "SGLANG_FORWARD_COST_PROFILE_DIR supports neither PP nor elastic EP"
            )
        # One writer per DP rank; attn TP/CP peers run the same forward.
        if parallel.attn_tp_rank != 0 or parallel.attn_cp_rank != 0:
            return None
        dp_rank = parallel.dp_rank if parallel.dp_rank is not None else 0
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"forward_cost.dp{dp_rank}.jsonl")
        return cls(path=path, dp_rank=dp_rank)

    @contextmanager
    def scope(self, *, forward_batch: "ForwardBatch", split_forward_count: int):
        shape = _snapshot_shape(
            forward_batch=forward_batch, split_forward_count=split_forward_count
        )
        self._num_started += 1
        metadata = {"launch_unix_s": time.time(), "failed": False, "shape": shape}
        with self._timer.wrap(metadata=metadata):
            try:
                yield
            except BaseException:
                # Set before wrap's exit, which may already write the sample.
                metadata["failed"] = True
                raise

    def poll(self) -> None:
        """Write samples whose end event has completed, without blocking."""
        self._timer._report()
        self._file.flush()

    def close(self, *, drain: bool) -> None:
        # Event queries can raise on a faulted device; the file still closes.
        try:
            if drain:
                torch.cuda.synchronize()
            self._timer._report()
        finally:
            self._file.close()
        lost = self._num_started - self._num_written
        if lost:
            logger.warning(
                "forward cost profiler closed with %d unwritten samples", lost
            )

    def _write(
        self, *, t: float, launch_unix_s: float, failed: bool, shape: ForwardCostShape
    ) -> None:
        record = ForwardCostRecord(
            launch_unix_s=launch_unix_s,
            elapsed_s=t,
            dp_rank=self._dp_rank,
            failed=failed,
            shape=shape,
        )
        self._file.write(self._encoder.encode(record) + b"\n")
        self._num_written += 1


def forward_cost_scope(
    profiler: Optional[ForwardCostProfiler],
    *,
    forward_batch: "ForwardBatch",
    split_forward_count: int,
):
    if profiler is None:
        return nullcontext()
    return profiler.scope(
        forward_batch=forward_batch, split_forward_count=split_forward_count
    )


def _snapshot_shape(
    *, forward_batch: "ForwardBatch", split_forward_count: int
) -> ForwardCostShape:
    # Copy the lists: the forward pads them in place for DP/graph runs, and
    # under overlap the next batch is prepared before this one completes.
    def copy(values):
        return None if values is None else [int(v) for v in values]

    return ForwardCostShape(
        forward_mode=forward_batch.forward_mode.name,
        batch_size=forward_batch.batch_size,
        extend_seq_lens=copy(forward_batch.extend_seq_lens_cpu),
        extend_prefix_lens=copy(forward_batch.extend_prefix_lens_cpu),
        global_num_tokens=copy(forward_batch.global_num_tokens_cpu),
        split_index=forward_batch.split_index,
        split_forward_count=split_forward_count,
        waiting_reqs_before=forward_batch.waiting_reqs_before,
        waiting_reqs_after=forward_batch.waiting_reqs_after,
    )
