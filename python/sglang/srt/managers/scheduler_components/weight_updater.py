from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


def _process_rss_bytes() -> int:
    with open("/proc/self/statm") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


@dataclass(slots=True)
class _StagedTensorUpdate:
    future: Optional[concurrent.futures.Future]
    request: UpdateWeightsFromTensorReqInput
    scheduler_rss_before_bytes: int


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    tensor_stage_executor: concurrent.futures.ThreadPoolExecutor = field(
        default_factory=lambda: concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sglang-weight-stage"
        ),
        repr=False,
    )
    staged_tensor_updates: Dict[str, _StagedTensorUpdate] = field(
        default_factory=dict, repr=False
    )

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            if recv_req.draft_only:
                if self.draft_worker is None:
                    success = False
                    message = (
                        "draft_only weight update requires a speculative draft model."
                    )
                else:
                    success, message = self.draft_worker.update_weights_from_disk(
                        recv_req
                    )
            else:
                success, message = self.tp_worker.update_weights_from_disk(recv_req)
                tp_success = success
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_disk(
                        recv_req
                    )
                if tp_success:
                    self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def _select_tensor_update_worker(
        self, recv_req: UpdateWeightsFromTensorReqInput
    ) -> Tuple[Optional[Any], Optional[str]]:
        if recv_req.draft_only and recv_req.disable_draft_model:
            return None, "draft_only and disable_draft_model are mutually exclusive."
        if recv_req.draft_only and self.draft_worker is None:
            return None, "draft_only weight update requires a speculative draft model."
        if recv_req.disable_draft_model:
            return self.tp_worker, None
        return self.draft_worker or self.tp_worker, None

    @staticmethod
    def _prepare_tensor_update_in_background(worker, recv_req):
        started = time.perf_counter()
        staged = worker.prepare_weights_from_tensor(recv_req)
        staged.phase_timings_ms["background_stage_wall_ms"] = (
            time.perf_counter() - started
        ) * 1000
        return staged

    def _handle_staged_tensor_update(
        self,
        worker,
        recv_req: UpdateWeightsFromTensorReqInput,
    ) -> UpdateWeightsFromTensorReqOutput:
        update_id = recv_req.update_id
        if not recv_req.draft_only:
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message="staged tensor updates currently require draft_only=true.",
                update_id=update_id,
            )
        if not update_id:
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message=f"operation={recv_req.operation!r} requires update_id.",
            )

        if recv_req.operation == "stage":
            if update_id in self.staged_tensor_updates:
                return UpdateWeightsFromTensorReqOutput(
                    success=False,
                    message=f"staged tensor update {update_id!r} already exists.",
                    update_id=update_id,
                )
            rss_before = _process_rss_bytes()
            dispatch_started = time.perf_counter()
            future = self.tensor_stage_executor.submit(
                self._prepare_tensor_update_in_background, worker, recv_req
            )
            self.staged_tensor_updates[update_id] = _StagedTensorUpdate(
                future=future,
                request=recv_req,
                scheduler_rss_before_bytes=rss_before,
            )
            return UpdateWeightsFromTensorReqOutput(
                success=True,
                message="CPU tensor staging started.",
                update_id=update_id,
                staging_state="pending",
                phase_timings_ms={
                    "stage_dispatch_ms": (time.perf_counter() - dispatch_started) * 1000
                },
                host_memory_bytes={"scheduler_rss_before_stage_bytes": rss_before},
            )

        record = self.staged_tensor_updates.get(update_id)
        if record is None or record.future is None:
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message=f"unknown staged tensor update {update_id!r}.",
                update_id=update_id,
            )

        if recv_req.operation == "discard":
            record.future.cancel()
            self.staged_tensor_updates.pop(update_id, None)
            record.future = None
            return UpdateWeightsFromTensorReqOutput(
                success=True,
                message="CPU staged tensor update discarded.",
                update_id=update_id,
                host_memory_bytes={
                    "scheduler_rss_after_release_bytes": _process_rss_bytes()
                },
            )

        if not record.future.done():
            return UpdateWeightsFromTensorReqOutput(
                success=True,
                message="CPU tensor staging is still running.",
                update_id=update_id,
                staging_state="pending",
                host_memory_bytes={
                    "scheduler_rss_before_stage_bytes": (
                        record.scheduler_rss_before_bytes
                    ),
                    "scheduler_rss_current_bytes": _process_rss_bytes(),
                },
            )

        stage_error = record.future.exception()
        if stage_error is not None:
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message=f"CPU tensor staging failed: {stage_error}",
                update_id=update_id,
                staging_state="failed",
                host_memory_bytes={
                    "scheduler_rss_before_stage_bytes": (
                        record.scheduler_rss_before_bytes
                    ),
                    "scheduler_rss_current_bytes": _process_rss_bytes(),
                },
            )

        staged = record.future.result()
        host_memory = dict(staged.host_memory_bytes)
        host_memory.update(
            {
                "scheduler_rss_before_stage_bytes": record.scheduler_rss_before_bytes,
                "scheduler_rss_after_stage_bytes": _process_rss_bytes(),
            }
        )
        if recv_req.operation == "status":
            return UpdateWeightsFromTensorReqOutput(
                success=True,
                message="CPU tensor staging is ready.",
                update_id=update_id,
                staging_state="ready",
                phase_timings_ms=dict(staged.phase_timings_ms),
                host_memory_bytes=host_memory,
            )
        if recv_req.operation != "commit":
            return UpdateWeightsFromTensorReqOutput(
                success=False,
                message=f"unsupported staged tensor operation {recv_req.operation!r}.",
                update_id=update_id,
            )

        phase_timings: Dict[str, float] = {}
        update_status: Dict[str, Any] = {}
        try:
            success, message = worker.apply_prepared_weights_from_tensor(
                staged,
                record.request,
                phase_timings_ms=phase_timings,
                update_status=update_status,
            )
        except BaseException as error:
            success = False
            message = f"Staged tensor commit failed: {error}"
            update_status["partial_update"] = True

        barrier_started = time.perf_counter()
        torch.distributed.barrier(group=self.tp_cpu_group)
        phase_timings["rank_barrier_ms"] = (
            time.perf_counter() - barrier_started
        ) * 1000
        self.staged_tensor_updates.pop(update_id, None)
        record.future = None
        del staged
        host_memory["scheduler_rss_after_release_bytes"] = _process_rss_bytes()
        if not success:
            logger.error(message)
        return UpdateWeightsFromTensorReqOutput(
            success=success,
            message=message,
            update_id=update_id,
            partial_update=bool(update_status.get("partial_update", False)),
            phase_timings_ms=phase_timings,
            host_memory_bytes=host_memory,
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update model tensors or manage an additive CPU-staged transaction."""
        with self._observe_weight_load("tensor"):
            worker, routing_error = self._select_tensor_update_worker(recv_req)
            if routing_error is not None:
                barrier_started = time.perf_counter()
                torch.distributed.barrier(group=self.tp_cpu_group)
                return UpdateWeightsFromTensorReqOutput(
                    success=False,
                    message=routing_error,
                    update_id=recv_req.update_id,
                    phase_timings_ms={
                        "rank_barrier_ms": (time.perf_counter() - barrier_started)
                        * 1000
                    },
                )
            assert worker is not None

            if recv_req.operation != "apply":
                return self._handle_staged_tensor_update(worker, recv_req)

            phase_timings: Dict[str, float] = {}
            update_status: Dict[str, Any] = {}
            try:
                success, message = worker.update_weights_from_tensor(
                    recv_req,
                    phase_timings_ms=phase_timings,
                    update_status=update_status,
                )
            except TypeError as error:
                # Preserve non-draft legacy workers that have not opted into the
                # additive timing kwargs.
                if recv_req.draft_only:
                    raise
                success, message = worker.update_weights_from_tensor(recv_req)
            except BaseException as error:
                success = False
                message = f"Tensor update raised unexpectedly: {error}"
                update_status["partial_update"] = True
            if success and not recv_req.draft_only:
                self.flush_cache_after_weight_update(recv_req)
            elif not success:
                logger.error(message)
            barrier_started = time.perf_counter()
            torch.distributed.barrier(group=self.tp_cpu_group)
            phase_timings["rank_barrier_ms"] = (
                time.perf_counter() - barrier_started
            ) * 1000
            return UpdateWeightsFromTensorReqOutput(
                success=success,
                message=message,
                update_id=recv_req.update_id,
                partial_update=bool(update_status.get("partial_update", False)),
                phase_timings_ms=phase_timings,
            )

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _assert_weight_cache_inactive(self, op: str) -> None:
        """Reject freeing/restoring model weights while the CUDA IPC weight
        cache is active: the weights are shared with the daemon via CUDA IPC, so
        freeing them would leave the daemon and every peer pointing at released
        memory.
        """
        mode = self.tp_worker.model_runner.server_args.weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("release_memory_occupation")
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self._assert_weight_cache_inactive("resume_memory_occupation")
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
