from __future__ import annotations

import gc
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    MultiprocessingSerializer,
    dynamic_import,
    get_available_gpu_memory,
    init_custom_process_group,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedWeightUpdate:
    """Validated host tensors ready for one in-place model update."""

    named_tensors: List[Tuple[str, torch.Tensor]]
    tensors_are_pre_sharded: bool
    phase_timings_ms: Dict[str, float]
    host_memory_bytes: Dict[str, int]
    fault_injection_after_tensors: Optional[int] = None


@contextmanager
def _presharded_weight_loader_mode(model: Any, enabled: bool) -> Iterator[None]:
    """Temporarily tell TP linear loaders that input tensors are rank-local."""

    if not enabled:
        yield
        return

    changed = []
    for module in model.modules():
        if hasattr(module, "use_presharded_weights"):
            changed.append((module, module.use_presharded_weights))
            module.use_presharded_weights = True
    try:
        yield
    finally:
        for module, previous in changed:
            module.use_presharded_weights = previous


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _unsupported_derived_weight_cache_error() -> Optional[str]:
    """Reject online weight updates that derived-weight caches cannot survive.

    The HPC-Ops bf16xfp32 GEMM caches the fp32 weight split; in-place loader
    writes are invisible to it, so an update would silently keep serving the
    old weights. The check is startup-determined and rank-uniform, so an
    update never proceeds on some workers while rejected on others.
    """
    from sglang.kernels.ops.attention.dsv4.gemm import hpc_bf16xfp32_gemm_enabled

    if hpc_bf16xfp32_gemm_enabled():
        return (
            "Online weight updates are not supported while the HPC-Ops "
            "bf16xfp32 GEMM optimization is enabled: the cached weight "
            "split would keep serving the old weights."
        )
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class WeightUpdater:
    tp_rank: int
    device: str
    gpu_id: int
    model_config: ModelConfig
    custom_weight_loaders: dict
    get_model: Callable[[], Any]
    update_model_fields: Callable[..., None]
    recapture_cuda_graph: Callable[[], None]
    get_model_runner: Callable[[], ModelRunner]
    _model_update_group: dict = field(default_factory=dict)

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def _assert_weight_cache_inactive(self: WeightUpdater, op: str) -> None:
        """Reject weight mutations while the CUDA IPC weight cache is active:
        param.data is the daemon's master copy shared with every co-attached
        engine, so an in-place update would silently corrupt them all.
        """
        mode = self.get_model_runner().server_args.weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} is not supported while the weight cache is "
                f"active (--weight-cache-mode {mode}): model weights are shared "
                f"with the daemon via CUDA IPC, so mutating them in place would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def update_weights_from_disk(
        self: WeightUpdater,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        self._assert_weight_cache_inactive("update_weights_from_disk")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.get_model())
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.get_model(), iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                model_load_weights(self.get_model(), iter)
                return False, message

        self.update_model_fields(
            model,
            model_path=model_path,
            load_format=load_format,
            load_config=load_config,
        )

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.recapture_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def update_weights_from_distributed(
        self: WeightUpdater,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        self._assert_weight_cache_inactive("update_weights_from_distributed")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.get_model().load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self: WeightUpdater, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (
                        name,
                        torch.empty(shape, dtype=target_dtype, device=self.device),
                    )
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.get_model().load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def prepare_weights_from_tensor(
        self: WeightUpdater,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
        *,
        stream_tensors: bool = False,
        tensors_are_pre_sharded: bool = False,
        pin_memory: bool = False,
        fault_injection_after_tensors: Optional[int] = None,
        phase_timings_ms: Optional[Dict[str, float]] = None,
    ) -> PreparedWeightUpdate:
        """Validate and optionally pin a tensor update without touching the GPU model."""

        phases = phase_timings_ms if phase_timings_ms is not None else {}
        prepare_started = time.perf_counter()
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            raise RuntimeError(error)

        monkey_patch_torch_reductions()
        self._assert_weight_cache_inactive("update_weights_from_tensor")
        if load_format == "flattened_bucket":
            raise ValueError("flattened_bucket does not support staged tensor updates")
        if fault_injection_after_tensors is not None:
            if os.getenv("SGLANG_ENABLE_WEIGHT_UPDATE_FAULT_INJECTION") != "1":
                raise ValueError(
                    "fault_injection_after_tensors requires "
                    "SGLANG_ENABLE_WEIGHT_UPDATE_FAULT_INJECTION=1"
                )
            if fault_injection_after_tensors < 1:
                raise ValueError("fault_injection_after_tensors must be positive")

        validation_started = time.perf_counter()
        normalized: List[Tuple[str, torch.Tensor]] = []
        seen_names = set()
        for index, item in enumerate(named_tensors):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(f"named_tensors[{index}] must be a (name, tensor) pair")
            name, tensor = item
            if not isinstance(name, str) or not name:
                raise TypeError(f"named_tensors[{index}] has an invalid name")
            if name in seen_names:
                raise ValueError(f"duplicate tensor name {name!r}")
            seen_names.add(name)
            if isinstance(tensor, LocalSerializedTensor):
                tensor = tensor.get(self.tp_rank)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"weight {name!r} is not a torch.Tensor")
            if tensor.layout != torch.strided:
                raise TypeError(
                    f"weight {name!r} must use torch.strided layout, got {tensor.layout}"
                )
            normalized.append((name, tensor))
        if not normalized:
            raise ValueError("named_tensors must not be empty")

        model = self.get_model()
        stream_tensors = stream_tensors and load_format in (None, "direct")
        validator = getattr(model, "validate_weights", None)
        if validator is not None and load_format is None:
            with _presharded_weight_loader_mode(model, tensors_are_pre_sharded):
                validator(
                    normalized,
                    tensors_are_pre_sharded=tensors_are_pre_sharded,
                )
        phases["input_and_model_validation_ms"] = (
            time.perf_counter() - validation_started
        ) * 1000

        weight_name_filter = (
            getattr(model, "should_materialize_weight", None)
            if stream_tensors and load_format is None
            else None
        )
        materialized = [
            (name, tensor)
            for name, tensor in normalized
            if weight_name_filter is None or weight_name_filter(name)
        ]
        source_bytes = sum(_tensor_bytes(tensor) for _, tensor in materialized)

        pin_started = time.perf_counter()
        staged = materialized
        if pin_memory:
            pinned = []
            for name, tensor in materialized:
                if tensor.device.type != "cpu":
                    raise ValueError(
                        f"pinned staging requires CPU tensors, got {tensor.device} "
                        f"for {name!r}"
                    )
                if tensor.is_pinned() and tensor.is_contiguous():
                    staged_tensor = tensor
                else:
                    staged_tensor = tensor.contiguous().pin_memory()
                pinned.append((name, staged_tensor))
            staged = pinned
        phases["host_pin_ms"] = (time.perf_counter() - pin_started) * 1000
        phases["prepare_total_ms"] = (time.perf_counter() - prepare_started) * 1000
        staged_bytes = sum(_tensor_bytes(tensor) for _, tensor in staged)
        return PreparedWeightUpdate(
            named_tensors=staged,
            tensors_are_pre_sharded=tensors_are_pre_sharded,
            phase_timings_ms=phases,
            host_memory_bytes={
                "source_tensor_bytes": source_bytes,
                "staged_tensor_bytes": staged_bytes,
                "pinned_tensor_bytes": staged_bytes if pin_memory else 0,
            },
            fault_injection_after_tensors=fault_injection_after_tensors,
        )

    def apply_prepared_weights_from_tensor(
        self: WeightUpdater,
        prepared: PreparedWeightUpdate,
        load_format: Optional[str] = None,
        *,
        stream_tensors: bool = False,
        collect_phase_timings: bool = False,
        update_status: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Apply one validated image in place and return a completion fence."""

        phases = prepared.phase_timings_ms
        status = update_status if update_status is not None else {}
        apply_started = time.perf_counter()
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()
        stream_tensors = stream_tensors and load_format in (None, "direct")
        model = self.get_model()
        source_named_tensors = prepared.named_tensors
        cuda_events = []
        mutation_started = False
        applied_tensor_count = 0
        failure: Optional[BaseException] = None

        def new_event():
            return device_module.Event(enable_timing=True)

        def materialize_tensors():
            nonlocal mutation_started, applied_tensor_count
            for name, tensor in source_named_tensors:
                h2d_start = h2d_end = copy_start = copy_end = None
                if collect_phase_timings and self.device != "cpu":
                    h2d_start, h2d_end = new_event(), new_event()
                    h2d_start.record()
                device_tensor = _unwrap_tensor(
                    tensor,
                    tp_rank=self.tp_rank,
                    device=infered_device,
                    non_blocking=bool(
                        tensor.device.type == "cpu" and tensor.is_pinned()
                    ),
                )
                if collect_phase_timings and self.device != "cpu":
                    h2d_end.record()
                    copy_start, copy_end = new_event(), new_event()
                    copy_start.record()
                mutation_started = True
                try:
                    yield name, device_tensor
                finally:
                    if copy_end is not None:
                        copy_end.record()
                        cuda_events.append((h2d_start, h2d_end, copy_start, copy_end))
                    del device_tensor
                applied_tensor_count += 1
                if (
                    prepared.fault_injection_after_tensors is not None
                    and applied_tensor_count >= prepared.fault_injection_after_tensors
                ):
                    raise RuntimeError(
                        "injected mid-update failure after "
                        f"{applied_tensor_count} tensors"
                    )

        named_tensors = (
            materialize_tensors() if stream_tensors else list(materialize_tensors())
        )
        model_gpu_start = model_gpu_end = None
        if collect_phase_timings and self.device != "cpu":
            model_gpu_start, model_gpu_end = new_event(), new_event()
            model_gpu_start.record()
        model_load_started = time.perf_counter()
        try:
            with _presharded_weight_loader_mode(
                model, prepared.tensors_are_pre_sharded
            ):
                if load_format == "direct":
                    _model_load_weights_direct(model, named_tensors)
                elif load_format in self.custom_weight_loaders:
                    custom_loader = dynamic_import(load_format)
                    custom_loader(model, named_tensors)
                elif load_format is None:
                    model.load_weights(named_tensors)
                else:
                    raise NotImplementedError(f"Unknown load_format={load_format}")
        except BaseException as error:
            failure = error
        finally:
            if model_gpu_end is not None:
                model_gpu_end.record()
            phases["model_load_wall_ms"] = (
                time.perf_counter() - model_load_started
            ) * 1000

        if stream_tensors and self.device != "cpu":
            sync_started = time.perf_counter()
            try:
                # Completion fence for producer-owned IPC/pinned tensors and all
                # derived-cache writes queued by the model loader.
                device_module.current_stream().synchronize()
            except BaseException as error:
                if failure is None:
                    failure = error
            phases["completion_fence_ms"] = (time.perf_counter() - sync_started) * 1000

        if cuda_events:
            phases["gpu_h2d_ms"] = sum(
                start.elapsed_time(end) for start, end, _, _ in cuda_events
            )
            phases["gpu_param_copy_ms"] = sum(
                start.elapsed_time(end) for _, _, start, end in cuda_events
            )
        if model_gpu_start is not None and model_gpu_end is not None:
            phases["gpu_model_total_ms"] = model_gpu_start.elapsed_time(model_gpu_end)
            phases["gpu_derived_postprocess_ms"] = max(
                phases["gpu_model_total_ms"]
                - phases.get("gpu_h2d_ms", 0.0)
                - phases.get("gpu_param_copy_ms", 0.0),
                0.0,
            )
        phases["apply_total_ms"] = (time.perf_counter() - apply_started) * 1000
        status["partial_update"] = mutation_started and failure is not None
        status["applied_tensor_count"] = applied_tensor_count

        if failure is not None:
            suffix = (
                " The draft may be partially updated; keep the instance paused "
                "until an explicit recovery succeeds."
                if mutation_started
                else " No model tensor was applied."
            )
            return False, f"Failed to update weights from tensor: {failure}.{suffix}"
        return True, "Success"

    def update_weights_from_tensor(
        self: WeightUpdater,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
        *,
        stream_tensors: bool = False,
        tensors_are_pre_sharded: bool = False,
        pin_memory: bool = False,
        collect_phase_timings: bool = False,
        fault_injection_after_tensors: Optional[int] = None,
        phase_timings_ms: Optional[Dict[str, float]] = None,
        update_status: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        if load_format == "flattened_bucket":
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )
        phases = phase_timings_ms if phase_timings_ms is not None else {}
        status = update_status if update_status is not None else {}
        try:
            prepared = self.prepare_weights_from_tensor(
                named_tensors,
                load_format,
                stream_tensors=stream_tensors,
                tensors_are_pre_sharded=tensors_are_pre_sharded,
                pin_memory=pin_memory,
                fault_injection_after_tensors=fault_injection_after_tensors,
                phase_timings_ms=phases,
            )
        except BaseException as error:
            status["partial_update"] = False
            return False, f"Failed to validate weights from tensor: {error}."
        return self.apply_prepared_weights_from_tensor(
            prepared,
            load_format,
            stream_tensors=stream_tensors,
            collect_phase_timings=collect_phase_timings,
            update_status=status,
        )

    def _update_weights_from_flattened_bucket(
        self: WeightUpdater,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.get_model().load_weights(reconstructed_tensors)

        return True, "Success"

    def update_weights_from_ipc(self: WeightUpdater, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        self._assert_weight_cache_inactive("update_weights_from_ipc")
        error = _unsupported_derived_weight_cache_error()
        if error is not None:
            return False, error

        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self.get_model_runner())
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device, non_blocking: bool = False):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device, non_blocking=non_blocking)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
