"""CPU tensor-parallel sharding for DFlash/DSpark online draft updates.

The returned per-rank tensors own compact storage. This is important for
filename-backed multiprocessing serialization: a contiguous view can otherwise
keep the full checkpoint storage alive and defeat rank-local payloads.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch

_DSPARK_TARGET_SHARED_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "rotary_emb.",
)
_COLUMN_PARALLEL_NAMES = (
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".gate_proj.",
    ".up_proj.",
)
_ROW_PARALLEL_WEIGHT_NAMES = (
    ".o_proj.weight",
    ".down_proj.weight",
)


def _without_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def _draft_tp_shard_dim(name: str) -> int | None:
    normalized = _without_model_prefix(name)
    if any(token in normalized for token in _COLUMN_PARALLEL_NAMES):
        return 0
    if any(normalized.endswith(token) for token in _ROW_PARALLEL_WEIGHT_NAMES):
        return 1
    return None


def preshard_dflash_named_tensors(
    named_tensors: Iterable[Tuple[str, torch.Tensor]],
    tp_size: int,
) -> tuple[List[List[Tuple[str, torch.Tensor]]], dict]:
    """Filter target-shared tensors and create compact DFlash TP payloads.

    This format retains checkpoint source names; the server runs the normal
    fused QKV/gate-up loader with ``tensors_are_pre_sharded=True`` so existing
    in-place parameter loaders and derived-cache refreshes remain authoritative.
    """

    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    per_rank: List[List[Tuple[str, torch.Tensor]]] = [[] for _ in range(tp_size)]
    source_bytes = 0
    skipped_bytes = 0
    sharded_tensor_count = 0
    replicated_tensor_count = 0

    for name, tensor in named_tensors:
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("named_tensors must contain (str, torch.Tensor) pairs")
        if tensor.device.type != "cpu":
            raise ValueError(
                f"host pre-sharding requires CPU tensor {name!r}, got {tensor.device}"
            )
        tensor_bytes = tensor.numel() * tensor.element_size()
        source_bytes += tensor_bytes
        normalized = _without_model_prefix(name)
        if any(
            normalized.startswith(prefix) for prefix in _DSPARK_TARGET_SHARED_PREFIXES
        ):
            skipped_bytes += tensor_bytes
            continue

        shard_dim = _draft_tp_shard_dim(name)
        if shard_dim is None or tp_size == 1:
            for rank in range(tp_size):
                per_rank[rank].append((name, tensor))
            replicated_tensor_count += 1
            continue
        if tensor.ndim <= shard_dim:
            raise ValueError(
                f"cannot shard {name!r} with shape={tuple(tensor.shape)} on dim={shard_dim}"
            )
        dimension = int(tensor.shape[shard_dim])
        if dimension % tp_size:
            raise ValueError(
                f"{name!r} shape={tuple(tensor.shape)} is not divisible by TP={tp_size} "
                f"on dim={shard_dim}"
            )
        shard_size = dimension // tp_size
        for rank in range(tp_size):
            # clone() forces rank-local storage even when narrow() is already
            # contiguous along dim 0.
            shard = tensor.narrow(shard_dim, rank * shard_size, shard_size).clone(
                memory_format=torch.contiguous_format
            )
            per_rank[rank].append((name, shard))
        sharded_tensor_count += 1

    per_rank_bytes = [
        sum(tensor.numel() * tensor.element_size() for _, tensor in items)
        for items in per_rank
    ]
    return per_rank, {
        "source_bytes": source_bytes,
        "skipped_target_shared_bytes": skipped_bytes,
        "per_rank_bytes": per_rank_bytes,
        "total_rank_payload_bytes": sum(per_rank_bytes),
        "sharded_tensor_count": sharded_tensor_count,
        "replicated_tensor_count": replicated_tensor_count,
        "per_rank_tensor_count": [len(items) for items in per_rank],
    }
