# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP).

The default ``token`` layout preserves the historical one-token round robin.
``page`` assigns a physical KV page worth of logical positions to one rank.
All helpers take the *physical* page size explicitly: the allocator's widened
DCP page is ``dcp_size * physical_page_size`` and is deliberately not used as
the page-layout granularity.
"""

import torch

from sglang.srt.runtime_context import get_parallel


def _validate_layout(layout: str, physical_page_size: int | None) -> None:
    if layout not in ("token", "page"):
        raise ValueError(f"unsupported DCP KV layout: {layout!r}")
    if layout == "page" and (physical_page_size is None or physical_page_size <= 0):
        raise ValueError("page DCP layout requires a positive physical_page_size")


def dcp_slot_owner_mask(
    slots: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    *,
    layout: str = "token",
    physical_page_size: int | None = None,
) -> torch.Tensor:
    """Return the owner mask for widened slots, excluding invalid slots.

    Slots, unlike logical positions, include arbitrary allocation page IDs.
    Their offsets still follow the widened virtual-page contract, so page
    ownership is recovered from ``slot // physical_page_size``.
    """
    _validate_layout(layout, physical_page_size)
    valid = slots >= 0
    if layout == "page":
        assert physical_page_size is not None
        owner = torch.remainder(slots // physical_page_size, dcp_size)
    else:
        owner = torch.remainder(slots, dcp_size)
    return valid & (owner == dcp_rank)


def dcp_slots_to_local_rows(
    slots: torch.Tensor,
    dcp_size: int,
    *,
    layout: str = "token",
    physical_page_size: int | None = None,
) -> torch.Tensor:
    """Collapse already-owned widened slots into this rank's physical rows."""
    _validate_layout(layout, physical_page_size)
    if layout == "token":
        return slots // dcp_size
    assert physical_page_size is not None
    virtual_page_size = dcp_size * physical_page_size
    rows = (slots // virtual_page_size) * physical_page_size + torch.remainder(
        slots, physical_page_size
    )
    return torch.where(slots >= 0, rows, torch.full_like(slots, -1))


def _page_prefix_len(
    length: torch.Tensor, dcp_size: int, dcp_rank: int, physical_page_size: int
) -> torch.Tensor:
    """``F(length, rank)`` from the page-layout contract for ``[0, length)``."""
    virtual_page_size = dcp_size * physical_page_size
    whole_pages = length // virtual_page_size
    remainder = torch.remainder(length, virtual_page_size)
    tail = torch.clamp(
        remainder - dcp_rank * physical_page_size, min=0, max=physical_page_size
    )
    return whole_pages * physical_page_size + tail


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
    *,
    layout: str = "token",
    physical_page_size: int | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length for a token or physical-page owner rule.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    _validate_layout(layout, physical_page_size)
    if dcp_size == 1:
        return lens
    if layout == "page":
        assert physical_page_size is not None
        if start is None:
            return _page_prefix_len(lens, dcp_size, dcp_rank, physical_page_size)
        return _page_prefix_len(
            start + lens, dcp_size, dcp_rank, physical_page_size
        ) - _page_prefix_len(start, dcp_size, dcp_rank, physical_page_size)
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    """Keep this rank's share of a read-index tensor, still WIDENED.

    Selection only; the caller collapses via translate_dcp_read_ids.
    """
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
    return kv_indices


def filter_dcp_local_chunk_kv_indices(
    kv_indices: torch.Tensor,
    chunk_starts_cpu: torch.Tensor,
    chunk_seq_lens_cpu: torch.Tensor,
) -> torch.Tensor:
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return kv_indices

    dcp_size = parallel.dcp_size
    parts = []
    offset = 0
    for start, length in zip(chunk_starts_cpu.tolist(), chunk_seq_lens_cpu.tolist()):
        first = (parallel.dcp_rank - start) % dcp_size
        parts.append(kv_indices[offset + first : offset + length : dcp_size])
        offset += length
    return torch.cat(parts)


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))


def guard_dcp_page_layout_forward_mode(forward_mode) -> None:
    """Reject page-layout modes that read KV outside ordinary decode."""
    parallel = get_parallel()
    if not (
        parallel.dcp_enabled and getattr(parallel, "dcp_kv_layout", "token") == "page"
    ):
        return
    if forward_mode.is_decode_or_idle() or forward_mode.is_prebuilt():
        return
    raise NotImplementedError(
        "DCP page layout supports ordinary decode only; extend and target "
        "verify must not read page-layout KV."
    )
