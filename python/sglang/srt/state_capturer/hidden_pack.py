"""GPU pack for committed speculative-verify hidden-state rows.

The verify producer owns fixed-capacity, capture-private device twins.  This
module packs only request ``i``'s ``[0, commit_lens[i])`` rows into the
contiguous prefix of such a twin without reading a device scalar on the host.
The total row count stays in a one-element device tensor and is copied to a
pinned header later by the two-phase D2H path.

There are two source layouts for the post-norm hidden state:

* dense verify: aux and last are both ``[bs * stride, ...]``;
* compact verify: aux is already scattered/strided, while last remains ragged
  and request ``i`` starts at ``exclusive_cumsum(verify_lens)[i]``.

The CPU implementation is intentionally kept as the executable reference for
unit tests.  Production CUDA inputs use fixed-shape Triton launches and never
materialize a dynamic-size index tensor.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _build_commit_offsets_kernel(
    commit_lens_ptr,
    bound_lens_ptr,
    out_lens_ptr,
    out_offsets_ptr,
    total_rows_ptr,
    bs,
    stride,
    HAS_BOUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < bs
    lens = tl.load(commit_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = tl.maximum(0, tl.minimum(lens, stride))
    if HAS_BOUND:
        bounds = tl.load(bound_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
        bounds = tl.maximum(0, tl.minimum(bounds, stride))
        lens = tl.minimum(lens, bounds)
    inclusive = tl.cumsum(lens, axis=0)
    tl.store(out_lens_ptr + offs, lens, mask=mask)
    tl.store(out_offsets_ptr + offs, inclusive - lens, mask=mask)
    tl.store(total_rows_ptr, tl.sum(lens, axis=0))


@triton.jit
def _build_offsets_kernel(
    lens_ptr,
    out_offsets_ptr,
    bs,
    stride,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < bs
    lens = tl.load(lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = tl.maximum(0, tl.minimum(lens, stride))
    inclusive = tl.cumsum(lens, axis=0)
    tl.store(out_offsets_ptr + offs, inclusive - lens, mask=mask)


@triton.jit
def _pack_committed_rows_kernel(
    aux_ptr,
    last_ptr,
    cache_loc_ptr,
    tokens_ptr,
    commit_lens_ptr,
    commit_offsets_ptr,
    verify_offsets_ptr,
    out_aux_ptr,
    out_last_ptr,
    out_cache_loc_ptr,
    out_tokens_ptr,
    n_rows,
    stride,
    aux_width,
    last_width,
    aux_row_stride,
    last_row_stride,
    LAST_COMPACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    src_row = tl.program_id(0).to(tl.int64)
    d_block = tl.program_id(1).to(tl.int64)
    req = src_row // stride
    within = src_row % stride
    commit_len = tl.load(commit_lens_ptr + req).to(tl.int64)
    committed = (src_row < n_rows) & (within < commit_len)
    dst_row = tl.load(commit_offsets_ptr + req).to(tl.int64) + within

    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    aux_mask = committed & (d < aux_width)
    aux = tl.load(
        aux_ptr + src_row * aux_row_stride + d,
        mask=aux_mask,
        other=0,
    )
    tl.store(out_aux_ptr + dst_row * aux_width + d, aux, mask=aux_mask)

    if LAST_COMPACT:
        last_src_row = tl.load(verify_offsets_ptr + req).to(tl.int64) + within
    else:
        last_src_row = src_row
    last_mask = committed & (d < last_width)
    last = tl.load(
        last_ptr + last_src_row * last_row_stride + d,
        mask=last_mask,
        other=0,
    )
    tl.store(out_last_ptr + dst_row * last_width + d, last, mask=last_mask)

    scalar_mask = committed & (d_block == 0) & (d == 0)
    # Add ``d * 0`` so Triton sees block pointers matching the block mask;
    # only lane zero performs the scalar row metadata copy.
    cache_loc = tl.load(cache_loc_ptr + src_row + d * 0, mask=scalar_mask, other=0)
    token = tl.load(tokens_ptr + src_row + d * 0, mask=scalar_mask, other=0)
    tl.store(out_cache_loc_ptr + dst_row + d * 0, cache_loc, mask=scalar_mask)
    tl.store(out_tokens_ptr + dst_row + d * 0, token, mask=scalar_mask)


def _validate_pack_args(
    *,
    aux_strided: torch.Tensor,
    last_strided: Optional[torch.Tensor],
    last_compact: Optional[torch.Tensor],
    verify_lens: Optional[torch.Tensor],
    verify_cache_loc: torch.Tensor,
    verify_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
    bs: int,
    stride: int,
    out_aux: torch.Tensor,
    out_last: torch.Tensor,
    out_cache_loc: torch.Tensor,
    out_tokens: torch.Tensor,
    out_commit_lens: torch.Tensor,
    out_commit_offsets: torch.Tensor,
    out_total_rows: torch.Tensor,
    out_verify_offsets: torch.Tensor,
) -> bool:
    if bs <= 0 or stride <= 0:
        raise ValueError(
            f"bs and stride must be positive, got bs={bs}, stride={stride}"
        )
    n_rows = bs * stride
    if aux_strided.ndim != 2 or aux_strided.shape[0] < n_rows:
        raise ValueError("aux_strided does not cover bs * stride rows")
    if verify_cache_loc.numel() < n_rows or verify_tokens.numel() < n_rows:
        raise ValueError("verify cache/token inputs do not cover bs * stride rows")
    if commit_lens.numel() < bs:
        raise ValueError("commit_lens does not cover bs requests")
    if out_aux.shape[0] < n_rows or out_aux.shape[1] != aux_strided.shape[1]:
        raise ValueError("out_aux has incompatible capacity or width")
    if out_cache_loc.numel() < n_rows or out_tokens.numel() < n_rows:
        raise ValueError("output cache/token tensors do not cover bs * stride rows")
    if (
        out_commit_lens.numel() < bs
        or out_commit_offsets.numel() < bs
        or out_verify_offsets.numel() < bs
        or out_total_rows.numel() < 1
    ):
        raise ValueError("output header tensors do not cover bs requests")
    compact = last_strided is None
    last = last_compact if compact else last_strided
    if last is None or last.ndim != 2:
        raise ValueError("exactly one usable last-hidden source is required")
    if out_last.shape[0] < n_rows or out_last.shape[1] != last.shape[1]:
        raise ValueError("out_last has incompatible capacity or width")
    if compact and (verify_lens is None or verify_lens.numel() < bs):
        raise ValueError("compact last-hidden input requires verify_lens")
    return compact


def _pack_committed_rows_cpu(
    *,
    aux_strided: torch.Tensor,
    last_strided: Optional[torch.Tensor],
    last_compact: Optional[torch.Tensor],
    verify_lens: Optional[torch.Tensor],
    verify_cache_loc: torch.Tensor,
    verify_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
    bs: int,
    stride: int,
    out_aux: torch.Tensor,
    out_last: torch.Tensor,
    out_cache_loc: torch.Tensor,
    out_tokens: torch.Tensor,
    out_commit_lens: torch.Tensor,
    out_commit_offsets: torch.Tensor,
    out_total_rows: torch.Tensor,
    out_verify_offsets: torch.Tensor,
) -> None:
    compact = last_strided is None
    lens = commit_lens[:bs].to(dtype=torch.int64).clamp_(0, stride)
    if compact:
        verify = verify_lens[:bs].to(dtype=torch.int64).clamp_(0, stride)
        lens = torch.minimum(lens, verify)
        verify_offsets = torch.cumsum(verify, dim=0) - verify
        out_verify_offsets[:bs].copy_(verify_offsets.to(out_verify_offsets.dtype))
    else:
        verify_offsets = None
        out_verify_offsets[:bs].zero_()
    offsets = torch.cumsum(lens, dim=0) - lens
    out_commit_lens[:bs].copy_(lens.to(out_commit_lens.dtype))
    out_commit_offsets[:bs].copy_(offsets.to(out_commit_offsets.dtype))
    total = int(lens.sum().item())
    out_total_rows[0] = total

    dst = 0
    for req in range(bs):
        count = int(lens[req].item())
        if count == 0:
            continue
        src = req * stride
        out_aux[dst : dst + count].copy_(aux_strided[src : src + count])
        if compact:
            last_src = int(verify_offsets[req].item())
            out_last[dst : dst + count].copy_(last_compact[last_src : last_src + count])
        else:
            out_last[dst : dst + count].copy_(last_strided[src : src + count])
        out_cache_loc[dst : dst + count].copy_(verify_cache_loc[src : src + count])
        out_tokens[dst : dst + count].copy_(verify_tokens[src : src + count])
        dst += count


def pack_committed_verify_rows_into(
    *,
    aux_strided: torch.Tensor,
    last_strided: Optional[torch.Tensor],
    last_compact: Optional[torch.Tensor],
    verify_lens: Optional[torch.Tensor],
    verify_cache_loc: torch.Tensor,
    verify_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
    bs: int,
    stride: int,
    out_aux: torch.Tensor,
    out_last: torch.Tensor,
    out_cache_loc: torch.Tensor,
    out_tokens: torch.Tensor,
    out_commit_lens: torch.Tensor,
    out_commit_offsets: torch.Tensor,
    out_total_rows: torch.Tensor,
    out_verify_offsets: torch.Tensor,
) -> None:
    """Pack committed rows into preallocated output tensors.

    CUDA launches are fixed by ``bs * stride`` and tensor widths; the device
    total is never read on this call.  CPU inputs use the parity reference.
    """
    compact = _validate_pack_args(
        aux_strided=aux_strided,
        last_strided=last_strided,
        last_compact=last_compact,
        verify_lens=verify_lens,
        verify_cache_loc=verify_cache_loc,
        verify_tokens=verify_tokens,
        commit_lens=commit_lens,
        bs=bs,
        stride=stride,
        out_aux=out_aux,
        out_last=out_last,
        out_cache_loc=out_cache_loc,
        out_tokens=out_tokens,
        out_commit_lens=out_commit_lens,
        out_commit_offsets=out_commit_offsets,
        out_total_rows=out_total_rows,
        out_verify_offsets=out_verify_offsets,
    )
    if not aux_strided.is_cuda:
        _pack_committed_rows_cpu(
            aux_strided=aux_strided,
            last_strided=last_strided,
            last_compact=last_compact,
            verify_lens=verify_lens,
            verify_cache_loc=verify_cache_loc,
            verify_tokens=verify_tokens,
            commit_lens=commit_lens,
            bs=bs,
            stride=stride,
            out_aux=out_aux,
            out_last=out_last,
            out_cache_loc=out_cache_loc,
            out_tokens=out_tokens,
            out_commit_lens=out_commit_lens,
            out_commit_offsets=out_commit_offsets,
            out_total_rows=out_total_rows,
            out_verify_offsets=out_verify_offsets,
        )
        return

    device = aux_strided.device
    tensors = (
        last_compact if compact else last_strided,
        verify_cache_loc,
        verify_tokens,
        commit_lens,
        out_aux,
        out_last,
        out_cache_loc,
        out_tokens,
        out_commit_lens,
        out_commit_offsets,
        out_total_rows,
        out_verify_offsets,
    )
    if any(t.device != device for t in tensors):
        raise ValueError("all pack tensors must be on the aux_strided device")
    if compact and verify_lens.device != device:
        raise ValueError("verify_lens must be on the aux_strided device")

    block = triton.next_power_of_2(bs)
    _build_commit_offsets_kernel[(1,)](
        commit_lens,
        verify_lens if compact else commit_lens,
        out_commit_lens,
        out_commit_offsets,
        out_total_rows,
        bs,
        stride,
        HAS_BOUND=compact,
        BLOCK=block,
    )
    if compact:
        _build_offsets_kernel[(1,)](
            verify_lens,
            out_verify_offsets,
            bs,
            stride,
            BLOCK=block,
        )
    else:
        out_verify_offsets[:bs].zero_()

    last = last_compact if compact else last_strided
    block_d = 256
    width = max(aux_strided.shape[1], last.shape[1])
    grid = (bs * stride, triton.cdiv(width, block_d))
    _pack_committed_rows_kernel[grid](
        aux_strided,
        last,
        verify_cache_loc,
        verify_tokens,
        out_commit_lens,
        out_commit_offsets,
        out_verify_offsets,
        out_aux,
        out_last,
        out_cache_loc,
        out_tokens,
        bs * stride,
        stride,
        aux_strided.shape[1],
        last.shape[1],
        aux_strided.stride(0),
        last.stride(0),
        LAST_COMPACT=compact,
        BLOCK_D=block_d,
    )
