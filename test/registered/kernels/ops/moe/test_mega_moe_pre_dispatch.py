import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires Blackwell GPU (sm_100+)",
)
def test_mxfp8_scale_output_uses_padded_row_stride() -> None:
    """Write each MXFP8 scale row without overwriting its physical padding."""
    torch.manual_seed(42)
    num_tokens, padded_max, hidden, top_k = 5, 8, 2304, 8
    num_groups = hidden // 32
    logical_scale_int32 = num_groups // 4
    scale_stride_int32 = 20
    marker = 0xA5

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    topk_idx = (
        torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
        .reshape(num_tokens, top_k)
        .remainder(256)
    )
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)

    buf_x = torch.empty(padded_max, hidden, device="cuda", dtype=torch.float8_e4m3fn)
    scale_bytes = torch.full(
        (padded_max, scale_stride_int32 * 4),
        marker,
        device="cuda",
        dtype=torch.uint8,
    )
    buf_x_sf = scale_bytes.view(torch.int32)[:, :logical_scale_int32]
    buf_topk_idx = torch.empty(padded_max, top_k, device="cuda", dtype=torch.int64)
    buf_topk_weights = torch.empty(
        padded_max, top_k, device="cuda", dtype=torch.float32
    )

    assert buf_x_sf.shape == (padded_max, logical_scale_int32)
    assert buf_x_sf.stride() == (scale_stride_int32, 1)
    mega_moe_pre_dispatch(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )
    torch.cuda.synchronize()

    logical_scale_bytes = logical_scale_int32 * 4
    assert torch.all(scale_bytes[:num_tokens, :logical_scale_bytes] != marker)
    assert torch.all(scale_bytes[:num_tokens, logical_scale_bytes:] == marker)
    assert torch.all(scale_bytes[num_tokens:] == marker)
    torch.testing.assert_close(buf_topk_idx[:num_tokens], topk_idx.to(torch.int64))
    torch.testing.assert_close(buf_topk_weights[:num_tokens], topk_weights)
    assert torch.all(buf_topk_idx[num_tokens:] == -1)
    assert torch.all(buf_topk_weights[num_tokens:] == 0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires Blackwell MegaMoE (SM100 family)",
)
def test_kimi_k3_padding_preserves_mega_moe_output(tmp_path, monkeypatch) -> None:
    """Masking padding must preserve real rows through SiTU dispatch/combine."""
    monkeypatch.setenv("DG_COMM_KERNEL_DEBUG", "0")
    import deep_gemm
    import torch.distributed as dist
    from deep_gemm.utils import per_token_cast_to_fp4

    from sglang.srt.layers.moe.topk import TopKConfig, build_precomputed_topk_output

    # Reuse DeepGEMM v0.1.7's sgl_deep_gemm/tests/test_mega_moe_situ.py
    # shapes, FP4 weight preparation and SiTU call. One rank tests the real
    # kernel without requiring the full K3 checkpoint or a distributed launcher.
    num_tokens, hidden, intermediate, num_experts, top_k = 8, 1024, 512, 8, 2
    generator = torch.Generator(device="cuda").manual_seed(42)

    def cast_weights(shape):
        weights = 0.02 * torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        groups, n, k = shape
        data = torch.empty((groups, n, k // 2), dtype=torch.int8, device="cuda")
        scales = torch.empty((groups, n, k // 32), dtype=torch.float, device="cuda")
        for group_idx in range(groups):
            data[group_idx], scales[group_idx] = per_token_cast_to_fp4(
                weights[group_idx], use_ue8m0=True, gran_k=32
            )
        scales = deep_gemm.transform_sf_into_required_layout(
            scales, n, k, (1, 32), groups
        )
        return data, scales

    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        cast_weights((num_experts, 2 * intermediate, hidden)),
        cast_weights((num_experts, hidden, intermediate)),
        activation="situ",
    )
    x = torch.randn(
        (num_tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    scores = torch.randn((num_tokens, num_experts), generator=generator, device="cuda")
    weights, ids = scores.softmax(dim=-1).topk(top_k, dim=-1)
    weights /= weights.sum(dim=-1, keepdim=True)
    ids = ids.to(torch.int32)
    routed_ids = torch.empty_like(ids)
    count = torch.tensor(num_tokens, dtype=torch.int32, device="cuda")
    output = torch.empty_like(x)
    config = TopKConfig(
        top_k=top_k, renormalize=True, allow_routed_experts_capture=False
    )

    dist.init_process_group(
        "nccl",
        init_method=(tmp_path / "dist_init").as_uri(),
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    buffer = None
    graph = torch.cuda.CUDAGraph()
    try:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD,
            num_experts,
            8192,
            top_k,
            hidden,
            intermediate,
            activation="situ",
        )

        def run(non_padded):
            # Routing produces fresh IDs each forward. Include the restore in
            # capture so growing the valid prefix cannot retain an old -1.
            routed_ids.copy_(ids)
            topk = build_precomputed_topk_output(
                weights, routed_ids, config, layer_id=0, num_token_non_padded=non_padded
            )
            mega_moe_pre_dispatch(
                x,
                topk.topk_ids,
                topk.topk_weights,
                buffer.x,
                buffer.x_sf,
                buffer.topk_idx,
                buffer.topk_weights,
                quant_group_size=32,
            )
            deep_gemm.fp8_fp4_mega_moe(
                output,
                l1,
                l2,
                buffer,
                recipe=(1, 1, 32),
                activation="situ",
                fast_math=True,
            )

        run(None)
        baseline = output.clone()
        assert torch.isfinite(baseline).all()
        assert torch.count_nonzero(baseline) > 0

        def check_output(valid):
            assert torch.isfinite(output).all()
            # Same inputs, weights, kernel and bucket: use the exact equality
            # check from DeepGEMM's tests/test_mega_moe.py correctness comparison.
            torch.testing.assert_close(output[:valid], baseline[:valid], rtol=0, atol=0)
            assert torch.count_nonzero(output[valid:]) == 0
            # Intentionally retain nonzero weights for masked IDs, just as K3
            # does. The kernel must ignore these slots based on the IDs alone.
            torch.testing.assert_close(buffer.topk_weights[:num_tokens], weights)

        for valid in (8, 3, 0, 5):
            count.fill_(valid)
            output.fill_(float("nan"))
            run(count)
            check_output(valid)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(count)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            run(count)

        for valid in (8, 3, 0, 5):
            count.fill_(valid)
            output.fill_(float("nan"))
            graph.replay()
            check_output(valid)
    finally:
        torch.cuda.synchronize()
        graph.reset()
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
