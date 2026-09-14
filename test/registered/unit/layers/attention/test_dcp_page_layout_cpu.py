"""CPU contracts for the DCP page-layout MLA integration seams."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention import set_mla_kv_concat_q as fused_module
from sglang.kernels.ops.kvcache import mla_buffer as mla_buffer_module
from sglang.srt.layers.attention import cutedsl_mla_backend as cute_module
from sglang.srt.layers.attention import trtllm_mla_backend as trt_module
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.layers.dcp import layout as dcp_layout_module
from sglang.srt.mem_cache import memory_pool as memory_pool_module
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _parallel(*, layout: str, dcp_size: int = 3, dcp_rank: int = 1):
    return SimpleNamespace(
        dcp_enabled=True,
        dcp_kv_layout=layout,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=dcp_rank,
    )


class TestDcpPageMlaWriterContracts(CustomTestCase):
    def test_pool_passes_its_physical_page_size_to_dcp_writer(self):
        pool = object.__new__(MLATokenToKVPool)
        pool.write_loc_is_dcp_resolved = False
        pool.page_size = 64
        buffer = torch.empty(1)
        loc = torch.tensor([0])
        nope = torch.empty(1)
        rope = torch.empty(1)
        with patch.object(
            memory_pool_module, "set_mla_kv_buffer_dcp_sharded_triton"
        ) as writer:
            pool._scatter_mla_rows(buffer, loc, nope, rope)

        writer.assert_called_once_with(
            buffer,
            loc,
            nope,
            rope,
            physical_page_size=64,
        )

    def test_fused_fp8_wrapper_passes_page_or_token_layout_to_kernel(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 64
        backend.kv_index_translator = SimpleNamespace(is_translating=False)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer: torch.zeros((4, 1, 576))
        )
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=1, v_head_dim=512, head_dim=576
        )
        q = torch.zeros((1, 512), dtype=torch.bfloat16)
        q_rope = torch.zeros((1, 64), dtype=torch.bfloat16)
        k = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
        k_rope = torch.zeros((1, 1, 64), dtype=torch.bfloat16)

        with (
            patch.object(
                trt_module, "set_mla_kv_concat_q_fp8_covered", return_value=True
            ),
            patch.object(
                trt_module, "set_mla_kv_concat_q_fp8", return_value=q
            ) as fused,
        ):
            for layout, expected_page_size in (("token", 0), ("page", 64)):
                with (
                    self.subTest(layout=layout),
                    patch.object(
                        trt_module,
                        "get_parallel",
                        return_value=_parallel(layout=layout),
                    ),
                ):
                    backend._set_kv_and_concat_q_fp8_fused(
                        layer, torch.tensor([3]), q, q_rope, k, k_rope
                    )
                    self.assertEqual(
                        fused.call_args.kwargs["dcp_page_size"], expected_page_size
                    )

    def test_regular_writer_launches_page_constants(self):
        calls = []

        class Kernel:
            def __getitem__(self, _grid):
                def launch(*_args, **kwargs):
                    calls.append(kwargs)

                return launch

        with (
            patch.object(mla_buffer_module, "set_mla_kv_buffer_kernel", Kernel()),
            patch.object(mla_buffer_module, "is_arch_support_pdl", return_value=False),
            patch.object(
                mla_buffer_module,
                "get_parallel",
                return_value=_parallel(layout="page"),
            ),
        ):
            mla_buffer_module.set_mla_kv_buffer_dcp_sharded_triton(
                torch.empty((8, 576)),
                torch.tensor([0, 2, 4]),
                torch.empty((3, 1, 512)),
                torch.empty((3, 1, 64)),
                physical_page_size=2,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["DCP_WORLD_SIZE"], 3)
        self.assertEqual(calls[0]["DCP_RANK"], 1)
        self.assertEqual(calls[0]["DCP_PAGE_SIZE"], 2)
        self.assertTrue(calls[0]["PAGE_LAYOUT"])

    def test_fp8_fused_wrapper_forwards_page_size_through_ffi(self):
        calls = []

        class Module:
            def set_mla_kv_concat_q_fp8(self, *args):
                calls.append(args)

        with patch.object(
            fused_module, "set_mla_kv_concat_q_fp8_module", return_value=Module()
        ):
            fused_module.set_mla_kv_concat_q_fp8(
                torch.empty((4, 576), dtype=torch.float8_e4m3fn),
                torch.tensor([2], dtype=torch.int64),
                torch.zeros((1, 512), dtype=torch.bfloat16),
                torch.zeros((1, 64), dtype=torch.bfloat16),
                torch.zeros((1, 1, 512), dtype=torch.bfloat16),
                torch.zeros((1, 1, 64), dtype=torch.bfloat16),
                dcp_world_size=3,
                dcp_rank=1,
                dcp_page_size=64,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-3:], (3, 1, 64))


class TestDcpPageMetadataContracts(CustomTestCase):
    def test_graph_metadata_refreshes_local_lengths_for_dynamic_batch(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 2
        backend.num_draft_tokens = 8
        backend._fill_dcp_block_kv_indices = MagicMock()
        backend.decode_cuda_graph_metadata = {}

        for batch_size, seq_lens, expected in (
            (3, [1, 4, 9], [0, 2, 3]),
            (2, [5, 7], [2, 2]),
        ):
            with self.subTest(batch_size=batch_size):
                metadata = TRTLLMMLADecodeMetadata(
                    block_kv_indices=torch.full((batch_size, 8), -1, dtype=torch.int32),
                    seq_lens_k=torch.zeros(batch_size, dtype=torch.int32),
                    global_seq_lens_k=torch.zeros(batch_size, dtype=torch.int32),
                )
                backend.decode_cuda_graph_metadata[batch_size] = metadata
                block_storage = metadata.block_kv_indices.data_ptr()
                with patch.object(
                    trt_module, "get_parallel", return_value=_parallel(layout="page")
                ):
                    backend._apply_cuda_graph_metadata(
                        batch_size,
                        torch.arange(batch_size),
                        torch.tensor(seq_lens),
                        ForwardMode.DECODE,
                    )
                torch.testing.assert_close(
                    metadata.global_seq_lens_k,
                    torch.tensor(seq_lens, dtype=torch.int32),
                )
                torch.testing.assert_close(
                    metadata.seq_lens_k, torch.tensor(expected, dtype=torch.int32)
                )
                torch.testing.assert_close(
                    backend._fill_dcp_block_kv_indices.call_args.args[2],
                    torch.tensor(expected, dtype=torch.int32),
                )
                self.assertEqual(metadata.block_kv_indices.data_ptr(), block_storage)
                with patch.object(
                    trt_module, "get_parallel", return_value=_parallel(layout="page")
                ):
                    self.assertEqual(
                        backend._get_dcp_local_max_seq_len(max(seq_lens)), max(expected)
                    )

        # A later replay in the same graph bucket rewrites the existing capture
        # buffers; page tails and a padded zero-length row stay rank-local.
        metadata = backend.decode_cuda_graph_metadata[3]
        block_storage = metadata.block_kv_indices.data_ptr()
        with patch.object(
            trt_module, "get_parallel", return_value=_parallel(layout="page")
        ):
            backend._apply_cuda_graph_metadata(
                3,
                torch.arange(3),
                torch.tensor([8, 0, 7]),
                ForwardMode.DECODE,
            )
        torch.testing.assert_close(
            metadata.seq_lens_k, torch.tensor([2, 0, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.global_seq_lens_k, torch.tensor([8, 0, 7], dtype=torch.int32)
        )
        self.assertEqual(metadata.block_kv_indices.data_ptr(), block_storage)

    def test_page_tail_sets_the_capture_maximum_above_token_stripe_bound(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 2
        with patch.object(
            trt_module,
            "get_parallel",
            return_value=_parallel(layout="page", dcp_size=3, dcp_rank=0),
        ):
            # N=3, S=2, L=8: rank 0 owns pages [0, 1] and the tail [6, 7],
            # so the page-table bound is 4 (the old token stripe bound is 3).
            self.assertEqual(backend._get_dcp_local_max_seq_len(8), 4)

    def test_page_table_launch_passes_page_layout_constants(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = 2
        backend.req_to_token = torch.tensor(
            [[30, 31, 32, 33, 34, 35, 6, 7, 8, 9, 10, 11]], dtype=torch.int64
        )
        backend.kv_index_translator = SimpleNamespace(
            full_v2p_table=None, full_page_multiplier=1
        )
        calls = []

        class Kernel:
            def __getitem__(self, _grid):
                def launch(*args, **kwargs):
                    calls.append((args, kwargs))

                return launch

        table = torch.full((1, 8), -1, dtype=torch.int32)
        with (
            patch.object(trt_module, "create_mla_kv_page_table_for_dcp", Kernel()),
            patch.object(
                trt_module, "get_parallel", return_value=_parallel(layout="page")
            ),
        ):
            backend._fill_dcp_block_kv_indices(
                table, torch.tensor([0]), torch.tensor([3], dtype=torch.int32)
            )

        self.assertIs(calls[0][0][0], backend.req_to_token)
        self.assertEqual(calls[0][1]["PHYSICAL_PAGE_SIZE"], 2)
        self.assertEqual(calls[0][1]["DCP_SIZE"], 3)
        self.assertEqual(calls[0][1]["DCP_RANK"], 1)
        self.assertTrue(calls[0][1]["PAGE_LAYOUT"])


class TestCuteDslDcpWrapperContracts(CustomTestCase):
    def test_forward_decode_derives_page_local_and_token_global_coordinates(self):
        backend = object.__new__(CuteDslMLABackend)
        backend.data_type = torch.bfloat16
        backend.q_data_type = torch.bfloat16
        backend.kv_cache_dim = 576
        backend.page_size = 2
        backend.workspace_buffer = object()
        backend.qk_nope_head_dim = 128
        backend.qk_rope_head_dim = 64
        backend.q_indptr_decode = torch.tensor([0, 1], dtype=torch.int32)
        backend.forward_decode_metadata = SimpleNamespace(
            batch_size=1,
            block_kv_indices=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens_k=None,
            global_seq_lens_k=None,
            max_seq_len_k=3,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: torch.zeros((4, 576), dtype=torch.bfloat16)
        )
        backend.kv_lora_rank = 512
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=1, v_head_dim=512, head_dim=576, scaling=1.0
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            seq_lens=torch.tensor([9], dtype=torch.int32),
        )
        with (
            patch.object(cute_module, "get_in_autotune_dummy_run", return_value=False),
            patch.object(cute_module, "fixup_zero_kv_rows"),
            patch.object(backend, "_compute_decode_bmm1_scale", return_value=1.0),
        ):
            calls = []

            def fake_decode(**kwargs):
                calls.append(kwargs)
                return torch.zeros((1, 1, 1, 512)), torch.zeros((1, 1, 1))

            for layout, causal_seq, cp_world, cp_rank in (
                ("token", 9, 3, 1),
                ("page", 3, 1, 0),
            ):
                with self.subTest(layout=layout):
                    with (
                        patch.object(
                            cute_module,
                            "get_parallel",
                            return_value=_parallel(layout=layout),
                        ),
                        patch.object(
                            trt_module,
                            "get_parallel",
                            return_value=_parallel(layout=layout),
                        ),
                        patch.object(
                            dcp_layout_module,
                            "get_parallel",
                            return_value=_parallel(layout=layout),
                        ),
                        patch.object(
                            cute_module,
                            "flashinfer",
                            SimpleNamespace(
                                decode=SimpleNamespace(
                                    trtllm_batch_decode_with_kv_cache_mla=fake_decode
                                )
                            ),
                            create=True,
                        ),
                    ):
                        backend.forward_decode(
                            torch.zeros((1, 576), dtype=torch.bfloat16),
                            torch.empty(0),
                            torch.empty(0),
                            layer,
                            forward_batch,
                            save_kv_cache=False,
                        )
                    kwargs = calls[-1]
                    self.assertEqual(kwargs["cp_world"], cp_world)
                    self.assertEqual(kwargs["cp_rank"], cp_rank)
                    torch.testing.assert_close(
                        kwargs["causal_seqlens_kv_global"],
                        torch.tensor([causal_seq], dtype=torch.int32),
                    )


class TestDcpPageModeGuard(CustomTestCase):
    def test_extend_is_rejected_before_the_shared_kv_gather(self):
        with patch.object(
            dcp_layout_module, "get_parallel", return_value=_parallel(layout="page")
        ):
            with self.assertRaisesRegex(NotImplementedError, "must not read"):
                dcp_layout_module.guard_dcp_page_layout_forward_mode(
                    ForwardMode.TARGET_VERIFY
                )
            with self.assertRaisesRegex(NotImplementedError, "must not read"):
                dcp_layout_module.guard_dcp_page_layout_forward_mode(
                    ForwardMode.DRAFT_EXTEND_V2
                )
            # PREBUILT is scheduler-only and ordinary decode keeps the existing
            # decode/autotune path; neither is rejected by this model guard.
            dcp_layout_module.guard_dcp_page_layout_forward_mode(ForwardMode.PREBUILT)
            dcp_layout_module.guard_dcp_page_layout_forward_mode(ForwardMode.DECODE)

    def test_cute_out_graph_rejects_unsupported_mode_before_parent_metadata(self):
        backend = object.__new__(CuteDslMLABackend)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        with (
            patch.object(
                dcp_layout_module,
                "get_parallel",
                return_value=_parallel(layout="page"),
            ),
            patch.object(
                trt_module.TRTLLMMLABackend, "init_forward_metadata_out_graph"
            ) as parent_init,
        ):
            with self.assertRaisesRegex(NotImplementedError, "ordinary decode"):
                backend.init_forward_metadata_out_graph(forward_batch)
        parent_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
