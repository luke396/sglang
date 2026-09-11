"""K3 padding regressions; the B300 test checks the recorder's HTTP export."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import TopKConfig, build_precomputed_topk_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_moe():
    # Follow test_kimi_k3_bfa_overlap.py: call the real model method with a small
    # owner, replacing weight-dependent compute only.
    def route(hidden_states, *, prefix_sum, num_token_non_padded):
        return build_precomputed_topk_output(
            torch.ones_like(hidden_states, dtype=torch.float32),
            hidden_states.clone(),
            TopKConfig(top_k=2, renormalize=True),
            layer_id=1,
            num_token_non_padded=num_token_non_padded,
        ).topk_ids

    return SimpleNamespace(
        _dp_attention=True,
        _ep_a2a=True,
        _record_expert_distribution=True,
        _eligible_for_fused_front=False,
        _forward_unfused=route,
    )


class TestKimiK3ExpertMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(get_parallel().override(attn_cp_size=1))
        self.batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=8,
            input_ids=torch.arange(8),
            req_pool_indices=torch.arange(8),
            seq_lens=torch.ones(8, dtype=torch.int64),
            out_cache_loc=torch.arange(8),
            seq_lens_sum=8,
        )

        def record(*, topk_ids):
            self.recorded_ids = topk_ids.clone()

        recorder = self.stack.enter_context(
            patch("sglang.srt.layers.moe.topk.get_global_expert_distribution_recorder")
        ).return_value
        recorder.on_select_experts.side_effect = record
        self.stack.enter_context(patch("sglang.srt.layers.moe.topk._is_cuda", False))
        self.stack.enter_context(
            patch(
                "sglang.srt.layers.moe.topk._can_fuse_padded_region",
                return_value=False,
            )
        )

    def test_padding_reaches_recorder_after_sp_sharding(self):
        # Same layout-table approach as test_num_token_non_padded_localization.py.
        # An eight-row DP bucket is split into four two-row SP shards.
        ids = torch.tensor([[0, 32], [0, 32]], dtype=torch.int32)
        owner = _make_moe()
        for valid, expected_counts in (
            (8, (2, 2, 2, 2)),
            (3, (2, 1, 0, 0)),
            (1, (1, 0, 0, 0)),
            (0, (0, 0, 0, 0)),
        ):
            self.batch.global_num_token_non_padded = torch.tensor(valid)
            for shard, count in enumerate(expected_counts):
                with self.subTest(valid=valid, shard=shard):
                    self.batch.num_token_non_padded = torch.tensor(count)
                    self.batch.attn_tp_sequence_sharded = True
                    with patch(
                        "sglang.srt.models.kimi_k3.torch.clamp",
                        side_effect=AssertionError("Runner-local count was recomputed"),
                    ):
                        routed = KimiK3MoE.forward(
                            owner, ids, forward_batch=self.batch, token_offset=shard * 2
                        )
                    expected = ids.clone()
                    expected[count:] = -1
                    torch.testing.assert_close(routed, expected)
                    torch.testing.assert_close(self.recorded_ids, expected)

                    # With upstream gather disabled, K3 still shards the MoE.
                    self.batch.attn_tp_sequence_sharded = False
                    self.batch.num_token_non_padded = torch.tensor(valid)
                    routed = KimiK3MoE.forward(
                        owner, ids, forward_batch=self.batch, token_offset=shard * 2
                    )
                    torch.testing.assert_close(routed, expected)
                    torch.testing.assert_close(self.recorded_ids, expected)

    def test_graph_batch_uses_existing_local_scalar(self):
        # Decode capture batches historically only carry the LOCAL scalar.
        self.batch.num_token_non_padded = torch.tensor(1)
        self.batch.attn_tp_sequence_sharded = True
        ids = torch.tensor([[0, 32], [0, 32]], dtype=torch.int32)
        routed = KimiK3MoE.forward(
            _make_moe(), ids, forward_batch=self.batch, token_offset=2
        )
        torch.testing.assert_close(
            routed, torch.tensor([[0, 32], [-1, -1]], dtype=ids.dtype)
        )

    def test_full_batch_fallback_uses_global_count(self):
        # The runner localized its count, but this layer fell back to all-reduce.
        self.batch.global_num_token_non_padded = torch.tensor(3)
        self.batch.num_token_non_padded = torch.tensor(1)
        self.batch.attn_tp_sequence_sharded = True
        ids = torch.tensor([[0, 32]] * 8, dtype=torch.int32)
        routed = KimiK3MoE.forward(_make_moe(), ids, forward_batch=self.batch)
        expected = ids.clone()
        expected[3:] = -1
        torch.testing.assert_close(routed, expected)
        torch.testing.assert_close(self.recorded_ids, expected)

    def test_recording_off_does_not_require_token_counts(self):
        owner = _make_moe()
        owner._record_expert_distribution = False
        ids = torch.tensor([[0, 32], [0, 32]], dtype=torch.int32)
        routed = KimiK3MoE.forward(owner, ids, forward_batch=self.batch)
        torch.testing.assert_close(routed, ids)


if __name__ == "__main__":
    unittest.main()
