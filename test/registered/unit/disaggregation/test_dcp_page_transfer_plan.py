import unittest

import numpy as np

from sglang.srt.disaggregation.common.utils import build_dcp_page_transfer_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _build_plan(*, src, dst, page_size, dcp_size, dcp_rank, **kwargs):
    return build_dcp_page_transfer_plan(
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        physical_page_size=page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        **kwargs,
    )


def _enumerated_spans(
    *, src, dst, page_size, dcp_size, dcp_rank, src_page_offset, decode_prefix_len, m
):
    """Reference each token independently, then coalesce adjacent addresses."""
    rows = []
    for token_offset in range(m):
        source_chunk_page, source_page_offset = divmod(token_offset, page_size)
        relative_page = src_page_offset + source_chunk_page
        logical_position = (
            decode_prefix_len + relative_page * page_size + source_page_offset
        )
        if (logical_position // page_size) % dcp_size != dcp_rank:
            continue
        destination_page = dst[relative_page // dcp_size]
        rows.append(
            (
                src[source_chunk_page] * page_size + source_page_offset,
                destination_page * page_size + source_page_offset,
            )
        )

    spans = []
    for src_row, dst_row in rows:
        if (
            spans
            and src_row == spans[-1][0] + spans[-1][2]
            and dst_row == spans[-1][1] + spans[-1][2]
        ):
            spans[-1][2] += 1
        else:
            spans.append([src_row, dst_row, 1])
    return spans


class TestDcpPageTransferPlan(CustomTestCase):
    def assertPlanMatchesEnumeration(self, *, src, dst, **kwargs):
        plan = _build_plan(src=src, dst=dst, **kwargs)
        reference_kwargs = dict(kwargs)
        m = reference_kwargs.pop("num_kv_tokens")
        expected = _enumerated_spans(src=src, dst=dst, m=m, **reference_kwargs)
        self.assertEqual(
            list(zip(plan.src_token_starts, plan.dst_token_starts, plan.token_counts)),
            [tuple(span) for span in expected],
        )

    def test_fragmented_pages_prefix_offset_and_tail(self):
        # The five source pages are intentionally fragmented.  The fifth page
        # contributes the one-token tail; src_page_offset=1 shifts ownership.
        src = [9, 3, 8, 1, 7]
        dst = [20, 4]
        for rank in range(3):
            with self.subTest(rank=rank):
                self.assertPlanMatchesEnumeration(
                    src=src,
                    dst=dst,
                    page_size=2,
                    dcp_size=3,
                    dcp_rank=rank,
                    src_page_offset=1,
                    decode_prefix_len=6,
                    num_kv_tokens=9,
                )

    def test_chunk_offsets_keep_owner_phase_through_the_final_tail(self):
        chunks = [
            ([30, 9], 1, 6),
            ([5, 18], 3, 6),
            ([12], 5, 2),
        ]
        dst = [100, 40]
        for rank in range(3):
            for src, src_page_offset, num_kv_tokens in chunks:
                with self.subTest(
                    rank=rank,
                    src_page_offset=src_page_offset,
                    num_kv_tokens=num_kv_tokens,
                ):
                    self.assertPlanMatchesEnumeration(
                        src=src,
                        dst=dst,
                        page_size=3,
                        dcp_size=3,
                        dcp_rank=rank,
                        src_page_offset=src_page_offset,
                        decode_prefix_len=0,
                        num_kv_tokens=num_kv_tokens,
                    )

    def test_merges_only_when_both_physical_page_sequences_are_adjacent(self):
        plan = _build_plan(
            src=[10, 3, 11],
            dst=[5, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=0,
            decode_prefix_len=0,
            num_kv_tokens=6,
        )
        # rank 0 owns source chunk pages 0 and 2.  They are separate logical
        # pages but adjacent physical source and destination pages, so one RDMA
        # span covers both.
        np.testing.assert_array_equal(plan.src_token_starts, [20])
        np.testing.assert_array_equal(plan.dst_token_starts, [10])
        np.testing.assert_array_equal(plan.token_counts, [4])

    def test_rank_with_no_local_page_needs_no_destination_page(self):
        plan = _build_plan(
            src=[13],
            dst=[],
            page_size=2,
            dcp_size=4,
            dcp_rank=3,
            decode_prefix_len=0,
            num_kv_tokens=2,
        )
        self.assertEqual(plan.src_token_starts.size, 0)
        self.assertEqual(plan.dst_token_starts.size, 0)
        self.assertEqual(plan.token_counts.size, 0)

    def test_rejects_invalid_rdma_boundaries_before_a_plan_exists(self):
        base = {
            "src": [0],
            "dst": [0],
            "page_size": 2,
            "dcp_size": 2,
            "dcp_rank": 0,
            "decode_prefix_len": 0,
            "num_kv_tokens": 2,
        }
        for overrides in (
            {"decode_prefix_len": 1},
            {"num_kv_tokens": 3},
            {"dst": []},
            {"src": [-1]},
            {"dst": [-1]},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                _build_plan(**(base | overrides))


if __name__ == "__main__":
    unittest.main()
