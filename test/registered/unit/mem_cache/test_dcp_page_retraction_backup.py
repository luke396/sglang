import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.mem_cache.memory_pool import (
    DCPPageRetractionError,
    DCPPageRetractionSnapshot,
    HybridLinearKVPool,
    MambaPool,
    MLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pool(rows: int = 32) -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.page_size = 2
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 2
    pool.kv_buffer = [
        torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 1, 3) + layer * 1000
        for layer in range(pool.layer_num)
    ]
    return pool


def _page_parallel(rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        dcp_enabled=True,
        dcp_size=3,
        dcp_rank=rank,
        dcp_kv_layout="page",
    )


def _token_parallel() -> SimpleNamespace:
    return SimpleNamespace(dcp_enabled=True, dcp_kv_layout="token")


def _hybrid_pool() -> HybridLinearKVPool:
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = _pool()
    mamba = object.__new__(MambaPool)
    mamba.mamba_cache = MambaPool.State(
        conv=[torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)],
        temporal=torch.arange(10, dtype=torch.float32).reshape(1, 5, 2) + 100,
    )
    mamba._slot_siblings = []
    pool.mamba_pool = mamba
    pool._mamba_translate = lambda ids: ids
    return pool


class TestDCPPageRetractionBackup(CustomTestCase):
    def test_restores_owned_rows_to_new_fragmented_slots(self):
        pool = _pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])
        next_destination = torch.tensor([60, 61, 62, 63, 64, 65, 18])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            backup = pool.get_cpu_copy(source)
            self.assertIsInstance(backup, DCPPageRetractionSnapshot)
            self.assertEqual(backup.row_count, 3)
            expected = [chunk.clone() for chunk in backup.cpu_tensors[0]]
            for buffer in pool.kv_buffer:
                buffer.zero_()
            pool.load_cpu_copy(backup, destination)
            second_backup = pool.get_cpu_copy(destination)
            pool.load_cpu_copy(second_backup, next_destination)

        self.assertTrue(torch.equal(pool.kv_buffer[0][[16, 17]], expected[0]))
        self.assertTrue(torch.equal(pool.kv_buffer[0][[4]], expected[1]))
        self.assertTrue(torch.equal(pool.kv_buffer[0][[20, 21]], expected[0]))
        self.assertTrue(torch.equal(pool.kv_buffer[0][[6]], expected[1]))
        self.assertTrue(
            torch.equal(pool.kv_buffer[0][[0, 1, 2, 3]], torch.zeros(4, 1, 3))
        )

    def test_rejects_mismatch_before_any_layer_write(self):
        pool = _pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            backup = pool.get_cpu_copy(source)
            wrong_geometry = msgspec.structs.replace(
                backup, row_count=backup.row_count + 1
            )
            before = [buffer.clone() for buffer in pool.kv_buffer]
            with self.assertRaises(ValueError):
                pool.load_cpu_copy(wrong_geometry, destination)

        self.assertTrue(torch.equal(pool.kv_buffer[0], before[0]))
        self.assertTrue(torch.equal(pool.kv_buffer[1], before[1]))

    def test_rejects_bad_late_layer_before_any_layer_write(self):
        pool = _pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            backup = pool.get_cpu_copy(source)
            corrupted_layers = [list(layer) for layer in backup.cpu_tensors]
            corrupted_layers[-1][-1] = torch.zeros((1, 1, 2), dtype=torch.float32)
            corrupted = msgspec.structs.replace(backup, cpu_tensors=corrupted_layers)
            before = [buffer.clone() for buffer in pool.kv_buffer]
            with self.assertRaises(DCPPageRetractionError):
                pool.load_cpu_copy(corrupted, destination)

        self.assertTrue(torch.equal(pool.kv_buffer[0], before[0]))
        self.assertTrue(torch.equal(pool.kv_buffer[1], before[1]))

    def test_rank_with_no_owned_rows_has_empty_snapshot(self):
        pool = _pool()
        first_page = torch.tensor([30, 31])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=2),
        ):
            backup = pool.get_cpu_copy(first_page)
            self.assertEqual(backup.row_count, 0)
            pool.load_cpu_copy(backup, torch.tensor([48, 49]))

        self.assertEqual(backup.cpu_tensors, [[], []])

    def test_rejects_cross_layout_snapshots(self):
        pool = _pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            page_backup = pool.get_cpu_copy(source)
            with self.assertRaises(DCPPageRetractionError):
                pool.load_cpu_copy(page_backup.cpu_tensors, destination)

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_token_parallel(),
        ):
            with self.assertRaises(DCPPageRetractionError):
                pool.load_cpu_copy(page_backup, destination)

    def test_hybrid_rejects_bad_mamba_before_mla_restore(self):
        pool = _hybrid_pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])
        source_state = torch.tensor([1])
        destination_state = torch.tensor([3])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            backup = pool.get_cpu_copy(source, mamba_indices=source_state)
            before_mla = [buffer.clone() for buffer in pool.full_kv_pool.kv_buffer]
            before_mamba = pool.mamba_pool.mamba_cache.temporal.clone()
            bad_mamba = (
                [torch.zeros((1, 2, 2), dtype=torch.float32)],
                backup[1][1],
            )
            with self.assertRaises(DCPPageRetractionError):
                pool.load_cpu_copy(
                    (backup[0], bad_mamba),
                    destination,
                    mamba_indices=destination_state,
                )

        self.assertTrue(torch.equal(pool.full_kv_pool.kv_buffer[0], before_mla[0]))
        self.assertTrue(torch.equal(pool.full_kv_pool.kv_buffer[1], before_mla[1]))
        self.assertTrue(torch.equal(pool.mamba_pool.mamba_cache.temporal, before_mamba))

    def test_hybrid_rejects_missing_mamba_before_mla_restore(self):
        pool = _hybrid_pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])
        source_state = torch.tensor([1])
        destination_state = torch.tensor([3])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            backup = pool.get_cpu_copy(source, mamba_indices=source_state)
            before_mla = [buffer.clone() for buffer in pool.full_kv_pool.kv_buffer]
            before_mamba = pool.mamba_pool.mamba_cache.temporal.clone()
            with self.assertRaises(DCPPageRetractionError):
                pool.load_cpu_copy(
                    (backup[0], None),
                    destination,
                    mamba_indices=destination_state,
                )

        self.assertTrue(torch.equal(pool.full_kv_pool.kv_buffer[0], before_mla[0]))
        self.assertTrue(torch.equal(pool.full_kv_pool.kv_buffer[1], before_mla[1]))
        self.assertTrue(torch.equal(pool.mamba_pool.mamba_cache.temporal, before_mamba))

    def test_hybrid_restores_mla_and_mamba_in_one_snapshot(self):
        pool = _hybrid_pool()
        source = torch.tensor([30, 31, 32, 33, 34, 35, 6])
        destination = torch.tensor([48, 49, 50, 51, 52, 53, 12])
        source_state = torch.tensor([1])
        destination_state = torch.tensor([3])

        with patch(
            "sglang.srt.mem_cache.memory_pool.get_parallel",
            return_value=_page_parallel(rank=0),
        ):
            backup = pool.get_cpu_copy(source, mamba_indices=source_state)
            expected_mla = [
                [chunk.clone() for chunk in layer] for layer in backup[0].cpu_tensors
            ]
            expected_conv = [chunk.clone() for chunk in backup[1][0]]
            expected_mamba = backup[1][1].clone()
            for buffer in pool.full_kv_pool.kv_buffer:
                buffer.zero_()
            pool.mamba_pool.mamba_cache.temporal[:, destination_state] = 0
            pool.load_cpu_copy(backup, destination, mamba_indices=destination_state)

        for layer, expected in enumerate(expected_mla):
            self.assertTrue(
                torch.equal(pool.full_kv_pool.kv_buffer[layer][[16, 17]], expected[0])
            )
            self.assertTrue(
                torch.equal(pool.full_kv_pool.kv_buffer[layer][[4]], expected[1])
            )
        for conv, expected in zip(
            pool.mamba_pool.mamba_cache.conv, expected_conv, strict=True
        ):
            self.assertTrue(torch.equal(conv[:, destination_state], expected))
        self.assertTrue(
            torch.equal(
                pool.mamba_pool.mamba_cache.temporal[:, destination_state],
                expected_mamba,
            )
        )

    def test_resume_discards_and_releases_before_propagating_page_mismatch(self):
        req = SimpleNamespace(
            rid="bad-page-snapshot",
            is_retracted=True,
            return_logprob=False,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = [req]
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.token_to_kv_pool_allocator = SimpleNamespace()
        queue.tree_cache = SimpleNamespace()
        queue.scheduler = SimpleNamespace(output_streamer=MagicMock())
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=4)
        queue._prealloc_required_tokens = MagicMock(return_value=(1, 0))
        queue._pre_alloc = MagicMock()

        with (
            patch(
                "sglang.srt.disaggregation.decode.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup="cpu_tensor"
                ),
            ),
            patch(
                "sglang.srt.disaggregation.decode.retraction_restore",
                side_effect=DCPPageRetractionError("bad geometry"),
            ),
            patch("sglang.srt.disaggregation.decode.retraction_discard") as discard,
            patch("sglang.srt.disaggregation.decode.release_kv_cache") as release,
        ):
            with self.assertRaisesRegex(DCPPageRetractionError, "bad geometry"):
                queue.resume_retracted_reqs()

        discard.assert_called_once()
        release.assert_called_once_with(req, queue.tree_cache, is_insert=False)
        queue.scheduler.output_streamer.stream_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
