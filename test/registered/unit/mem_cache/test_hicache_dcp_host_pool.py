"""HiCache under decode context parallelism (DCP): host-pool index math.

Under DCP the radix/controller layer works in a widened logical index space
(page_size * dcp_size wide pages, dcp_size * physical capacity), while each
rank's device and host buffers only materialize the owned 1/dcp_size token
shard (owner rule: index % dcp_size == dcp_rank, physical row = index //
dcp_size — the same rule the device-side KV write and page-table kernels
use). These tests cover the translation helper, the logical/physical host
pool sizing, and that the transfer entry points hand *physical* rows to the
kernels.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZE = 8
PHYSICAL_PAGE = 64
WIDENED_PAGE = PHYSICAL_PAGE * DCP_SIZE


def _fake_mla_device_pool(size: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        size=size,
        # Match KVCache's default: this static pool's size already counts tokens.
        host_capacity_tokens=None,
        store_dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )


def _make_host_pool(dcp_rank: int, device_size: int = 1024) -> MLATokenToKVPoolHost:
    return MLATokenToKVPoolHost(
        _fake_mla_device_pool(device_size),
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=WIDENED_PAGE,
        layout="layer_first",
        pin_memory=False,
        device="cpu",
        dcp_size=DCP_SIZE,
        dcp_rank=dcp_rank,
    )


class TestDcpKernelIndices(CustomTestCase):
    def test_identity_without_dcp(self):
        indices = torch.arange(37)
        self.assertIs(maybe_dcp_kernel_indices(indices, 1, 0), indices)

    def test_aligned_page_translates_to_full_physical_page(self):
        # One widened page starting at logical 512 covers physical rows
        # 64..127 on every rank.
        indices = torch.arange(WIDENED_PAGE, 2 * WIDENED_PAGE)
        for rank in range(DCP_SIZE):
            out = maybe_dcp_kernel_indices(indices, DCP_SIZE, rank)
            torch.testing.assert_close(
                out, torch.arange(PHYSICAL_PAGE, 2 * PHYSICAL_PAGE)
            )

    def test_matches_owner_rule_on_merged_unordered_pages(self):
        # Concatenation of non-adjacent widened pages in arbitrary order, as
        # produced by merged CacheOperations after allocator churn.
        pages = [3, 0, 5]
        indices = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in pages]
        )
        for rank in range(DCP_SIZE):
            out = maybe_dcp_kernel_indices(indices, DCP_SIZE, rank)
            expected = (
                indices[indices % DCP_SIZE == rank] // DCP_SIZE
            )  # owner rule, same as filter_dcp_local_kv_indices
            torch.testing.assert_close(out, expected)
            self.assertEqual(out.numel() * DCP_SIZE, indices.numel())

    def test_positional_residue_pairing_survives_host_sort(self):
        # move_indices (direct/layer_first) sorts host indices and permutes
        # device indices to match. Independent residue filtering of both
        # tensors must keep the same token positions on every rank.
        g = torch.Generator().manual_seed(0)
        host_pages = [7, 2]
        device_pages = [1, 4]
        host = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in host_pages]
        )
        device = torch.cat(
            [
                torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE)
                for p in device_pages
            ]
        )
        # token identity: position i pairs host[i] <-> device[i]
        perm = torch.randperm(host.numel(), generator=g)
        # sort host as move_indices does, permuting device alongside
        host_sorted, order = host[perm].sort()
        device_matched = device[perm][order]
        for rank in range(DCP_SIZE):
            host_mask = host_sorted % DCP_SIZE == rank
            device_mask = device_matched % DCP_SIZE == rank
            # same positions selected on both sides -> pairing preserved
            torch.testing.assert_close(host_mask, device_mask)
            self.assertEqual(
                maybe_dcp_kernel_indices(host_sorted, DCP_SIZE, rank).numel(),
                host.numel() // DCP_SIZE,
            )


class TestHostPoolSizingUnderDcp(CustomTestCase):
    def test_logical_and_physical_sizing(self):
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
            return_value=10**12,
        ):
            pool = _make_host_pool(dcp_rank=3)
        # kernel-facing page is physical
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_page_size, WIDENED_PAGE)
        # physical rows = ratio * device physical size, page aligned
        self.assertEqual(pool.size, pool.page_num * PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size * DCP_SIZE)
        # buffers materialize physical rows only
        self.assertEqual(pool.kv_buffer.shape[1], pool.size)
        # allocator surface is logical
        self.assertEqual(pool.free_slots.numel(), pool.logical_size)
        self.assertEqual(pool.mem_state.numel(), pool.logical_size)

    def test_alloc_is_widened_page_granular(self):
        pool = _make_host_pool(dcp_rank=0)
        out = pool.alloc(WIDENED_PAGE)
        self.assertEqual(out.numel(), WIDENED_PAGE)
        with self.assertRaises(AssertionError):
            pool.alloc(PHYSICAL_PAGE)  # not a multiple of the widened page

    def test_non_dcp_pool_unchanged(self):
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
            return_value=10**12,
        ):
            pool = MLATokenToKVPoolHost(
                _fake_mla_device_pool(),
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=PHYSICAL_PAGE,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
            )
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size)
        self.assertEqual(pool.logical_page_size, PHYSICAL_PAGE)

    def test_zero_copy_metadata_folds_widened_indices(self):
        """L3 metadata must identify physical pages, including non-DCP FP8."""
        for layout, dtype in (
            ("layer_first", torch.float16),
            ("page_first_direct", torch.float16),
            ("page_first_direct", torch.float8_e4m3fn),
        ):
            for dcp_size in (1, DCP_SIZE):
                for dcp_rank in range(dcp_size):
                    with self.subTest(
                        layout=layout, dtype=dtype, dcp_size=dcp_size, dcp_rank=dcp_rank
                    ):
                        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
                        pool.dcp_size = dcp_size
                        pool.dcp_rank = dcp_rank
                        pool.page_size = PHYSICAL_PAGE
                        pool.layout = layout
                        pool.layer_num = 2
                        pool.size = 1024
                        pool.kv_cache_dim = 8
                        pool.dtype = dtype
                        shape = (
                            (pool.layer_num, pool.size, 1, pool.kv_cache_dim)
                            if layout == "layer_first"
                            else (
                                pool.size // PHYSICAL_PAGE,
                                pool.layer_num,
                                PHYSICAL_PAGE,
                                1,
                                pool.kv_cache_dim,
                            )
                        )
                        pool.kv_buffer = torch.zeros(shape, dtype=dtype)
                        logical_page = PHYSICAL_PAGE * dcp_size
                        # Nonzero, noncontiguous logical pages must map to the
                        # corresponding physical pages on every DCP rank.
                        logical_indices = torch.cat(
                            [
                                torch.arange(p * logical_page, (p + 1) * logical_page)
                                for p in (2, 4)
                            ]
                        )
                        ptrs, sizes = pool.get_page_buffer_meta(logical_indices)
                        if layout == "layer_first":
                            stride = pool.kv_cache_dim * pool.dtype.itemsize
                            layer_stride = pool.size * stride
                            expected = []
                            for page_start in (2 * PHYSICAL_PAGE, 4 * PHYSICAL_PAGE):
                                for layer_id in range(pool.layer_num):
                                    expected.append(
                                        pool.kv_buffer.data_ptr()
                                        + page_start * stride
                                        + layer_id * layer_stride
                                    )
                            self.assertEqual(ptrs, expected)
                            self.assertEqual(sizes, [PHYSICAL_PAGE * stride] * 4)
                        else:
                            buffers = [pool.kv_buffer[p] for p in (2, 4)]
                            self.assertEqual(ptrs, [buf.data_ptr() for buf in buffers])
                            self.assertEqual(
                                sizes,
                                [buf.numel() * buf.element_size() for buf in buffers],
                            )


class TestTransferEntryPointsTranslate(CustomTestCase):
    def _run_backup(self, pool, host_indices, device_indices):
        device_pool = SimpleNamespace(
            data_ptrs=torch.zeros(2, dtype=torch.uint64),
            kv_buffer=[torch.zeros(1)] * 2,
        )
        # create=True: mla.py imports the kernel only under `if _is_cuda or
        # _is_hip`, so the name is absent on the CPU runner this test targets.
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_all_layer_mla",
            create=True,
        ) as kernel:
            pool.can_use_jit = False
            pool.can_use_write_back_jit = False
            with mock.patch.object(
                MLATokenToKVPoolHost, "_is_device_layer_sharded", return_value=False
            ):
                pool.backup_from_device_all_layer(
                    device_pool, host_indices, device_indices, io_backend="kernel"
                )
            return kernel.call_args.kwargs

    def test_backup_receives_physical_rows(self):
        pool = _make_host_pool(dcp_rank=5)
        logical = torch.arange(2 * WIDENED_PAGE)
        kwargs = self._run_backup(pool, logical, logical.clone())
        expected = torch.arange(2 * PHYSICAL_PAGE)
        torch.testing.assert_close(kwargs["src_indices"], expected)
        torch.testing.assert_close(kwargs["dst_indices"], expected)

    def test_l3_data_page_is_guarded(self):
        pool = _make_host_pool(dcp_rank=0)
        with self.assertRaises(AssertionError):
            pool.get_data_page(0)


if __name__ == "__main__":
    unittest.main()
