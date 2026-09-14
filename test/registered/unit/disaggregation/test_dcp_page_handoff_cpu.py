"""CPU handoff and retraction contracts for page-layout DCP MLA KV."""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from disagg_test_utils import CopyTransport

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
    DCP_KV_LAYOUT_TOKEN_STRIPE_V1,
)
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    TransferInfo,
)
from sglang.srt.layers.attention import cutedsl_mla_backend as cute_module
from sglang.srt.layers.attention import trtllm_mla_backend as trt_module
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.dcp import layout as dcp_layout_module
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache import memory_pool as memory_pool_module
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _page_parallel(rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        dcp_enabled=True,
        dcp_kv_layout="page",
        dcp_size=3,
        dcp_rank=rank,
        attn_dcp_size=3,
        attn_dcp_rank=rank,
    )


def _pool(rows: int = 64) -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.page_size = 2
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 2
    pool.dtype = torch.float32
    pool.store_dtype = torch.float32
    pool.start_layer = 0
    pool.layer_transfer_counter = None
    pool.kv_buffer = [
        torch.full((rows, 1, 576), -1.0, dtype=torch.float32)
        for _ in range(pool.layer_num)
    ]
    return pool


class TestDcpPageHandoffCpu(CustomTestCase):
    def _allocate_page_retraction(self, pool):
        """Allocate the global page map before P writes the local rows."""
        allocator = PagedTokenToKVPoolAllocator(
            size=192,
            page_size=6,
            dtype=torch.float32,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        priority_pages = [20, 4, 6, 9, 8, 2, 10, 13]
        allocator.free_pages = torch.tensor(
            priority_pages
            + [page for page in range(1, 33) if page not in priority_pages],
            dtype=torch.int64,
        )
        source_slots_all = allocator.alloc(24)
        source_slots = source_slots_all[:19].to(torch.int32)
        torch.testing.assert_close(
            source_slots,
            torch.tensor(
                list(range(120, 126))
                + list(range(24, 30))
                + list(range(36, 42))
                + [54],
                dtype=torch.int32,
            ),
        )
        req = Req(
            rid="page-retraction",
            origin_input_text="",
            origin_input_ids=list(range(19)),
            sampling_params=SamplingParams(),
        )
        req.output_ids.append(5)
        req_to_token = ReqToTokenPool(
            size=3, max_context_len=24, device="cpu", enable_memory_saver=False
        )
        self.assertIsNotNone(req_to_token.alloc([req]))
        req_to_token.write((req.kv.req_pool_idx, slice(0, 19)), source_slots)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 19
        tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=req_to_token,
                token_to_kv_pool_allocator=allocator,
                page_size=6,
            )
        )
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        return req, tree_cache, source_slots_all

    def _send_page_chunks(self, manager, destination, source):
        """Send three chunks that make the complete global prefix [0, 18)."""
        chunks = [
            ([9, 3, 8, 1, 7], [20, 4], 0, 10),
            ([4, 5], [20, 4, 6], 5, 4),
            ([2, 11], [20, 4, 6], 7, 4),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for src_pages, dst_pages, offset, num_tokens in chunks:
                self.assertEqual(
                    manager.send_kvcache_dcp_page(
                        "page-session",
                        np.asarray(src_pages, dtype=np.int32),
                        [destination[1].ctypes.data, destination[0].ctypes.data],
                        np.asarray(dst_pages, dtype=np.int32),
                        dcp_token_item_lens=[source[0][0].nbytes, source[1][0].nbytes],
                        dst_dcp_size=3,
                        dst_dcp_rank=0,
                        src_page_offset=offset,
                        decode_prefix_len=0,
                        num_kv_tokens=num_tokens,
                        executor=executor,
                        dst_layer_ids=[17, 11],
                    ),
                    0,
                )

    def _assert_handoff_bytes(self, pool, source):
        received_rows = torch.tensor([40, 41, 8, 9, 12, 13])
        source_rows = torch.tensor([18, 19, 2, 3, 10, 11])
        received = []
        for layer, buffer in enumerate(pool.kv_buffer):
            expected = torch.from_numpy(source[layer][source_rows.numpy()])
            torch.testing.assert_close(buffer[received_rows], expected)
            received.append(expected.clone())
        return received_rows, received

    def _cute_decode_inputs(self, pool):
        backend = object.__new__(CuteDslMLABackend)
        backend.data_type = torch.float32
        backend.q_data_type = torch.float32
        backend.kv_cache_dim = 576
        backend.page_size = 2
        backend.workspace_buffer = object()
        backend.qk_nope_head_dim = 128
        backend.qk_rope_head_dim = 64
        backend.q_indptr_decode = torch.tensor([0, 1], dtype=torch.int32)
        backend.forward_decode_metadata = SimpleNamespace(
            batch_size=1,
            block_kv_indices=torch.tensor([[20, 4, 6, 9]], dtype=torch.int32),
            seq_lens_k=torch.tensor([7], dtype=torch.int32),
            global_seq_lens_k=torch.tensor([19], dtype=torch.int32),
            max_seq_len_k=7,
        )
        backend.token_to_kv_pool = pool
        backend.kv_lora_rank = 512
        backend._decode_kernel_loc = None
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=1, v_head_dim=512, head_dim=576, scaling=1.0
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            seq_lens=torch.tensor([19], dtype=torch.int32),
            # Global position 18 uses widened slot 54 and local physical row 18.
            out_cache_loc=torch.tensor([54], dtype=torch.int64),
        )
        return backend, layer, forward_batch

    def _grow_at_cute_boundary(self, pool, received_rows, received):
        """Use Cute metadata and control only the final device-kernel edge."""
        backend, layer, forward_batch = self._cute_decode_inputs(pool)
        parallel = _page_parallel()
        kernel_calls = []
        grown_rows = []

        def fake_decode(**kwargs):
            kernel_calls.append(kwargs)
            return torch.zeros((1, 1, 1, 512)), torch.zeros((1, 1, 1))

        pool.size = 64
        pool.kernel_page_blocks = 1
        pool.write_loc_is_dcp_resolved = False
        pool.use_dsa = False
        pool.dsa_kv_cache_store_fp8 = False

        def cpu_writer(dst, loc, nope, rope, *, physical_page_size):
            self.assertEqual(physical_page_size, 2)
            torch.testing.assert_close(loc, torch.tensor([54]))
            # A fixed device boundary result, not another layout implementation.
            dst[18, :, :512] = nope
            dst[18, :, 512:] = rope
            grown_rows.append(dst[18].clone())

        with (
            patch.object(cute_module, "get_in_autotune_dummy_run", return_value=False),
            patch.object(cute_module, "fixup_zero_kv_rows"),
            patch.object(backend, "_compute_decode_bmm1_scale", return_value=1.0),
            patch.object(cute_module, "get_parallel", return_value=parallel),
            patch.object(trt_module, "get_parallel", return_value=parallel),
            patch.object(dcp_layout_module, "get_parallel", return_value=parallel),
            patch.object(memory_pool_module, "get_parallel", return_value=parallel),
            patch.object(
                memory_pool_module,
                "set_mla_kv_buffer_dcp_sharded_triton",
                side_effect=cpu_writer,
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
            for layer_id in range(2):
                layer.layer_id = layer_id
                backend.forward_decode(
                    torch.zeros((1, 576)),
                    torch.full((1, 1, 512), float(layer_id + 1)),
                    torch.empty(0),
                    layer,
                    forward_batch,
                    save_kv_cache=True,
                    k_rope=torch.full((1, 1, 64), float(layer_id + 3)),
                )
        self.assertEqual(kernel_calls[0]["cp_world"], 1)
        self.assertEqual(kernel_calls[0]["cp_rank"], 0)
        self.assertEqual(
            kernel_calls[0]["kv_cache"].data_ptr(), pool.kv_buffer[0].data_ptr()
        )
        torch.testing.assert_close(
            kernel_calls[0]["block_tables"],
            torch.tensor([[20, 4, 6, 9]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            kernel_calls[0]["seq_lens"], torch.tensor([7], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kernel_calls[0]["causal_seqlens_kv_global"],
            torch.tensor([7], dtype=torch.int32),
        )
        self.assertEqual(len(grown_rows), 2)
        received_rows = torch.cat((received_rows, torch.tensor([18])))
        return received_rows, [
            torch.cat((old, grown.unsqueeze(0)))
            for old, grown in zip(received, grown_rows)
        ]

    def _restore_and_release(
        self,
        pool,
        req,
        tree_cache,
        source_slots_all,
        received_rows,
        received,
    ):
        allocator = tree_cache.token_to_kv_pool_allocator
        req_to_token = tree_cache.req_to_token_pool
        with patch.object(
            memory_pool_module, "get_parallel", return_value=_page_parallel()
        ):
            req.offload_kv_cache(req_to_token, allocator)
            self.assertIsNotNone(req.kv.retraction_backup)
            release_kv_cache(req, tree_cache, is_insert=False)
            self.assertEqual(allocator.available_size(), allocator.size)
            self.assertEqual(req_to_token.available_size(), req_to_token.size)
            blocker = Req(
                rid="page-retraction-blocker",
                origin_input_text="",
                origin_input_ids=list(range(24)),
                sampling_params=SamplingParams(),
            )
            self.assertIsNotNone(req_to_token.alloc([blocker]))
            blocker_slots = allocator.alloc(24)
            self.assertIsNotNone(blocker_slots)
            torch.testing.assert_close(blocker_slots, source_slots_all)
            req_to_token.write((blocker.kv.req_pool_idx, slice(0, 24)), blocker_slots)
            blocker.kv.kv_committed_len = blocker.kv.kv_allocated_len = 24
            self.assertIsNotNone(req_to_token.alloc([req]))
            new_slots_all = allocator.alloc(24)
            self.assertIsNotNone(new_slots_all)
            new_slots = new_slots_all[:19].to(torch.int32)
            torch.testing.assert_close(
                new_slots,
                torch.tensor(
                    list(range(48, 54))
                    + list(range(12, 18))
                    + list(range(60, 66))
                    + [78],
                    dtype=torch.int32,
                ),
            )
            req_to_token.write((req.kv.req_pool_idx, slice(0, 19)), new_slots)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 19
            for buffer in pool.kv_buffer:
                buffer.zero_()
            req.load_kv_cache(req_to_token, allocator)
        self.assertIsNone(req.kv.retraction_backup)
        new_rows = torch.tensor([16, 17, 4, 5, 20, 21, 26])
        for layer, buffer in enumerate(pool.kv_buffer):
            torch.testing.assert_close(buffer[new_rows], received[layer])
            self.assertTrue(
                torch.equal(
                    buffer[received_rows], torch.zeros_like(buffer[received_rows])
                )
            )
        release_kv_cache(req, tree_cache, is_insert=False)
        release_kv_cache(blocker, tree_cache, is_insert=False)
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(req_to_token.available_size(), req_to_token.size)

    def test_page_handoff_cute_boundary_and_oom_restore_use_new_rows(self):
        """Allocate, hand off, grow, then restore and release the same prefix."""
        pool = _pool()
        source = [
            np.arange(64 * 576, dtype=np.float32).reshape(64, 1, 576) + layer * 100_000
            for layer in range(2)
        ]
        destination = [buffer.numpy() for buffer in pool.kv_buffer]
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=2,
            kv_data_ptrs=[buffer.ctypes.data for buffer in source],
            kv_layer_ids=[11, 17],
            num_draft_entries=0,
        )
        manager.enable_custom_mem_pool = False
        manager.enable_deferred_decode_kv_release = False
        manager._transfer_data = CopyTransport(source, destination)
        req, tree_cache, source_slots_all = self._allocate_page_retraction(pool)
        self._send_page_chunks(manager=manager, destination=destination, source=source)
        received_rows, received = self._assert_handoff_bytes(pool=pool, source=source)
        received_rows, received = self._grow_at_cute_boundary(
            pool=pool, received_rows=received_rows, received=received
        )
        self._restore_and_release(
            pool=pool,
            req=req,
            tree_cache=tree_cache,
            source_slots_all=source_slots_all,
            received_rows=received_rows,
            received=received,
        )

    def test_rank_without_mla_page_still_sends_mamba_and_aux(self):
        """A short request can have no local page without dropping K3 state."""
        kv_source = np.arange(64, dtype=np.uint8).reshape(32, 1, 2)
        kv_destination = np.full_like(kv_source, 0xEE)
        mamba_source = np.arange(32, dtype=np.uint8).reshape(4, 8)
        aux_source = (np.arange(24, dtype=np.uint8) + 100).reshape(3, 8)
        mamba_destination = np.full_like(mamba_source, 0xEE)
        aux_destination = np.full_like(aux_source, 0xEE)

        manager = object.__new__(MooncakeKVManager)
        manager.enable_custom_mem_pool = False
        manager.enable_deferred_decode_kv_release = False
        manager.pp_size = 1
        manager.attn_tp_size = 1
        manager.kv_args = SimpleNamespace(
            page_size=2,
            kv_data_ptrs=[kv_source.ctypes.data],
            kv_layer_ids=[],
            num_draft_entries=0,
            state_types=[StateType.MAMBA],
            state_data_ptrs=[[mamba_source.ctypes.data]],
            state_item_lens=[[mamba_source[0].nbytes]],
            state_dim_per_tensor=[[]],
            state_conv_shard_groups=[[]],
            state_slice_outer_counts=[[]],
            state_layer_ids=[[]],
            aux_data_ptrs=[aux_source.ctypes.data],
            aux_item_lens=[aux_source[0].nbytes],
        )
        manager._transfer_data = CopyTransport(
            [kv_source, mamba_source, aux_source],
            [kv_destination, mamba_destination, aux_destination],
        )

        # Rank 3 owns none of the one-page, two-token request.  The real page
        # sender therefore has no MLA descriptor and does not require a target
        # KV page registration for this rank.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            self.assertEqual(
                manager.send_kvcache_dcp_page(
                    "short-request",
                    np.asarray([13], dtype=np.int32),
                    [kv_destination.ctypes.data],
                    np.asarray([], dtype=np.int32),
                    dcp_token_item_lens=[kv_source[0].nbytes],
                    dst_dcp_size=4,
                    dst_dcp_rank=3,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=2,
                    executor=executor,
                    dst_layer_ids=[],
                ),
                0,
            )

            transfer = SimpleNamespace(
                mooncake_session_id="short-request",
                dst_state_indices=[[1]],
                dst_aux_index=1,
            )
            target = SimpleNamespace(
                dst_state_data_ptrs=[[mamba_destination.ctypes.data]],
                dst_state_item_lens=[[mamba_destination[0].nbytes]],
                dst_state_dim_per_tensor=[[]],
                dst_state_layer_ids=[[]],
                dst_attn_tp_size=1,
                dst_tp_rank=0,
            )
            self.assertEqual(
                manager.maybe_send_extra(transfer, [[0]], executor, target), 0
            )
            self.assertEqual(
                manager.send_aux(transfer, 0, [aux_destination.ctypes.data]), 0
            )

        np.testing.assert_array_equal(mamba_destination[1], mamba_source[0])
        np.testing.assert_array_equal(aux_destination[1], aux_source[0])

    def test_one_prefill_routes_token_and_page_targets_per_session(self):
        """One P worker keeps the target layout selection session-local."""
        room = 81
        source = np.zeros((8, 1, 2), dtype=np.uint8)
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = False
        manager._page_transfer_rooms = set()
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager.failed_sessions = set()
        manager.session_lock = threading.Lock()
        manager.kv_args = SimpleNamespace(
            page_size=2,
            kv_data_ptrs=[source.ctypes.data],
            kv_layer_ids=[],
            num_draft_entries=0,
        )
        manager._dcp_pack_buffers = []
        manager.req_to_decode_prefix_len = {}
        manager._prefill_unique_rank = lambda: 0
        manager._get_dsa_cache_transfer_skip_flags = lambda _registration: (
            False,
            False,
        )

        def registration(session, layout):
            return KVArgsRegisterInfo(
                room="None",
                endpoint="127.0.0.1",
                dst_port=9999,
                mooncake_session_id=session,
                dst_kv_ptrs=[source.ctypes.data],
                dst_aux_ptrs=[],
                dst_state_data_ptrs=[],
                dst_tp_rank=0,
                dst_attn_tp_size=1,
                dst_kv_item_len=source[0].nbytes,
                dst_state_item_lens=[],
                dst_state_dim_per_tensor=[],
                dst_kv_layer_ids=[],
                dst_state_layer_ids=[],
                dst_dcp_size=3,
                dst_dcp_rank=0,
                requires_dcp_relayout=True,
                dcp_token_item_lens=[source[0].nbytes],
                dcp_kv_layout=layout,
            )

        def transfer(session):
            return TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=9999,
                mooncake_session_id=session,
                dst_kv_indices=np.asarray([0], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=[],
                required_dst_info_num=2,
                is_dummy=False,
                decode_prefix_len=0,
            )

        manager.decode_kv_args_table = {
            "page": registration("page", DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1),
            "token": registration("token", DCP_KV_LAYOUT_TOKEN_STRIPE_V1),
        }
        manager.transfer_infos = {
            room: {"page": transfer("page"), "token": transfer("token")}
        }
        calls = []
        dispatched = threading.Event()

        def record(layout):
            def sender(session, *_args, **_kwargs):
                calls.append((layout, session))
                if len(calls) == 2:
                    dispatched.set()
                return 0

            return sender

        manager.send_kvcache_dcp_page = MagicMock(side_effect=record("page"))
        manager.send_kvcache_dcp = MagicMock(side_effect=record("token"))
        queue = FastQueue()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown)
        worker = threading.Thread(
            target=manager.transfer_worker,
            args=(queue, executor),
            daemon=True,
        )
        worker.start()
        queue.put(
            TransferKVChunk(
                room=room,
                prefill_kv_indices=np.asarray([0], dtype=np.int32),
                index_slice=slice(0, 1),
                is_last_chunk=False,
                prefill_aux_index=None,
                state_indices=None,
                num_kv_tokens=2,
            )
        )

        self.assertTrue(dispatched.wait(timeout=2))
        self.assertCountEqual(calls, [("page", "page"), ("token", "token")])


if __name__ == "__main__":
    unittest.main()
