"""CPU lifecycle contracts for Mooncake DCP page transfers.

These tests deliberately use the real sender/worker/receiver lifecycle methods.
Only the RDMA executor, network send, clock, and cache release boundaries are
controlled, so a successful test demonstrates the ordering contract rather
than a duplicate implementation of it.
"""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
    CommonKVManager,
    KVTransferError,
)
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _RunningWrite(concurrent.futures.Future):
    """A future whose RDMA completion is released by the test."""

    def __init__(self):
        super().__init__()
        self.set_running_or_notify_cancel()


class _IdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _bare_decode_manager(room):
    manager = CommonKVManager.__new__(CommonKVManager)
    manager._deferred_abort_ack_tracker = {room: set()}
    manager.failure_lock = threading.Lock()
    manager.failure_records = {room: "prefill failed"}
    manager.request_status = {room: KVPoll.Failed}
    manager.required_prefill_response_num_table = {}
    manager.prefill_response_tracker = {}
    manager.addr_to_rooms_tracker = defaultdict(set)
    return manager


def _bare_decode_queue():
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue._deferred_releases = []
    queue.deferred_kv_release_timeout = 1.0
    queue.enable_staging = False
    queue.staging_handler = None
    queue.tree_cache = object()
    queue.metadata_buffers = SimpleNamespace(bootstrap_room={})
    queue.req_to_metadata_buffer_idx_allocator = _IdxAllocator()
    return queue


class TestPageSenderPoll(CustomTestCase):
    def test_failed_page_room_stays_transferring_until_active_write_drains(self):
        room = 71
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager._page_transfer_rooms = {room}
        manager._staging_outstanding = defaultdict(int, {room: 1})

        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.bootstrap_room = room
        sender.kv_mgr = manager
        sender.conclude_state = None
        sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)

        # This is the state transition performed by the P bootstrap thread when
        # Decode aborts while the page RDMA future is still running.
        CommonKVManager.update_status(manager, room, KVPoll.Failed)
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNone(sender.conclude_state)
        self.assertEqual(manager.request_status[room], KVPoll.Failed)

        manager._staging_outstanding[room] = 0
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)


class TestPageWorkerFailureDrain(CustomTestCase):
    def _manager_and_chunk(self):
        room = 72
        source = [np.zeros((8, 1), dtype=np.uint8) for _ in range(2)]
        destination = [np.zeros((8, 1), dtype=np.uint8) for _ in range(2)]
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = False
        manager.enable_custom_mem_pool = True
        manager._page_transfer_rooms = {room}
        manager._staging_outstanding = defaultdict(int, {room: 1})
        manager._deferred_ack_targets = {room: ("127.0.0.1", 9999)}
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.session_lock = threading.Lock()
        manager.attn_tp_rank = 0
        manager.attn_cp_rank = 0
        manager.attn_cp_size = 1
        manager.pp_rank = 0
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            page_size=4,
            kv_data_ptrs=[item.ctypes.data for item in source],
            kv_layer_ids=[0, 1],
            num_draft_entries=0,
        )
        manager.req_to_decode_prefix_len = {room: 0}
        manager._get_dsa_cache_transfer_skip_flags = lambda _info: (False, False)
        manager._send_multipart_locked = lambda *_args, **_kwargs: None

        registration = KVArgsRegisterInfo(
            room="None",
            endpoint="127.0.0.1",
            dst_port=9999,
            mooncake_session_id="page-session",
            dst_kv_ptrs=[item.ctypes.data for item in destination],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            dst_tp_rank=0,
            dst_attn_tp_size=1,
            dst_kv_item_len=4,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_kv_layer_ids=[0, 1],
            dst_state_layer_ids=[],
            dst_dcp_size=1,
            dst_dcp_rank=0,
            requires_dcp_relayout=True,
            dcp_token_item_lens=[1, 1],
            dcp_kv_layout=DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
        )
        transfer_info = TransferInfo(
            room=room,
            endpoint="127.0.0.1",
            dst_port=9999,
            mooncake_session_id="page-session",
            dst_kv_indices=np.array([0], dtype=np.int32),
            dst_aux_index=0,
            dst_state_indices=[],
            required_dst_info_num=1,
            is_dummy=False,
            decode_prefix_len=0,
            dcp_kv_layout=DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
        )
        manager.decode_kv_args_table = {"page-session": registration}
        manager.transfer_infos = {room: {"page-session": transfer_info}}
        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([0], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            num_kv_tokens=4,
            staging_counted=True,
        )
        return manager, room, chunk

    def test_partial_submit_exception_concludes_and_acks_after_drain(self):
        manager, room, chunk = self._manager_and_chunk()
        running = _RunningWrite()
        submit_failed = threading.Event()
        acked = threading.Event()
        acks = []

        def submit(_function, *_args):
            if submit_failed.is_set():
                raise RuntimeError("executor rejected second page layer")
            submit_failed.set()
            return running

        manager._send_abort_ack = lambda _ip, _port, ack_room: (
            acks.append(ack_room),
            acked.set(),
        )
        queue = FastQueue()
        worker = threading.Thread(
            target=manager.transfer_worker,
            args=(queue, SimpleNamespace(submit=submit)),
            daemon=True,
        )
        worker.start()
        queue.put(chunk)

        self.assertTrue(submit_failed.wait(timeout=2))
        self.assertFalse(acked.is_set(), "worker must wait for submitted RDMA")
        self.assertEqual(manager._staging_outstanding[room], 1)

        running.set_result(0)
        self.assertTrue(acked.wait(timeout=2))
        self.assertEqual(acks, [room])
        self.assertEqual(manager.request_status[room], KVPoll.Failed)
        self.assertEqual(manager._staging_outstanding.get(room, 0), 0)
        self.assertNotIn(room, manager._deferred_ack_targets)

    def test_cancel_during_mla_write_stops_state_and_aux_submission(self):
        manager, room, chunk = self._manager_and_chunk()
        chunk.is_last_chunk = True
        chunk.state_indices = [[3]]
        running = _RunningWrite()
        submitted = threading.Event()
        acked = threading.Event()
        manager.maybe_send_extra = MagicMock(return_value=0)
        manager.send_aux = MagicMock(return_value=0)
        manager._send_abort_ack = lambda *_args: acked.set()

        def submit(*_args):
            submitted.set()
            return running

        queue = FastQueue()
        worker = threading.Thread(
            target=manager.transfer_worker,
            args=(queue, SimpleNamespace(submit=submit)),
            daemon=True,
        )
        worker.start()
        queue.put(chunk)
        self.assertTrue(submitted.wait(timeout=2))
        manager.update_status(room, KVPoll.Failed)
        running.set_result(0)
        self.assertTrue(acked.wait(timeout=2))
        manager.maybe_send_extra.assert_not_called()
        manager.send_aux.assert_not_called()
        self.assertEqual(manager.request_status[room], KVPoll.Failed)
        self.assertEqual(manager._staging_outstanding.get(room, 0), 0)

    def test_state_and_aux_exceptions_follow_the_page_drain_path(self):
        for failing_write in ("maybe_send_extra", "send_aux"):
            with self.subTest(failing_write=failing_write):
                manager, room, chunk = self._manager_and_chunk()
                chunk.is_last_chunk = True
                chunk.state_indices = [[3]]
                manager.enable_custom_mem_pool = False
                manager._transfer_data = MagicMock(return_value=0)
                manager.maybe_send_extra = MagicMock(return_value=0)
                manager.send_aux = MagicMock(return_value=0)
                getattr(manager, failing_write).side_effect = RuntimeError(
                    "synchronous write failed"
                )
                acked = threading.Event()
                manager._send_abort_ack = lambda *_args: acked.set()
                queue = FastQueue()
                worker = threading.Thread(
                    target=manager.transfer_worker,
                    args=(queue, None),
                    daemon=True,
                )
                worker.start()
                queue.put(chunk)
                self.assertTrue(acked.wait(timeout=2))
                getattr(manager, failing_write).assert_called_once()
                if failing_write == "maybe_send_extra":
                    manager.send_aux.assert_not_called()
                self.assertEqual(manager.request_status[room], KVPoll.Failed)
                self.assertEqual(manager._staging_outstanding.get(room, 0), 0)
                self.assertNotIn(room, manager._deferred_ack_targets)


class TestPageDecodeQuarantine(CustomTestCase):
    def _receiver_and_request(self, room=73):
        manager = _bare_decode_manager(room)
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = room
        receiver.bootstrap_addr = "bootstrap"
        receiver.conclude_state = KVPoll.Failed
        receiver._is_page_layout = True
        receiver._page_metadata_sent = True
        receiver.abort_notified = True
        receiver.bootstrap_infos = [{"rank": 0}, {"rank": 1}]
        request = SimpleNamespace(
            bootstrap_room=room, kv_state="held", aux_state="held"
        )
        return (
            manager,
            receiver,
            DecodeRequest(req=request, kv_receiver=receiver, metadata_buffer_index=9),
        )

    def test_failure_exception_keeps_receiver_until_full_drain_releases_once(self):
        manager, receiver, decode_request = self._receiver_and_request()
        queue = _bare_decode_queue()

        with self.assertRaises(KVTransferError):
            receiver.failure_exception()
        self.assertIs(decode_request.kv_receiver, receiver)
        self.assertIn(decode_request.req.bootstrap_room, manager.request_status)

        # Rank 0's ACK arrived before the queue starts resolving. Deferring the
        # request must not wipe that evidence, and a passed deadline is still not
        # permission to release a page target.
        manager.note_abort_ack(decode_request.req.bootstrap_room, 0)
        queue._defer_release(decode_request)
        queue._deferred_releases[0] = (
            decode_request,
            float("-inf"),
            decode_request.metadata_buffer_index,
            2,
        )
        with patch.object(decode_mod, "release_kv_cache") as release:
            queue.resolve_deferred_releases()
            release.assert_not_called()
            self.assertEqual(decode_request.req.kv_state, "held")
            self.assertEqual(decode_request.req.aux_state, "held")
            self.assertIs(decode_request.kv_receiver, receiver)

            manager.note_abort_ack(decode_request.req.bootstrap_room, 1)
            queue.resolve_deferred_releases()
            release.assert_called_once_with(
                decode_request.req, queue.tree_cache, is_insert=False
            )
            queue.resolve_deferred_releases()
            release.assert_called_once()

        self.assertIsNone(decode_request.kv_receiver)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.freed, [9])
        self.assertEqual(queue.metadata_buffers.bootstrap_room[9], 0)
        self.assertNotIn(
            decode_request.req.bootstrap_room, manager._deferred_abort_ack_tracker
        )

    def test_successful_page_receiver_clears_armed_drain_tracker(self):
        manager, receiver, _request = self._receiver_and_request()
        receiver.conclude_state = KVPoll.Success
        receiver.clear()
        self.assertNotIn(receiver.bootstrap_room, manager._deferred_abort_ack_tracker)

    def test_memory_teardown_rejects_a_page_hold_without_drain(self):
        _manager, _receiver, decode_request = self._receiver_and_request(room=74)
        queue = _bare_decode_queue()
        queue._defer_release(decode_request)

        with self.assertRaisesRegex(RuntimeError, "page DCP writes remain quarantined"):
            queue.release_memory_occupation()
        self.assertEqual(len(queue._deferred_releases), 1)


if __name__ == "__main__":
    unittest.main()
