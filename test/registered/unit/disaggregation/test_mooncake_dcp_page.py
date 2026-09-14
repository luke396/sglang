"""CPU transport contracts for direct Mooncake DCP page sends."""

import concurrent.futures
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CopyTransport:
    def __init__(self, sources, destinations):
        self.sources = sources
        self.destinations = destinations
        self.bytes_sent = 0

    @staticmethod
    def _region(buffers, address, size):
        for buffer in buffers:
            offset = address - buffer.ctypes.data
            if 0 <= offset and offset + size <= buffer.nbytes:
                return buffer.reshape(-1)[offset : offset + size]
        raise AssertionError("transfer descriptor exceeds a registered CPU buffer")

    def __call__(self, _session, blocks):
        for source, destination, size in blocks:
            self._region(self.destinations, destination, size)[:] = self._region(
                self.sources, source, size
            )
            self.bytes_sent += size
        return 0


class _RunningWrite(concurrent.futures.Future):
    def __init__(self):
        super().__init__()
        self.set_running_or_notify_cancel()
        self.cancel_attempted = threading.Event()

    def cancel(self):
        self.cancel_attempted.set()
        return super().cancel()


class TestMooncakeDcpPage(CustomTestCase):
    page_size = 4
    widths = (3, 5)

    def _case(self, *, custom_pool=False):
        sources = [
            ((np.arange(64 * width) + layer * 41) % 251)
            .astype(np.uint8)
            .reshape(64, width)
            for layer, width in enumerate(self.widths)
        ]
        destinations = [np.full_like(source, 0xEE) for source in sources]
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=self.page_size,
            kv_data_ptrs=[buffer.ctypes.data for buffer in sources],
            kv_layer_ids=[11, 17],
            num_draft_entries=0,
        )
        manager.enable_custom_mem_pool = custom_pool
        manager.enable_deferred_decode_kv_release = False
        transport = _CopyTransport(sources, destinations)
        manager._transfer_data = transport
        return manager, sources, destinations, transport

    def _send(self, manager, destinations, *, executor=None, **kwargs):
        return manager.send_kvcache_dcp_page(
            "session",
            kwargs.pop("source_pages", np.array([7], dtype=np.int32)),
            # Reversed destination registration exercises the existing layer map.
            [buffer.ctypes.data for buffer in reversed(destinations)],
            kwargs.pop("destination_pages", np.array([10], dtype=np.int32)),
            dcp_token_item_lens=list(self.widths),
            dst_dcp_size=3,
            dst_dcp_rank=kwargs.pop("rank", 0),
            src_page_offset=kwargs.pop("offset", 0),
            decode_prefix_len=12,
            num_kv_tokens=kwargs.pop("length", 4),
            executor=executor,
            dst_layer_ids=[17, 11],
        )

    def test_fragmented_chunks_copy_only_owned_rows_and_valid_tail_bytes(self):
        source_pages = np.array([7, 1, 9, 3, 12, 4, 8], dtype=np.int32)
        destination_pages = np.array([10, 2, 14], dtype=np.int32)
        for rank in range(3):
            with self.subTest(rank=rank):
                manager, sources, destinations, transport = self._case()
                with patch(
                    "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
                    side_effect=AssertionError("page send must not pack"),
                ):
                    for offset, count, length in ((0, 2, 8), (2, 3, 12), (5, 2, 5)):
                        self.assertEqual(
                            self._send(
                                manager,
                                destinations,
                                source_pages=source_pages[offset : offset + count],
                                destination_pages=destination_pages,
                                rank=rank,
                                offset=offset,
                                length=length,
                            ),
                            0,
                        )
                        if rank == 2 and offset == 0:
                            self.assertEqual(transport.bytes_sent, 0)

                owned = [i for i in range(25) if ((12 + i) // 4) % 3 == rank]
                for source, destination in zip(sources, destinations):
                    expected = np.full_like(destination, 0xEE)
                    for local, logical in enumerate(owned):
                        source_row = source_pages[logical // 4] * 4 + logical % 4
                        destination_row = destination_pages[local // 4] * 4 + local % 4
                        expected[destination_row] = source[source_row]
                    np.testing.assert_array_equal(destination, expected)
                self.assertEqual(transport.bytes_sent, len(owned) * sum(self.widths))

    def test_layer_failure_waits_for_an_uncancellable_late_write(self):
        for error in (73, RuntimeError("transport failed")):
            with self.subTest(error=error):
                manager, _, destinations, transport = self._case(custom_pool=True)
                failed = concurrent.futures.Future()
                if isinstance(error, Exception):
                    failed.set_exception(error)
                else:
                    failed.set_result(error)
                running = _RunningWrite()
                submitted = []

                def submit(function, *args):
                    submitted.append((function, args))
                    return failed if len(submitted) == 1 else running

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
                    result = worker.submit(
                        self._send,
                        manager,
                        destinations,
                        executor=SimpleNamespace(submit=submit),
                    )
                    try:
                        self.assertTrue(running.cancel_attempted.wait(timeout=2))
                        self.assertFalse(result.done())
                        function, args = submitted[1]
                        self.assertEqual(function(*args), 0)
                        self.assertEqual(transport.bytes_sent, 4 * self.widths[1])
                    finally:
                        running.set_result(0)
                    self.assertNotEqual(result.result(timeout=2), 0)

    def test_partial_submit_failure_drains_the_submitted_prefix(self):
        manager, _, destinations, transport = self._case(custom_pool=True)
        running = _RunningWrite()
        submit_failed = threading.Event()
        submitted = []

        def submit(function, *args):
            if submitted:
                submit_failed.set()
                raise RuntimeError("executor rejected the next layer")
            submitted.append((function, args))
            return running

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(
                self._send,
                manager,
                destinations,
                executor=SimpleNamespace(submit=submit),
            )
            try:
                self.assertTrue(submit_failed.wait(timeout=2))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    result.result(timeout=0.05)
                function, args = submitted[0]
                self.assertEqual(function(*args), 0)
                self.assertEqual(transport.bytes_sent, 4 * self.widths[0])
            finally:
                running.set_result(0)
            with self.assertRaisesRegex(RuntimeError, "executor rejected"):
                result.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
