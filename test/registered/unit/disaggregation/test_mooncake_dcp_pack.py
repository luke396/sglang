import concurrent.futures
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.common.dcp_pack import dcp_pack_buffer_bytes
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _GatherStream:
    def wait_stream(self, _stream):
        pass

    def synchronize(self):
        pass


class _CpuPackBuffer:
    def __init__(self, size_bytes):
        self.buffer = torch.zeros(size_bytes, dtype=torch.uint8)
        self._stream = _GatherStream()

    def fits(self, required_bytes):
        return required_bytes <= self.buffer.numel()

    def get_ptr(self):
        return self.buffer.data_ptr()

    def get_size(self):
        return self.buffer.numel()

    def get_gather_stream(self):
        return self._stream


class _PackedTransport:
    """CPU transport that accepts only registered packed-buffer reads."""

    def __init__(self, pack_buffer, dst_buffers, *, fail_call=None):
        self.pack_buffer = pack_buffer
        self.dst_buffers = dst_buffers
        self.fail_call = fail_call
        self.calls = []
        self.bytes_sent = 0

    def _buffer_slice(self, buffers, address, size):
        for buffer in buffers:
            start = buffer.data_ptr()
            offset = address - start
            if 0 <= offset and offset + size <= buffer.numel():
                return buffer.reshape(-1).narrow(0, offset, size)
        raise AssertionError(f"address 0x{address:x} is outside registered buffers")

    def __call__(self, _session_id, blocks):
        self.calls.append(list(blocks))
        if self.fail_call == len(self.calls):
            return 73

        pack_start = self.pack_buffer.get_ptr()
        pack_end = pack_start + self.pack_buffer.get_size()
        for src_addr, dst_addr, length in blocks:
            # A correct result through the old per-token fallback would still
            # fail here: the source must be the reusable pack region.
            if not (pack_start <= src_addr and src_addr + length <= pack_end):
                raise AssertionError("Mooncake DCP transfer bypassed the pack buffer")
            src = self.pack_buffer.buffer.narrow(0, src_addr - pack_start, length)
            self._buffer_slice(self.dst_buffers, dst_addr, length).copy_(src)
            self.bytes_sent += length
        return 0


class TestMooncakeDcpPack(CustomTestCase):
    page_size = 4
    widths = [3, 5]

    def _make_case(self, *, pack_tokens=2):
        # Physical page ids are deliberately fragmented. Each byte encodes its
        # layer and physical token, so the expected DCP mapping is calculated
        # below without consulting the production plan implementation.
        src_pages = np.array([7, 1, 9, 3, 12], dtype=np.int32)
        dst_pages = np.array([15, 5, 11], dtype=np.int32)
        src_slots = (int(src_pages.max()) + 1) * self.page_size
        dst_slots = (int(dst_pages.max()) + 1) * self.page_size
        src = [
            ((layer * 41 + torch.arange(src_slots * width)) % 251)
            .to(torch.uint8)
            .view(src_slots, width)
            for layer, width in enumerate(self.widths)
        ]
        dst = [
            torch.full((dst_slots, width), 0xEE, dtype=torch.uint8)
            for width in self.widths
        ]
        self._src, self._dst = src, dst
        pack = _CpuPackBuffer(pack_tokens * sum(self.widths))

        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            page_size=self.page_size,
            kv_data_ptrs=[buffer.data_ptr() for buffer in src],
            kv_layer_ids=[100, 101],
        )
        manager.enable_custom_mem_pool = False
        manager.enable_deferred_decode_kv_release = False
        manager.bootstrap_port = 0
        return manager, src_pages, dst_pages, pack

    def _patch_gather(self, src):
        ptr_to_tensor = {buffer.data_ptr(): buffer for buffer in src}

        def fake_gather(kv_ptrs, rows, pack, token_item_lens):
            offset = 0
            row_ids = rows.cpu().numpy().tolist()
            for ptr, width in zip(kv_ptrs, token_item_lens):
                values = ptr_to_tensor[ptr][row_ids].reshape(-1)
                pack.narrow(0, offset, values.numel()).copy_(values)
                offset += values.numel()

        return patch(
            "sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack",
            side_effect=fake_gather,
        )

    def _send(
        self,
        manager,
        src_pages,
        dst_pages,
        pack,
        *,
        num_kv_tokens,
        transport,
        executor=None,
        dcp_size=3,
        dcp_rank=1,
        decode_prefix_len=12,
    ):
        manager._transfer_data = transport
        with (
            self._patch_gather(self._src),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.default_stream"
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.stream",
                return_value=nullcontext(),
            ),
        ):
            return manager.send_kvcache_dcp(
                "session",
                src_pages,
                [buffer.data_ptr() for buffer in self._dst],
                dst_pages,
                dcp_token_item_lens=self.widths,
                dst_dcp_size=dcp_size,
                dst_dcp_rank=dcp_rank,
                src_page_offset=1,
                decode_prefix_len=decode_prefix_len,
                num_kv_tokens=num_kv_tokens,
                executor=executor,
                dst_layer_ids=[100, 101],
                pack_buffer=pack,
            )

    def _expected_owned_rows(
        self,
        src_pages,
        dst_pages,
        num_kv_tokens,
        *,
        dcp_size=3,
        dcp_rank=1,
        decode_prefix_len=12,
    ):
        # This is the DCP ownership calculation, written independently of
        # build_dcp_token_transfer_plan. The offset/prefix combination makes
        # the first source row and first decode-local row nonzero.
        chunk_start = decode_prefix_len + self.page_size
        pairs = []
        for logical_offset in range(num_kv_tokens):
            if (chunk_start + logical_offset) % dcp_size != dcp_rank:
                continue
            src_token = (
                int(src_pages[logical_offset // self.page_size]) * self.page_size
                + logical_offset % self.page_size
            )
            relative = self.page_size + logical_offset
            dst_local = relative // dcp_size
            dst_token = (
                int(dst_pages[dst_local // self.page_size]) * self.page_size
                + dst_local % self.page_size
            )
            pairs.append((src_token, dst_token))
        return pairs

    def _assert_destination(
        self,
        src_pages,
        dst_pages,
        num_kv_tokens,
        *,
        dcp_size=3,
        dcp_rank=1,
        decode_prefix_len=12,
    ):
        expected = self._expected_owned_rows(
            src_pages,
            dst_pages,
            num_kv_tokens,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            decode_prefix_len=decode_prefix_len,
        )
        for layer, (src, dst) in enumerate(zip(self._src, self._dst)):
            expected_dst = torch.full_like(dst, 0xEE)
            for src_token, dst_token in expected:
                expected_dst[dst_token] = src[src_token]
            torch.testing.assert_close(
                actual=dst,
                expected=expected_dst,
                rtol=0,
                atol=0,
                msg=lambda msg: f"layer={layer}: {msg}",
            )

    def test_boundaries_fragmented_pages_offsets_and_layer_widths(self):
        """Oversized intervals must stay packed without losing or misplacing KV."""
        # DCP rank 1 owns 0, 1, 2, 3, and 7 rows for these source lengths:
        # empty, one row, exactly K rows, K+1 rows, then several full slices
        # plus a tail (K=2). Every transfer uses the real DCP send entry and
        # try_pack_dcp_src; only gather and transport are CPU fakes.
        for num_kv_tokens, expected_rows in [(0, 0), (1, 1), (4, 2), (7, 3), (19, 7)]:
            with self.subTest(num_kv_tokens=num_kv_tokens):
                manager, src_pages, dst_pages, pack = self._make_case()
                transport = _PackedTransport(pack, self._dst)
                ret = self._send(
                    manager,
                    src_pages,
                    dst_pages,
                    pack,
                    num_kv_tokens=num_kv_tokens,
                    transport=transport,
                )

                self.assertEqual(ret, 0)
                self._assert_destination(src_pages, dst_pages, num_kv_tokens)
                self.assertEqual(transport.bytes_sent, expected_rows * sum(self.widths))
                self.assertEqual(len(transport.calls), (expected_rows + 1) // 2)

    def test_existing_capacity_only_splits_oversized_transfers(self):
        # Keep the production allocation formula. The DCP3 cases cover a
        # ready interval larger than C that fits, then one that exceeds it.
        cases = [(16, 8, 3, 32, 16, 1), (4, 3, 1, 12, 16, 1), (4, 3, 1, 12, 19, 2)]
        for ceiling, dcp_size, rank, prefix, tokens, calls in cases:
            with self.subTest(ceiling=ceiling, dcp_size=dcp_size, tokens=tokens):
                size = dcp_pack_buffer_bytes(
                    [width * self.page_size for width in self.widths],
                    self.page_size,
                    ceiling,
                    dcp_size,
                )
                manager, src_pages, dst_pages, pack = self._make_case(
                    pack_tokens=size // sum(self.widths)
                )
                transport = _PackedTransport(pack, self._dst)
                ret = self._send(
                    manager,
                    src_pages,
                    dst_pages,
                    pack,
                    num_kv_tokens=tokens,
                    transport=transport,
                    dcp_size=dcp_size,
                    dcp_rank=rank,
                    decode_prefix_len=prefix,
                )
                self.assertEqual(ret, 0)
                self._assert_destination(
                    src_pages,
                    dst_pages,
                    tokens,
                    dcp_size=dcp_size,
                    dcp_rank=rank,
                    decode_prefix_len=prefix,
                )
                self.assertEqual(len(transport.calls), calls)

    def test_pack_too_small_for_one_token_rejects_before_transfer(self):
        manager, src_pages, dst_pages, pack = self._make_case(pack_tokens=0)
        transport = _PackedTransport(pack, self._dst)

        with self.assertRaisesRegex(AssertionError, "Mooncake DCP slice must fit"):
            self._send(
                manager,
                src_pages,
                dst_pages,
                pack,
                num_kv_tokens=19,
                transport=transport,
            )

        self.assertEqual(transport.calls, [])

    def test_failure_in_middle_slice_stops_later_packed_submissions(self):
        manager, src_pages, dst_pages, pack = self._make_case()
        transport = _PackedTransport(pack, self._dst, fail_call=2)

        ret = self._send(
            manager,
            src_pages,
            dst_pages,
            pack,
            num_kv_tokens=19,
            transport=transport,
        )

        self.assertEqual(ret, 73)
        self.assertEqual(len(transport.calls), 2)

    def test_custom_pool_waits_before_packing_next_slice(self):
        manager, src_pages, dst_pages, pack = self._make_case()
        manager.enable_custom_mem_pool = True
        transport = _PackedTransport(pack, self._dst)
        reader_started = threading.Event()
        release_reader = threading.Event()

        def delayed_transfer(session_id, blocks):
            if blocks[0][0] == pack.get_ptr() and not reader_started.is_set():
                snapshot = pack.buffer.clone()
                reader_started.set()
                if not release_reader.wait(timeout=5):
                    raise TimeoutError("test reader was never released")
                self.assertTrue(torch.equal(snapshot, pack.buffer))
            return transport(session_id, blocks)

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as sender,
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as layers,
        ):
            result = sender.submit(
                self._send,
                manager,
                src_pages,
                dst_pages,
                pack,
                num_kv_tokens=7,
                transport=delayed_transfer,
                executor=layers,
            )
            try:
                self.assertTrue(reader_started.wait(timeout=2))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    result.result(timeout=0.1)
            finally:
                release_reader.set()
            self.assertEqual(result.result(timeout=3), 0)
        self._assert_destination(src_pages, dst_pages, 7)
        self.assertEqual(len(transport.calls), 4)


if __name__ == "__main__":
    unittest.main()
