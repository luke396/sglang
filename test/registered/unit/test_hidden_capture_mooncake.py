"""Unit tests for srt/state_capturer/hidden_mooncake.py.

A fake in-memory store implements the five client calls the sink uses
(register_buffer / unregister_buffer / is_exist / put_from / put), with
put_from reading real bytes out of the caller's staging buffer via ctypes —
so key layout, byte fidelity, ordering, and bounds are all exercised without
a running Mooncake master.
"""

import ctypes
import json
import unittest

import torch

from sglang.srt.state_capturer.hidden_mooncake import (
    MooncakeHiddenSink,
    SampleTooLargeError,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

AUX_WIDTH = 8
LAST_WIDTH = 4
DTYPE = torch.bfloat16
ROW_BYTES = (AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize + 8


class FakeStore:
    """In-memory stand-in for MooncakeDistributedStore's sink-facing API."""

    def __init__(self):
        self.data = {}
        self.registered = []  # (ptr, size) spans
        self.put_order = []

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        self.registered = [(p, s) for p, s in self.registered if p != ptr]
        return 0

    def is_exist(self, key):
        return 1 if key in self.data else 0

    def put_from(self, key, ptr, size, config):
        # The RDMA contract: the source must lie inside a registered span.
        if not any(p <= ptr and ptr + size <= p + s for p, s in self.registered):
            return -1
        self.data[key] = ctypes.string_at(ptr, size)
        self.put_order.append(key)
        return 0

    def put(self, key, value, config):
        self.data[key] = bytes(value)
        self.put_order.append(key)
        return 0


def _make_sink(max_export_tokens=16):
    store = FakeStore()
    sink = MooncakeHiddenSink(
        store_id="test_store",
        row_bytes=ROW_BYTES,
        max_export_tokens=max_export_tokens,
        store=store,
        replicate_config=object(),
    )
    return sink, store


def _record(num_tokens, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.arange(num_tokens, dtype=torch.long),
        "loss_mask": torch.ones(num_tokens, dtype=torch.long),
        "aux_hidden_state": torch.randn(
            1, num_tokens, AUX_WIDTH, generator=g, dtype=torch.float32
        ).to(DTYPE),
        "hidden_state": torch.randn(
            1, num_tokens, LAST_WIDTH, generator=g, dtype=torch.float32
        ).to(DTYPE),
        "rid": "rid-1",
    }


class TestMooncakeSinkKeysAndBytes(CustomTestCase):
    def test_key_layout_matches_specforge_tkey(self):
        """The key shape {store_id}/{sample_id}/g0/{name} is SpecForge
        MooncakeFeatureStore._tkey's contract — the trainer's existing
        zero-copy reader consumes it verbatim. Renaming any segment silently
        orphans every exported sample."""
        sink, store = _make_sink()
        self.assertTrue(sink.put("s1", _record(4)))
        self.assertEqual(
            set(store.data),
            {
                "test_store/s1/g0/aux",
                "test_store/s1/g0/last_hidden",
                "test_store/s1/g0/input_ids",
                "test_store/s1/g0/meta",
            },
        )

    def test_payload_bytes_roundtrip(self):
        sink, store = _make_sink()
        record = _record(4, seed=7)
        sink.put("s1", record)
        aux_back = torch.frombuffer(
            bytearray(store.data["test_store/s1/g0/aux"]), dtype=DTYPE
        ).view(1, 4, AUX_WIDTH)
        self.assertTrue(torch.equal(aux_back, record["aux_hidden_state"]))
        ids_back = torch.frombuffer(
            bytearray(store.data["test_store/s1/g0/input_ids"]), dtype=torch.long
        )
        self.assertTrue(torch.equal(ids_back, record["input_ids"]))

    def test_meta_is_self_describing_and_last(self):
        """Bypass capture has no response channel: shape/dtype must live in
        the meta key, and meta must be written last so a scanner that finds it
        never sees a partial sample."""
        sink, store = _make_sink()
        sink.put("s1", _record(4))
        meta = json.loads(store.data["test_store/s1/g0/meta"])
        self.assertEqual(meta["rid"], "rid-1")
        self.assertEqual(meta["num_tokens"], 4)
        self.assertEqual(meta["tensors"]["aux"]["shape"], [1, 4, AUX_WIDTH])
        self.assertEqual(meta["tensors"]["aux"]["dtype"], "torch.bfloat16")
        self.assertEqual(meta["loss_mask"], "all_ones_placeholder")
        self.assertEqual(store.put_order[-1], "test_store/s1/g0/meta")

    def test_duplicate_sample_id_first_write_wins(self):
        sink, store = _make_sink()
        self.assertTrue(sink.put("s1", _record(4, seed=1)))
        first_aux = store.data["test_store/s1/g0/aux"]
        self.assertFalse(sink.put("s1", _record(4, seed=2)))
        self.assertEqual(store.data["test_store/s1/g0/aux"], first_aux)

    def test_oversized_sample_raises_too_large(self):
        sink, store = _make_sink(max_export_tokens=2)
        with self.assertRaises(SampleTooLargeError):
            sink.put("big", _record(64))
        # Nothing partial published.
        self.assertEqual(store.data, {})

    def test_staging_reuse_across_samples(self):
        """Consecutive samples reuse one registered buffer; the second put
        must not be corrupted by the first's leftovers."""
        sink, store = _make_sink()
        rec_a, rec_b = _record(6, seed=1), _record(3, seed=2)
        sink.put("a", rec_a)
        sink.put("b", rec_b)
        b_back = torch.frombuffer(
            bytearray(store.data["test_store/b/g0/aux"]), dtype=DTYPE
        ).view(1, 3, AUX_WIDTH)
        self.assertTrue(torch.equal(b_back, rec_b["aux_hidden_state"]))

    def test_fingerprint_idempotent(self):
        sink, store = _make_sink()
        sink.write_fingerprint({"model_path": "a"})
        sink.write_fingerprint({"model_path": "b"})
        self.assertEqual(
            json.loads(store.data["test_store/_fingerprint"])["model_path"], "a"
        )

    def test_close_unregisters_staging(self):
        sink, store = _make_sink()
        self.assertEqual(len(store.registered), 1)
        sink.close()
        self.assertEqual(store.registered, [])


if __name__ == "__main__":
    unittest.main()
