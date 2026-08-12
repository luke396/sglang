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
    """In-memory stand-in for MooncakeDistributedStore's sink-facing API.

    ``put``/``put_from`` are first-write-wins (rc=0, old value kept), matching
    the live master's verified behavior — the manifest sequence logic depends
    on it.
    """

    def __init__(self):
        self.data = {}
        self.registered = []  # (ptr, size) spans
        self.put_order = []
        self.put_configs = {}  # key -> the config object the put used

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
        if key in self.data:
            return 0
        self.data[key] = ctypes.string_at(ptr, size)
        self.put_order.append(key)
        self.put_configs[key] = config
        return 0

    def put(self, key, value, config):
        if key in self.data:
            return 0
        self.data[key] = bytes(value)
        self.put_order.append(key)
        self.put_configs[key] = config
        return 0


def _make_sink(max_export_tokens=16, store=None, stats=None, dp_rank=0):
    store = store if store is not None else FakeStore()
    sink = MooncakeHiddenSink(
        store_id="test_store",
        row_bytes=ROW_BYTES,
        max_export_tokens=max_export_tokens,
        dp_rank=dp_rank,
        stats=stats,
        store=store,
        replicate_config=object(),
        manifest_config=object(),
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
        orphans every exported sample. The manifest entry _seq/{dp_rank}/{n}
        is the SpecLoop discovery contract on top."""
        sink, store = _make_sink()
        self.assertTrue(sink.put("s1", _record(4)))
        self.assertEqual(
            set(store.data),
            {
                "test_store/s1/g0/aux",
                "test_store/s1/g0/last_hidden",
                "test_store/s1/g0/input_ids",
                "test_store/s1/g0/meta",
                "test_store/_seq/0/0",
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
        the meta key, and meta must be written after every tensor so a
        scanner that finds it never sees a partial sample. The manifest entry
        comes after meta — a sample must never be discoverable before it is
        complete."""
        sink, store = _make_sink()
        sink.put("s1", _record(4))
        meta = json.loads(store.data["test_store/s1/g0/meta"])
        self.assertEqual(meta["rid"], "rid-1")
        self.assertEqual(meta["num_tokens"], 4)
        self.assertEqual(meta["tensors"]["aux"]["shape"], [1, 4, AUX_WIDTH])
        self.assertEqual(meta["tensors"]["aux"]["dtype"], "torch.bfloat16")
        self.assertEqual(meta["loss_mask"], "all_ones_placeholder")
        self.assertEqual(
            store.put_order[-2:], ["test_store/s1/g0/meta", "test_store/_seq/0/0"]
        )

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


class _StatsStub:
    def __init__(self):
        self.bumped = []

    def bump(self, name, delta=1):
        self.bumped.append(name)


class TestManifest(CustomTestCase):
    """SpecLoop discovery contract: {store_id}/_seq/{dp_rank}/{n}.

    Consumers tail these keys knowing nothing but the store prefix; every
    property here (per-sample entry, monotone numbering, restart resume
    without overwrites, burn-on-failure) is load-bearing for that tail loop.
    """

    def test_one_entry_per_sample_monotone_and_bare_sample_id(self):
        sink, store = _make_sink(dp_rank=3)
        sink.put("alpha", _record(4, seed=1))
        sink.put("beta", _record(4, seed=2))
        self.assertEqual(store.data["test_store/_seq/3/0"], b"alpha")
        self.assertEqual(store.data["test_store/_seq/3/1"], b"beta")
        self.assertNotIn("test_store/_seq/3/2", store.data)

    def test_manifest_only_after_meta_and_none_for_rejected_puts(self):
        """A sample must never be discoverable unless complete: duplicate ids
        and oversized samples publish no manifest entry, and the entry for a
        good sample follows its meta (see put_order)."""
        sink, store = _make_sink(max_export_tokens=4)
        self.assertTrue(sink.put("s1", _record(4)))
        self.assertFalse(sink.put("s1", _record(4, seed=2)))  # duplicate
        with self.assertRaises(SampleTooLargeError):
            sink.put("big", _record(64))
        manifest_keys = [k for k in store.data if "/_seq/" in k]
        self.assertEqual(manifest_keys, ["test_store/_seq/0/0"])

    def test_manifest_uses_pinned_config_not_payload_config(self):
        """Manifests are hard-pinned, payloads soft-pinned; putting a manifest
        with the payload config silently reintroduces the orphan-by-eviction
        failure the pin exists to prevent."""
        payload_cfg, manifest_cfg = object(), object()
        store = FakeStore()
        sink = MooncakeHiddenSink(
            store_id="test_store",
            row_bytes=ROW_BYTES,
            max_export_tokens=16,
            store=store,
            replicate_config=payload_cfg,
            manifest_config=manifest_cfg,
        )
        sink.put("s1", _record(4))
        self.assertIs(store.put_configs["test_store/_seq/0/0"], manifest_cfg)
        self.assertIs(store.put_configs["test_store/s1/g0/meta"], payload_cfg)

    def test_restart_resumes_after_tail_without_overwrite(self):
        sink, store = _make_sink()
        for i in range(5):
            sink.put(f"s{i}", _record(4, seed=i))
        reborn, _ = _make_sink(store=store)
        reborn.put("after-restart", _record(4, seed=9))
        self.assertEqual(store.data["test_store/_seq/0/5"], b"after-restart")
        # No pre-restart entry was overwritten.
        self.assertEqual(store.data["test_store/_seq/0/0"], b"s0")

    def test_restart_tail_skips_holes(self):
        """Burnt numbers (failed puts) and degraded-mode evictions leave
        holes; the binary search alone would land in the first hole and the
        next put would collide with a surviving entry — first-write-wins
        turns that collision into silent entry loss. The tail scan must
        resume after the LAST survivor."""
        sink, store = _make_sink()
        for i in range(6):
            sink.put(f"s{i}", _record(4, seed=i))
        del store.data["test_store/_seq/0/2"]  # simulate eviction/burn
        del store.data["test_store/_seq/0/3"]
        reborn, _ = _make_sink(store=store)
        reborn.put("after-restart", _record(4, seed=9))
        self.assertEqual(store.data["test_store/_seq/0/6"], b"after-restart")
        self.assertEqual(store.data["test_store/_seq/0/5"], b"s5")

    def test_manifest_failure_is_counted_orphan_not_export_failure(self):
        """Requirement: manifest put failure = no retry, orphan counter +1,
        payload/meta NOT rolled back, and put() still reports success so the
        export worker doesn't double-count a sink failure. The failed attempt
        burns its sequence number (ambiguous failures must not retry into a
        first-write-wins key)."""

        class FailingManifestStore(FakeStore):
            def put(self, key, value, config):
                if "/_seq/" in key and key.endswith("/1"):
                    return -99
                return super().put(key, value, config)

        stats = _StatsStub()
        store = FailingManifestStore()
        sink, _ = _make_sink(store=store, stats=stats)
        self.assertTrue(sink.put("ok", _record(4, seed=1)))
        self.assertTrue(sink.put("orphaned", _record(4, seed=2)))  # manifest fails
        self.assertTrue(sink.put("next", _record(4, seed=3)))
        self.assertEqual(stats.bumped, ["manifest_orphan_miss_ct"])
        # Payload survives the manifest failure.
        self.assertIn("test_store/orphaned/g0/meta", store.data)
        # Number 1 is burnt; the next sample takes 2.
        self.assertNotIn("test_store/_seq/0/1", store.data)
        self.assertEqual(store.data["test_store/_seq/0/2"], b"next")

    def test_build_manifest_config_pin_degradation(self):
        from sglang.srt.state_capturer.hidden_mooncake import _build_manifest_config

        class PinnedCfg:
            replica_num = 0
            with_hard_pin = False

        class LegacyCfg:
            replica_num = 0

        pinned = _build_manifest_config(PinnedCfg)
        self.assertTrue(pinned.with_hard_pin)
        with self.assertLogs(
            "sglang.srt.state_capturer.hidden_mooncake", level="WARNING"
        ) as caught:
            _build_manifest_config(LegacyCfg)
        self.assertIn("gap detection", caught.output[0])


if __name__ == "__main__":
    unittest.main()
