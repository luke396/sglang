"""Unit tests for srt/state_capturer/hidden_mooncake.py.

A fake in-memory store implements the five client calls the sink uses
(register_buffer / unregister_buffer / is_exist / put_from / put), with
put_from reading real bytes out of the caller's staging buffer via ctypes —
so key layout, byte fidelity, ordering, and bounds are all exercised without
a running Mooncake master.
"""

import ctypes
import json
import threading
import time
import unittest
from unittest import mock

import torch

from sglang.srt.state_capturer.hidden_host import (
    HiddenCaptureStats,
    HiddenHostSidecar,
)
from sglang.srt.state_capturer.hidden_mooncake import (
    AsyncMooncakeHiddenSink,
    MooncakeHiddenSink,
    SampleTooLargeError,
    _connect,
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
        self.batch_put_calls = []

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        self.registered = [(p, s) for p, s in self.registered if p != ptr]
        return 0

    def is_exist(self, key):
        return 1 if key in self.data else 0

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]

    def get(self, key):
        return self.data.get(key)

    def batch_get_into(self, keys, ptrs, sizes):
        statuses = []
        for key, ptr, size in zip(keys, ptrs, sizes):
            value = self.data.get(key)
            if value is None or len(value) != size:
                statuses.append(-1)
                continue
            if not any(
                base <= ptr and ptr + size <= base + span
                for base, span in self.registered
            ):
                statuses.append(-2)
                continue
            ctypes.memmove(ptr, value, size)
            statuses.append(size)
        return statuses

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

    def batch_put_from(self, keys, ptrs, sizes, config):
        self.batch_put_calls.append(
            {
                "keys": list(keys),
                "ptrs": list(ptrs),
                "sizes": list(sizes),
                "config": config,
            }
        )
        return [
            self.put_from(key, ptr, size, config)
            for key, ptr, size in zip(keys, ptrs, sizes)
        ]

    def put(self, key, value, config):
        if key in self.data:
            return 0
        self.data[key] = bytes(value)
        self.put_order.append(key)
        self.put_configs[key] = config
        return 0


class FakeReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.group_ids = []
        self.with_hard_pin = False


def _make_sink(max_export_tokens=16, store=None, stats=None, dp_rank=0):
    """Minimal prefix sink + sidecar for manifest/lifecycle tests."""
    store = store if store is not None else FakeStore()
    stats_obj = stats if stats is not None else HiddenCaptureStats()
    sink = MooncakeHiddenSink(
        store_id="test_store",
        max_export_tokens=max_export_tokens,
        dp_rank=dp_rank,
        stats=stats,
        store=store,
        replicate_config=FakeReplicateConfig(),
        manifest_config=FakeReplicateConfig(),
        replicate_config_cls=FakeReplicateConfig,
        max_segment_rows=4,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
    )
    sink.write_fingerprint({"model_path": "test-model"})
    return sink, store


def _make_prefix_sink(
    *,
    max_export_tokens=32,
    max_segment_rows=4,
    prefix_lanes=2,
    store=None,
    stats=None,
    writer_epoch="epoch-1",
):
    store = store if store is not None else FakeStore()
    stats = stats if stats is not None else HiddenCaptureStats()
    sink = MooncakeHiddenSink(
        store_id="test_store",
        max_export_tokens=max_export_tokens,
        stats=stats,
        store=store,
        replicate_config=FakeReplicateConfig(),
        manifest_config=FakeReplicateConfig(),
        replicate_config_cls=FakeReplicateConfig,
        max_segment_rows=max_segment_rows,
        prefix_lanes=prefix_lanes,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
        writer_epoch=writer_epoch,
    )
    sink.write_fingerprint({"model_path": "test-model", "revision": "r1"})
    return sink, store, stats


def _make_sidecar(num_slots=64, stats=None):
    return HiddenHostSidecar(
        num_slots=num_slots,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
        stats=stats,
    )


def _write_rows(sidecar, slots, tokens, *, seed):
    g = torch.Generator().manual_seed(seed)
    aux = torch.randn(len(slots), AUX_WIDTH, generator=g, dtype=torch.float32).to(DTYPE)
    last = torch.randn(len(slots), LAST_WIDTH, generator=g, dtype=torch.float32).to(
        DTYPE
    )
    slot_tensor = torch.tensor(slots, dtype=torch.long)
    gens = sidecar.write_rows(
        slots=slot_tensor,
        aux_rows=aux,
        last_rows=last,
        tokens=torch.tensor(tokens, dtype=torch.long),
    )
    own = dict(zip(slots, gens.tolist()))
    return aux, last, own


def _put_prefix(sink, sidecar, sample_id, tokens, slots, own, prompt_len=0):
    return sink.put_prefix_sample(
        sample_id=sample_id,
        rid=f"rid-{sample_id}",
        tokens=torch.tensor(tokens, dtype=torch.long),
        slots=torch.tensor(slots, dtype=torch.long),
        prompt_len=prompt_len,
        own_slot_gens=own,
        sidecar=sidecar,
    )


def _read_prefix_sample(store, sample_id):
    meta = json.loads(store.data[f"test_store/_samples/{sample_id}/meta"])
    writer_epoch = meta["writer_epoch"]
    aux_parts = []
    last_parts = []
    for ref in meta["segments"]:
        base = f"test_store/_segments/{writer_epoch}/{ref['segment_id']}"
        rows = ref["rows"]
        aux_parts.append(
            torch.frombuffer(bytearray(store.data[f"{base}/aux"]), dtype=DTYPE)
            .view(rows, AUX_WIDTH)
            .clone()
        )
        last_parts.append(
            torch.frombuffer(bytearray(store.data[f"{base}/last_hidden"]), dtype=DTYPE)
            .view(rows, LAST_WIDTH)
            .clone()
        )
    ids = torch.frombuffer(
        bytearray(store.data[f"test_store/_samples/{sample_id}/input_ids"]),
        dtype=torch.long,
    ).clone()
    return meta, ids, torch.cat(aux_parts), torch.cat(last_parts)


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
    def test_failed_owned_store_setup_closes_client(self):
        class FailedStore:
            def __init__(self):
                self.closed = False

            def setup(self, *args):
                return -9

            def close(self):
                self.closed = True

        store = FailedStore()
        with mock.patch("mooncake.store.MooncakeDistributedStore", return_value=store):
            with self.assertRaisesRegex(RuntimeError, "status -9"):
                _connect("127.0.0.1:1")
        self.assertTrue(store.closed)

    def test_fingerprint_idempotent_but_mismatch_fails_closed(self):
        sink, store = _make_sink()  # helper already wrote {"model_path": "test-model"}
        sink.write_fingerprint({"model_path": "test-model"})  # idempotent re-put
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            sink.write_fingerprint({"model_path": "b"})
        self.assertEqual(
            json.loads(store.data["test_store/_fingerprint"])["model_path"],
            "test-model",
        )

    def test_close_unregisters_staging(self):
        sink, store = _make_sink()
        # input-id staging + boundary staging + 2 prefix lanes
        self.assertEqual(len(store.registered), 4)
        before = sink.state_snapshot()
        self.assertFalse(before["closed"])
        self.assertGreater(before["registered_bytes"], 0)
        self.assertEqual(
            before["registered_bytes"], sum(s for _, s in store.registered)
        )
        sink.close()
        self.assertEqual(store.registered, [])
        after = sink.state_snapshot()
        self.assertTrue(after["closed"])
        self.assertFalse(after["writer_alive"])

    def test_close_unregisters_then_closes_owned_store(self):
        class OwnedStore(FakeStore):
            def __init__(self):
                super().__init__()
                self.lifecycle = []

            def unregister_buffer(self, ptr):
                self.lifecycle.append("unregister")
                return super().unregister_buffer(ptr)

            def close(self):
                self.lifecycle.append("close")

        store = OwnedStore()
        with mock.patch(
            "sglang.srt.state_capturer.hidden_mooncake._connect",
            return_value=(store, object(), object(), FakeReplicateConfig),
        ):
            sink = MooncakeHiddenSink(
                store_id="owned",
                max_export_tokens=4,
                max_segment_rows=4,
                aux_width=AUX_WIDTH,
                last_width=LAST_WIDTH,
                dtype=DTYPE,
            )
        sink.close()
        # 4 arenas (staging + boundary + 2 lanes) unregister before close.
        self.assertEqual(store.lifecycle, ["unregister"] * 4 + ["close"])
        sink.close()
        self.assertEqual(store.lifecycle, ["unregister"] * 4 + ["close"])


class TestMooncakePrefixProtocol(CustomTestCase):
    def test_cold_sample_segments_roundtrip_and_commit_order(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        tokens = [10, 11, 12, 13, 14, 15]
        slots = list(range(len(tokens)))
        aux, last, own = _write_rows(sidecar, slots, tokens, seed=1)

        self.assertTrue(
            _put_prefix(sink, sidecar, "cold", tokens, slots, own, prompt_len=4)
        )
        meta, ids, aux_back, last_back = _read_prefix_sample(store, "cold")
        self.assertEqual(meta["schema"], "hidden-sample-view-v1")
        self.assertEqual(meta["prompt_len"], 4)
        self.assertEqual(
            [(ref["sample_offset"], ref["rows"]) for ref in meta["segments"]],
            [(0, 4), (4, 2)],
        )
        self.assertTrue(torch.equal(ids, torch.tensor(tokens)))
        self.assertTrue(torch.equal(aux_back, aux))
        self.assertTrue(torch.equal(last_back, last))

        for ref in meta["segments"]:
            base = f"test_store/_segments/epoch-1/{ref['segment_id']}"
            aux_key = f"{base}/aux"
            last_key = f"{base}/last_hidden"
            meta_key = f"{base}/meta"
            self.assertLess(
                store.put_order.index(aux_key), store.put_order.index(meta_key)
            )
            self.assertLess(
                store.put_order.index(last_key), store.put_order.index(meta_key)
            )
            group_ids = {
                store.put_configs[key].group_ids[0]
                for key in (aux_key, last_key, meta_key)
            }
            self.assertEqual(len(group_ids), 1)
        sample_ids_key = "test_store/_samples/cold/input_ids"
        sample_meta_key = "test_store/_samples/cold/meta"
        manifest_key = "test_store/_seq/0/0"
        self.assertLess(
            store.put_order.index(sample_ids_key),
            store.put_order.index(sample_meta_key),
        )
        self.assertLess(
            store.put_order.index(sample_meta_key), store.put_order.index(manifest_key)
        )
        self.assertEqual(stats.sidecar_direct_gather_rows_ct, len(tokens))

    def test_a_to_ab_reuses_whole_segment_and_gathers_only_suffix(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        a_tokens = [1, 2, 3, 4]
        a_aux, a_last, a_own = _write_rows(sidecar, [0, 1, 2, 3], a_tokens, seed=2)
        self.assertTrue(_put_prefix(sink, sidecar, "a", a_tokens, [0, 1, 2, 3], a_own))
        a_meta = json.loads(store.data["test_store/_samples/a/meta"])

        suffix_aux, suffix_last, suffix_own = _write_rows(
            sidecar, [10, 11], [5, 6], seed=3
        )
        self.assertTrue(
            _put_prefix(
                sink,
                sidecar,
                "ab",
                a_tokens + [5, 6],
                [0, 1, 2, 3, 10, 11],
                suffix_own,
            )
        )
        ab_meta, _, aux_back, last_back = _read_prefix_sample(store, "ab")
        self.assertEqual(
            ab_meta["segments"][0]["segment_id"],
            a_meta["segments"][0]["segment_id"],
        )
        self.assertTrue(torch.equal(aux_back, torch.cat((a_aux, suffix_aux))))
        self.assertTrue(torch.equal(last_back, torch.cat((a_last, suffix_last))))
        self.assertEqual(stats.prefix_reused_rows_ct, 4)
        self.assertEqual(stats.sidecar_direct_gather_rows_ct, 6)

    def test_fully_warm_sample_reuses_without_sidecar_hidden_read(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        tokens = [1, 2, 3, 4]
        slots = [0, 1, 2, 3]
        _, _, own = _write_rows(sidecar, slots, tokens, seed=20)
        self.assertTrue(_put_prefix(sink, sidecar, "cold", tokens, slots, own))
        gathered_after_cold = stats.sidecar_direct_gather_rows_ct

        self.assertTrue(_put_prefix(sink, sidecar, "warm", tokens, slots, {}))
        cold_meta = json.loads(store.data["test_store/_samples/cold/meta"])
        warm_meta = json.loads(store.data["test_store/_samples/warm/meta"])
        self.assertEqual(cold_meta["segments"], warm_meta["segments"])
        self.assertEqual(stats.sidecar_direct_gather_rows_ct, gathered_after_cold)
        self.assertEqual(stats.prefix_reused_rows_ct, len(tokens))

    def test_a_to_ab_and_ac_share_ancestor_after_a_sample_eviction(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        a_tokens = [1, 2, 3, 4]
        a_aux, a_last, a_own = _write_rows(sidecar, [0, 1, 2, 3], a_tokens, seed=21)
        self.assertTrue(_put_prefix(sink, sidecar, "a", a_tokens, [0, 1, 2, 3], a_own))
        a_meta = json.loads(store.data["test_store/_samples/a/meta"])
        ancestor_id = a_meta["segments"][0]["segment_id"]

        suffix_b_aux, suffix_b_last, own_b = _write_rows(sidecar, [10], [5], seed=22)
        suffix_c_aux, suffix_c_last, own_c = _write_rows(sidecar, [11], [6], seed=23)
        self.assertTrue(
            _put_prefix(
                sink,
                sidecar,
                "ab",
                a_tokens + [5],
                [0, 1, 2, 3, 10],
                own_b,
            )
        )
        self.assertTrue(
            _put_prefix(
                sink,
                sidecar,
                "ac",
                a_tokens + [6],
                [0, 1, 2, 3, 11],
                own_c,
            )
        )
        del store.data["test_store/_samples/a/input_ids"]
        del store.data["test_store/_samples/a/meta"]

        ab_meta, _, ab_aux, ab_last = _read_prefix_sample(store, "ab")
        ac_meta, _, ac_aux, ac_last = _read_prefix_sample(store, "ac")
        self.assertEqual(ab_meta["segments"][0]["segment_id"], ancestor_id)
        self.assertEqual(ac_meta["segments"][0]["segment_id"], ancestor_id)
        self.assertTrue(torch.equal(ab_aux, torch.cat((a_aux, suffix_b_aux))))
        self.assertTrue(torch.equal(ab_last, torch.cat((a_last, suffix_b_last))))
        self.assertTrue(torch.equal(ac_aux, torch.cat((a_aux, suffix_c_aux))))
        self.assertTrue(torch.equal(ac_last, torch.cat((a_last, suffix_c_last))))

    def test_stale_component_is_rebuilt_only_from_exact_own_rows(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        tokens = [1, 2, 3, 4]
        slots = [0, 1, 2, 3]
        aux, last, own = _write_rows(sidecar, slots, tokens, seed=24)
        self.assertTrue(_put_prefix(sink, sidecar, "first", tokens, slots, own))
        meta = json.loads(store.data["test_store/_samples/first/meta"])
        segment_id = meta["segments"][0]["segment_id"]
        last_key = f"test_store/_segments/epoch-1/{segment_id}/last_hidden"
        del store.data[last_key]

        # Simulate a cold recomputation of the same token path. Its exact
        # generations are trusted even though the writer-local index is stale.
        aux2, last2, own2 = _write_rows(sidecar, slots, tokens, seed=24)
        self.assertTrue(_put_prefix(sink, sidecar, "rebuilt", tokens, slots, own2))
        rebuilt_meta, _, aux_back, last_back = _read_prefix_sample(store, "rebuilt")
        self.assertEqual(rebuilt_meta["segments"][0]["segment_id"], segment_id)
        self.assertTrue(torch.equal(aux_back, aux2))
        self.assertTrue(torch.equal(last_back, last2))
        self.assertTrue(torch.equal(aux, aux2))
        self.assertTrue(torch.equal(last, last2))

    def test_missing_meta_is_repaired_without_rewriting_payload(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        tokens = [1, 2, 3, 4]
        slots = [0, 1, 2, 3]
        _, _, own = _write_rows(sidecar, slots, tokens, seed=25)
        self.assertTrue(_put_prefix(sink, sidecar, "first", tokens, slots, own))
        meta = json.loads(store.data["test_store/_samples/first/meta"])
        segment_id = meta["segments"][0]["segment_id"]
        base = f"test_store/_segments/epoch-1/{segment_id}"
        del store.data[f"{base}/meta"]
        segment_batch_calls = sum(
            bool(call["keys"] and "/_segments/" in call["keys"][0])
            for call in store.batch_put_calls
        )
        _, _, own2 = _write_rows(sidecar, slots, tokens, seed=25)

        self.assertTrue(_put_prefix(sink, sidecar, "repaired", tokens, slots, own2))
        self.assertIn(f"{base}/meta", store.data)
        self.assertEqual(
            sum(
                bool(call["keys"] and "/_segments/" in call["keys"][0])
                for call in store.batch_put_calls
            ),
            segment_batch_calls,
        )

    def test_page_internal_branch_reads_immutable_boundary(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        old_aux, old_last, old_own = _write_rows(
            sidecar, [0, 1, 2, 3], [1, 2, 3, 4], seed=4
        )
        self.assertTrue(
            _put_prefix(sink, sidecar, "old", [1, 2, 3, 4], [0, 1, 2, 3], old_own)
        )
        suffix_aux, suffix_last, suffix_own = _write_rows(
            sidecar, [20, 21], [9, 10], seed=5
        )
        self.assertTrue(
            _put_prefix(
                sink,
                sidecar,
                "branch",
                [1, 2, 9, 10],
                [0, 1, 20, 21],
                suffix_own,
            )
        )
        _, _, aux_back, last_back = _read_prefix_sample(store, "branch")
        self.assertTrue(torch.equal(aux_back, torch.cat((old_aux[:2], suffix_aux))))
        self.assertTrue(torch.equal(last_back, torch.cat((old_last[:2], suffix_last))))
        self.assertEqual(stats.prefix_boundary_republished_rows_ct, 2)
        self.assertEqual(stats.sidecar_direct_gather_rows_ct, 6)
        self.assertEqual(
            stats.segment_boundary_read_bytes_ct,
            4 * (AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize,
        )

    def test_middle_segment_branch_reuses_full_then_republishes_boundary(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        old_tokens = list(range(1, 9))
        old_aux, old_last, old_own = _write_rows(
            sidecar, list(range(8)), old_tokens, seed=6
        )
        self.assertTrue(
            _put_prefix(sink, sidecar, "old", old_tokens, list(range(8)), old_own)
        )
        old_meta = json.loads(store.data["test_store/_samples/old/meta"])
        new_aux, new_last, new_own = _write_rows(sidecar, [30, 31], [90, 91], seed=7)
        self.assertTrue(
            _put_prefix(
                sink,
                sidecar,
                "fork",
                old_tokens[:6] + [90, 91],
                list(range(6)) + [30, 31],
                new_own,
            )
        )
        fork_meta, _, aux_back, last_back = _read_prefix_sample(store, "fork")
        self.assertEqual(
            fork_meta["segments"][0]["segment_id"],
            old_meta["segments"][0]["segment_id"],
        )
        self.assertNotEqual(
            fork_meta["segments"][1]["segment_id"],
            old_meta["segments"][1]["segment_id"],
        )
        self.assertTrue(torch.equal(aux_back, torch.cat((old_aux[:6], new_aux))))
        self.assertTrue(torch.equal(last_back, torch.cat((old_last[:6], new_last))))
        self.assertEqual(stats.prefix_reused_rows_ct, 4)
        self.assertEqual(stats.prefix_boundary_republished_rows_ct, 2)

    def test_missing_reused_component_drops_warm_sample(self):
        sink, store, stats = _make_prefix_sink()
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        _, _, own = _write_rows(sidecar, [0, 1, 2, 3], [1, 2, 3, 4], seed=8)
        self.assertTrue(
            _put_prefix(sink, sidecar, "a", [1, 2, 3, 4], [0, 1, 2, 3], own)
        )
        meta = json.loads(store.data["test_store/_samples/a/meta"])
        missing = (
            "test_store/_segments/epoch-1/"
            f"{meta['segments'][0]['segment_id']}/last_hidden"
        )
        del store.data[missing]
        _, _, suffix_own = _write_rows(sidecar, [10], [5], seed=9)

        self.assertIsNone(
            _put_prefix(
                sink,
                sidecar,
                "ab",
                [1, 2, 3, 4, 5],
                [0, 1, 2, 3, 10],
                suffix_own,
            )
        )
        self.assertNotIn("test_store/_samples/ab/meta", store.data)
        self.assertEqual(stats.prefix_index_stale_hit_ct, 1)
        self.assertGreaterEqual(stats.segment_component_missing_miss_ct, 1)

    def test_segment_partial_put_never_publishes_meta_or_sample(self):
        class PartialSegmentStore(FakeStore):
            def batch_put_from(self, keys, ptrs, sizes, config):
                statuses = super().batch_put_from(keys, ptrs, sizes, config)
                if keys and "/_segments/" in keys[0]:
                    statuses[-1] = -99
                return statuses

        store = PartialSegmentStore()
        sink, _, stats = _make_prefix_sink(store=store)
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        _, _, own = _write_rows(sidecar, [0, 1], [1, 2], seed=10)
        self.assertIsNone(_put_prefix(sink, sidecar, "partial", [1, 2], [0, 1], own))
        self.assertFalse(any(key.endswith("/meta") for key in store.data))
        self.assertNotIn("test_store/_seq/0/0", store.data)
        self.assertEqual(stats.mooncake_batch_partial_failure_miss_ct, 1)
        self.assertGreaterEqual(stats.segment_orphan_ct, 1)
        self.assertEqual(stats.mooncake_payload_objects_ct, 1)
        self.assertGreater(stats.mooncake_payload_bytes_ct, 0)

    def test_partial_put_retry_commits_only_after_complete_existence_check(self):
        class FailFirstSegmentBatch(FakeStore):
            def __init__(self):
                super().__init__()
                self.failed = False

            def batch_put_from(self, keys, ptrs, sizes, config):
                statuses = super().batch_put_from(keys, ptrs, sizes, config)
                if keys and "/_segments/" in keys[0] and not self.failed:
                    self.failed = True
                    statuses[-1] = -99
                return statuses

        store = FailFirstSegmentBatch()
        sink, _, stats = _make_prefix_sink(store=store)
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        tokens = [1, 2]
        slots = [0, 1]
        _, _, own = _write_rows(sidecar, slots, tokens, seed=26)

        self.assertIsNone(_put_prefix(sink, sidecar, "retry", tokens, slots, own))
        self.assertNotIn("test_store/_samples/retry/meta", store.data)
        # The fake models an ambiguous status: both COMPLETE payload objects
        # landed even though one status was negative. Retry observes them,
        # repairs meta, and only then publishes the sample.
        self.assertTrue(_put_prefix(sink, sidecar, "retry", tokens, slots, own))
        self.assertIn("test_store/_samples/retry/meta", store.data)
        self.assertEqual(store.data["test_store/_seq/0/0"], b"retry")

    def test_input_ids_failure_leaves_segments_undiscoverable(self):
        class InputFailureStore(FakeStore):
            def batch_put_from(self, keys, ptrs, sizes, config):
                if keys and "/_samples/" in keys[0]:
                    return [-99] * len(keys)
                return super().batch_put_from(keys, ptrs, sizes, config)

        store = InputFailureStore()
        sink, _, stats = _make_prefix_sink(store=store)
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        _, _, own = _write_rows(sidecar, [0, 1], [1, 2], seed=27)

        self.assertIsNone(_put_prefix(sink, sidecar, "ids-fail", [1, 2], [0, 1], own))
        self.assertNotIn("test_store/_samples/ids-fail/meta", store.data)
        self.assertNotIn("test_store/_seq/0/0", store.data)
        self.assertGreaterEqual(stats.sample_publish_failed_miss_ct, 1)
        self.assertGreaterEqual(stats.segment_orphan_ct, 1)

    def test_segment_meta_failure_never_publishes_sample(self):
        class SegmentMetaFailureStore(FakeStore):
            def put(self, key, value, config):
                if "/_segments/" in key and key.endswith("/meta"):
                    return -99
                return super().put(key, value, config)

        store = SegmentMetaFailureStore()
        sink, _, stats = _make_prefix_sink(store=store)
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        _, _, own = _write_rows(sidecar, [0, 1], [1, 2], seed=28)

        self.assertIsNone(
            _put_prefix(sink, sidecar, "segment-meta-fail", [1, 2], [0, 1], own)
        )
        self.assertNotIn("test_store/_samples/segment-meta-fail/meta", store.data)
        self.assertNotIn("test_store/_seq/0/0", store.data)
        self.assertEqual(stats.segment_publish_failed_miss_ct, 1)
        self.assertGreaterEqual(stats.segment_orphan_ct, 1)

    def test_sample_meta_failure_never_appends_manifest(self):
        class SampleMetaFailureStore(FakeStore):
            def put(self, key, value, config):
                if "/_samples/" in key and key.endswith("/meta"):
                    return -99
                return super().put(key, value, config)

        store = SampleMetaFailureStore()
        sink, _, stats = _make_prefix_sink(store=store)
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        _, _, own = _write_rows(sidecar, [0, 1], [1, 2], seed=29)

        self.assertIsNone(
            _put_prefix(sink, sidecar, "sample-meta-fail", [1, 2], [0, 1], own)
        )
        self.assertNotIn("test_store/_samples/sample-meta-fail/meta", store.data)
        self.assertNotIn("test_store/_seq/0/0", store.data)
        self.assertEqual(stats.sample_publish_failed_miss_ct, 1)

    def test_restart_uses_new_epoch_without_breaking_old_sample(self):
        store = FakeStore()
        first, _, first_stats = _make_prefix_sink(store=store, writer_epoch="epoch-old")
        sidecar = _make_sidecar(stats=first_stats)
        old_aux, _, own = _write_rows(sidecar, [0, 1], [1, 2], seed=11)
        self.assertTrue(_put_prefix(first, sidecar, "old", [1, 2], [0, 1], own))
        first.close()

        second, _, second_stats = _make_prefix_sink(
            store=store, writer_epoch="epoch-new"
        )
        self.addCleanup(second.close)
        new_sidecar = _make_sidecar(stats=second_stats)
        _, _, new_own = _write_rows(new_sidecar, [3, 4], [1, 2], seed=12)
        self.assertTrue(
            _put_prefix(second, new_sidecar, "new", [1, 2], [3, 4], new_own)
        )
        old_meta, _, old_back, _ = _read_prefix_sample(store, "old")
        new_meta, _, _, _ = _read_prefix_sample(store, "new")
        self.assertTrue(torch.equal(old_back, old_aux))
        self.assertEqual(old_meta["writer_epoch"], "epoch-old")
        self.assertEqual(new_meta["writer_epoch"], "epoch-new")
        self.assertEqual(second_stats.prefix_reused_rows_ct, 0)

    def test_two_lanes_overlap_gather_with_blocked_writer(self):
        class BlockingStore(FakeStore):
            def __init__(self):
                super().__init__()
                self.first_segment_put = threading.Event()
                self.release_first_put = threading.Event()
                self._blocked_once = False

            def batch_put_from(self, keys, ptrs, sizes, config):
                if keys and "/_segments/" in keys[0] and not self._blocked_once:
                    self._blocked_once = True
                    self.first_segment_put.set()
                    self.release_first_put.wait(timeout=5.0)
                return super().batch_put_from(keys, ptrs, sizes, config)

        store = BlockingStore()
        sink, _, stats = _make_prefix_sink(store=store, prefix_lanes=2)
        self.addCleanup(sink.close)
        sidecar = _make_sidecar(stats=stats)
        tokens = list(range(12))
        slots = list(range(12))
        _, _, own = _write_rows(sidecar, slots, tokens, seed=13)
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                _put_prefix(sink, sidecar, "lanes", tokens, slots, own)
            )
        )
        worker.start()
        self.assertTrue(store.first_segment_put.wait(timeout=2.0))
        deadline = time.monotonic() + 2.0
        while stats.sidecar_direct_gather_rows_ct < 8 and time.monotonic() < deadline:
            time.sleep(0.001)
        # The first writer put is still blocked, but the second registered
        # lane has already gathered another segment.
        self.assertGreaterEqual(stats.sidecar_direct_gather_rows_ct, 8)
        store.release_first_put.set()
        worker.join(timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [True])
        self.assertGreaterEqual(stats.prefix_ready_queue_high_water_ct, 1)

    def test_prefix_startup_rejects_missing_group_or_batch_contract(self):
        class NoGroupConfig:
            def __init__(self):
                self.replica_num = 1

        store = FakeStore()
        with self.assertRaisesRegex(RuntimeError, "group_ids"):
            MooncakeHiddenSink(
                store_id="test_store",
                max_export_tokens=4,
                store=store,
                replicate_config=NoGroupConfig(),
                manifest_config=NoGroupConfig(),
                replicate_config_cls=NoGroupConfig,
                max_segment_rows=4,
                aux_width=AUX_WIDTH,
                last_width=LAST_WIDTH,
                dtype=DTYPE,
            )
        self.assertEqual(store.registered, [])

        class MissingBatchStore(FakeStore):
            batch_get_into = None

        store = MissingBatchStore()
        with self.assertRaisesRegex(RuntimeError, "batch_get_into"):
            MooncakeHiddenSink(
                store_id="test_store",
                max_export_tokens=4,
                store=store,
                replicate_config=FakeReplicateConfig(),
                manifest_config=FakeReplicateConfig(),
                replicate_config_cls=FakeReplicateConfig,
                max_segment_rows=4,
                aux_width=AUX_WIDTH,
                last_width=LAST_WIDTH,
                dtype=DTYPE,
            )
        self.assertEqual(store.registered, [])

    def test_prefix_registration_failure_rolls_back_prior_arenas(self):
        class RegistrationFailureStore(FakeStore):
            def __init__(self):
                super().__init__()
                self.register_calls = 0

            def register_buffer(self, ptr, size):
                self.register_calls += 1
                if self.register_calls == 3:
                    return -99
                return super().register_buffer(ptr, size)

        store = RegistrationFailureStore()
        with self.assertRaisesRegex(RuntimeError, "prefix lane 0"):
            MooncakeHiddenSink(
                store_id="test_store",
                max_export_tokens=4,
                store=store,
                replicate_config=FakeReplicateConfig(),
                manifest_config=FakeReplicateConfig(),
                replicate_config_cls=FakeReplicateConfig,
                max_segment_rows=4,
                aux_width=AUX_WIDTH,
                last_width=LAST_WIDTH,
                dtype=DTYPE,
            )
        self.assertEqual(store.registered, [])


class TestAsyncMooncakeStartup(CustomTestCase):
    def test_initial_timeout_returns_then_background_probe_enables_sink(self):
        allow_probe = threading.Event()
        stats = HiddenCaptureStats()
        created = []

        class FakeSink:
            registered_bytes = 123

            def __init__(self, **kwargs):
                self.fingerprint = None
                self.closed = False
                created.append((self, kwargs))

            def write_fingerprint(self, fingerprint):
                self.fingerprint = fingerprint

            def put(self, sample_id, record):
                return True

            def put_prefix_sample(self, **kwargs):
                return True

            def state_snapshot(self):
                return {"fake": True}

            def close(self):
                self.closed = True

        proxy = AsyncMooncakeHiddenSink(
            initial_probe_timeout_s=0.01,
            reconnect_interval_s=0.01,
            sink_factory=FakeSink,
            probe_fn=lambda _address, _timeout: allow_probe.is_set(),
            master_address="127.0.0.1:1",
            store_id="async-test",
            max_export_tokens=8,
            stats=stats,
        )
        started = time.monotonic()
        proxy.write_fingerprint({"model": "test"})
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(proxy.ready)
        self.assertEqual(stats.sink_initial_probe_timeout_ct, 1)

        allow_probe.set()
        deadline = time.monotonic() + 2.0
        while not proxy.ready and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(proxy.ready)
        self.assertEqual(stats.sink_reconnect_success_ct, 1)
        self.assertGreaterEqual(stats.sink_probe_failed_ct, 1)
        self.assertEqual(created[0][0].fingerprint, {"model": "test"})
        self.assertEqual(proxy.registered_bytes, 123)
        self.assertTrue(proxy.put("sample", {}))
        proxy.close()
        self.assertTrue(created[0][0].closed)


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

    def _publish(self, sink, sidecar, sample_id, *, slots, seed):
        tokens = [100 + s for s in slots]
        _, _, own = _write_rows(sidecar, slots, tokens, seed=seed)
        self.assertTrue(_put_prefix(sink, sidecar, sample_id, tokens, slots, own))

    def test_one_entry_per_sample_monotone_and_bare_sample_id(self):
        sink, store = _make_sink(dp_rank=3)
        sidecar = _make_sidecar()
        self._publish(sink, sidecar, "alpha", slots=[0, 1], seed=1)
        self._publish(sink, sidecar, "beta", slots=[2, 3], seed=2)
        self.assertEqual(store.data["test_store/_seq/3/0"], b"alpha")
        self.assertEqual(store.data["test_store/_seq/3/1"], b"beta")
        self.assertNotIn("test_store/_seq/3/2", store.data)

    def test_manifest_only_after_meta_and_none_for_rejected_puts(self):
        """A sample must never be discoverable unless complete: duplicate ids
        and oversized samples publish no manifest entry, and the entry for a
        good sample follows its meta (see put_order)."""
        sink, store = _make_sink(max_export_tokens=4)
        sidecar = _make_sidecar()
        self._publish(sink, sidecar, "s1", slots=[0, 1], seed=1)
        # Duplicate id: rejected, no new manifest entry.
        tokens = [100, 101]
        _, _, own = _write_rows(sidecar, [0, 1], tokens, seed=2)
        self.assertFalse(_put_prefix(sink, sidecar, "s1", tokens, [0, 1], own))
        # Oversized sample: input-id staging is max_export_tokens rows.
        big_slots = list(range(8))
        big_tokens = [100 + s for s in big_slots]
        _, _, big_own = _write_rows(sidecar, big_slots, big_tokens, seed=3)
        with self.assertRaises(SampleTooLargeError):
            _put_prefix(sink, sidecar, "big", big_tokens, big_slots, big_own)
        manifest_keys = [k for k in store.data if "/_seq/" in k]
        self.assertEqual(manifest_keys, ["test_store/_seq/0/0"])

    def test_manifest_uses_pinned_config_not_payload_config(self):
        """Manifests are hard-pinned while payloads stay unpinned; putting a
        manifest with the payload config silently reintroduces the
        orphan-by-eviction failure the pin exists to prevent."""
        payload_cfg = FakeReplicateConfig()
        manifest_cfg = FakeReplicateConfig()
        store = FakeStore()
        sink = MooncakeHiddenSink(
            store_id="test_store",
            max_export_tokens=16,
            store=store,
            replicate_config=payload_cfg,
            manifest_config=manifest_cfg,
            replicate_config_cls=FakeReplicateConfig,
            max_segment_rows=4,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
        )
        sink.write_fingerprint({"model_path": "test-model"})
        sidecar = _make_sidecar()
        tokens = [100, 101]
        _, _, own = _write_rows(sidecar, [0, 1], tokens, seed=1)
        self.assertTrue(_put_prefix(sink, sidecar, "s1", tokens, [0, 1], own))
        self.assertIs(store.put_configs["test_store/_seq/0/0"], manifest_cfg)
        self.assertIs(store.put_configs["test_store/_samples/s1/meta"], payload_cfg)

    def test_restart_resumes_after_tail_without_overwrite(self):
        sink, store = _make_sink()
        sidecar = _make_sidecar()
        for i in range(5):
            self._publish(sink, sidecar, f"s{i}", slots=[2 * i, 2 * i + 1], seed=i)
        reborn, _ = _make_sink(store=store)
        sidecar2 = _make_sidecar()
        tokens = [100, 101]
        _, _, own = _write_rows(sidecar2, [0, 1], tokens, seed=9)
        self.assertTrue(
            _put_prefix(reborn, sidecar2, "after-restart", tokens, [0, 1], own)
        )
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
        sidecar = _make_sidecar()
        for i in range(6):
            self._publish(sink, sidecar, f"s{i}", slots=[2 * i, 2 * i + 1], seed=i)
        del store.data["test_store/_seq/0/2"]  # simulate eviction/burn
        del store.data["test_store/_seq/0/3"]
        reborn, _ = _make_sink(store=store)
        sidecar2 = _make_sidecar()
        tokens = [100, 101]
        _, _, own = _write_rows(sidecar2, [0, 1], tokens, seed=9)
        self.assertTrue(
            _put_prefix(reborn, sidecar2, "after-restart", tokens, [0, 1], own)
        )
        self.assertEqual(store.data["test_store/_seq/0/6"], b"after-restart")
        self.assertEqual(store.data["test_store/_seq/0/5"], b"s5")

    def test_manifest_failure_is_counted_orphan_not_export_failure(self):
        """Requirement: manifest put failure = no retry, orphan counter +1,
        payload/meta NOT rolled back, and the put still reports success so
        the export worker doesn't double-count a sink failure. The failed
        attempt burns its sequence number (ambiguous failures must not retry
        into a first-write-wins key)."""

        class FailingManifestStore(FakeStore):
            def put(self, key, value, config):
                if "/_seq/" in key and key.endswith("/1"):
                    return -99
                return super().put(key, value, config)

        stats = _StatsStub()
        store = FailingManifestStore()
        sink, _ = _make_sink(store=store, stats=stats)
        sidecar = _make_sidecar()

        def publish(sample_id, slots, seed):
            tokens = [100 + s for s in slots]
            _, _, own = _write_rows(sidecar, slots, tokens, seed=seed)
            return _put_prefix(sink, sidecar, sample_id, tokens, slots, own)

        self.assertTrue(publish("ok", [0, 1], 1))
        self.assertTrue(publish("orphaned", [2, 3], 2))  # manifest fails
        self.assertTrue(publish("next", [4, 5], 3))
        self.assertEqual(stats.bumped.count("manifest_orphan_miss_ct"), 1)
        # Payload survives the manifest failure.
        self.assertIn("test_store/_samples/orphaned/meta", store.data)
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
