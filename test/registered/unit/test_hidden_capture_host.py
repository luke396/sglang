"""Unit tests for srt/state_capturer/hidden_host.py and hidden_sink.py.

CPU-only: the staging ring is built with pin_memory=False and null events
(synchronous copies), so the full stage -> finalize -> export pipeline runs
without a GPU.
"""

import os
import tempfile
import time
import unittest

import torch

from sglang.srt.state_capturer.hidden_host import (
    HiddenCaptureBookkeeper,
    HiddenCaptureStats,
    HiddenFinalizeWorker,
    HiddenHostSidecar,
    HiddenStagingRing,
)
from sglang.srt.state_capturer.hidden_sink import (
    HiddenExportJob,
    HiddenExportWorker,
    HiddenFileSink,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

AUX_WIDTH = 8
LAST_WIDTH = 4
NUM_SIDECAR_SLOTS = 64
DTYPE = torch.float32


def _make_ring(num_slots=2, slot_tokens=4):
    return HiddenStagingRing(
        num_slots=num_slots,
        slot_tokens=slot_tokens,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
        pin_memory=False,
        use_cuda_events=False,
    )


def _make_sidecar():
    return HiddenHostSidecar(
        num_slots=NUM_SIDECAR_SLOTS,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
    )


def _rows(num_rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(num_rows, AUX_WIDTH, generator=g, dtype=DTYPE),
        torch.randn(num_rows, LAST_WIDTH, generator=g, dtype=DTYPE),
    )


class TestStagingRing(CustomTestCase):
    def test_acquire_enqueue_release_cycle(self):
        ring = _make_ring(num_slots=2, slot_tokens=4)
        slots = ring.try_acquire(2)
        self.assertEqual(len(slots), 2)
        self.assertIsNone(ring.try_acquire(1))  # exhausted

        aux, last = _rows(3)
        ring.enqueue_segment(
            slots[0],
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.arange(3, dtype=torch.int64),
            tokens=torch.tensor([11, 12, 13], dtype=torch.int64),
            req_ranges=[("r1", 0, 3)],
        )
        self.assertEqual(slots[0].ring_seq, 0)
        self.assertEqual(ring.last_enqueued_seq, 0)

        ready = ring.pop_ready()
        self.assertIs(ready, slots[0])
        # slots[1] was never enqueued; nothing further is ready.
        self.assertIsNone(ring.pop_ready())

        ring.release(ready)
        self.assertEqual(len(ring.try_acquire(1)), 1)

    def test_fifo_pop_order(self):
        ring = _make_ring(num_slots=3, slot_tokens=4)
        acquired = ring.try_acquire(3)
        for i, slot in enumerate(acquired):
            aux, last = _rows(2, seed=i)
            ring.enqueue_segment(
                slot,
                aux_rows=aux,
                last_rows=last,
                cache_locs=torch.arange(2, dtype=torch.int64) + i * 2,
                tokens=torch.zeros(2, dtype=torch.int64),
                req_ranges=[(f"r{i}", 0, 2)],
            )
        seqs = [ring.pop_ready().ring_seq for _ in range(3)]
        self.assertEqual(seqs, [0, 1, 2])


class TestSidecarSeqlock(CustomTestCase):
    def test_write_then_validated_read(self):
        sidecar = _make_sidecar()
        slots = torch.tensor([3, 5, 7], dtype=torch.long)
        tokens = torch.tensor([100, 101, 102], dtype=torch.int64)
        aux, last = _rows(3)
        gens = sidecar.write_rows(
            slots=slots, aux_rows=aux, last_rows=last, tokens=tokens
        )
        self.assertEqual(gens.tolist(), [2, 2, 2])

        # Own rows: exact-gen validation.
        out = sidecar.read_rows_validated(
            slots=slots,
            expected_tokens=tokens,
            own_slot_gens={3: 2, 5: 2, 7: 2},
        )
        self.assertIsNotNone(out)
        aux_out, last_out = out
        self.assertTrue(torch.equal(aux_out, aux))
        self.assertTrue(torch.equal(last_out, last))

    def test_own_row_gen_mismatch_fails(self):
        sidecar = _make_sidecar()
        slots = torch.tensor([3], dtype=torch.long)
        tokens = torch.tensor([100], dtype=torch.int64)
        aux, last = _rows(1)
        sidecar.write_rows(slots=slots, aux_rows=aux, last_rows=last, tokens=tokens)
        # Slot rewritten by a later request: gen advanced to 4, ours was 2.
        sidecar.write_rows(slots=slots, aux_rows=aux, last_rows=last, tokens=tokens)
        out = sidecar.read_rows_validated(
            slots=slots, expected_tokens=tokens, own_slot_gens={3: 2}
        )
        self.assertIsNone(out)

    def test_prefix_row_token_mismatch_fails(self):
        sidecar = _make_sidecar()
        slots = torch.tensor([9], dtype=torch.long)
        aux, last = _rows(1)
        sidecar.write_rows(
            slots=slots,
            aux_rows=aux,
            last_rows=last,
            tokens=torch.tensor([100], dtype=torch.int64),
        )
        # Warm-prefix validation (slot not in own records) with wrong token id.
        out = sidecar.read_rows_validated(
            slots=slots,
            expected_tokens=torch.tensor([999], dtype=torch.int64),
            own_slot_gens={},
        )
        self.assertIsNone(out)

    def test_never_written_row_fails(self):
        sidecar = _make_sidecar()
        out = sidecar.read_rows_validated(
            slots=torch.tensor([0], dtype=torch.long),
            expected_tokens=torch.tensor([-1], dtype=torch.int64),
            own_slot_gens={},
        )
        self.assertIsNone(out)  # gen == 0: never settled

    def test_write_in_flight_odd_gen_fails(self):
        """Writes advance gen by 2 under the sidecar mutex, so an odd gen can
        only mean corruption; the validated read must reject it."""
        sidecar = _make_sidecar()
        slots = torch.tensor([4], dtype=torch.long)
        tokens = torch.tensor([50], dtype=torch.int64)
        aux, last = _rows(1)
        sidecar.write_rows(slots=slots, aux_rows=aux, last_rows=last, tokens=tokens)
        sidecar.slot_gen[4] += 1  # corrupt: gen no longer a multiple of 2
        out = sidecar.read_rows_validated(
            slots=slots, expected_tokens=tokens, own_slot_gens={}
        )
        self.assertIsNone(out)

    def test_concurrent_write_read_never_returns_torn_rows(self):
        """P1-2 regression: sidecar reads must be mutually exclusive with
        writes. Alternating full-row writes of two distinct patterns race a
        validated reader; any row mixing both patterns (torn) or passing
        validation with the wrong generation would fail this test. Guarded by
        the sidecar mutex; the old seqlock let torn reads validate on
        weakly-ordered hosts."""
        import threading as _threading

        sidecar = _make_sidecar()
        slots = torch.tensor([1, 2, 3], dtype=torch.long)
        tokens = torch.tensor([7, 8, 9], dtype=torch.int64)
        patterns = [
            (torch.full((3, AUX_WIDTH), float(v)), torch.full((3, LAST_WIDTH), float(v)))
            for v in (1.0, 2.0)
        ]
        stop = _threading.Event()
        torn = []

        def writer():
            i = 0
            while not stop.is_set():
                aux, last = patterns[i % 2]
                sidecar.write_rows(
                    slots=slots, aux_rows=aux, last_rows=last, tokens=tokens
                )
                i += 1

        t = _threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            # Time-bounded: enough read/write collisions to catch a missing
            # lock (torn rows show up within milliseconds without it) while
            # keeping CPU CI fast.
            deadline = time.monotonic() + 3.0
            reads = 0
            while time.monotonic() < deadline and reads < 2000:
                reads += 1
                out = sidecar.read_rows_validated(
                    slots=slots, expected_tokens=tokens, own_slot_gens={}
                )
                if out is None:
                    continue
                aux_rows, last_rows = out
                # A consistent snapshot is uniformly one pattern value.
                for buf in (aux_rows, last_rows):
                    if not (
                        torch.equal(buf, torch.full_like(buf, 1.0))
                        or torch.equal(buf, torch.full_like(buf, 2.0))
                    ):
                        torn.append(buf)
        finally:
            stop.set()
            t.join(timeout=5)
        self.assertEqual(torn, [])


class TestBookkeeper(CustomTestCase):
    def test_miss_propagation_and_pop(self):
        bk = HiddenCaptureBookkeeper()
        bk.record_rows("r1", [1, 2], [2, 2])
        bk.mark_miss(["r1"])
        self.assertTrue(bk.is_missed("r1"))
        records = bk.pop("r1")
        self.assertEqual(records, {1: 2, 2: 2})
        # pop clears the miss flag along with the records.
        self.assertFalse(bk.is_missed("r1"))
        self.assertEqual(bk.pop("r1"), {})

    def test_invalidate_on_retract(self):
        bk = HiddenCaptureBookkeeper()
        bk.record_rows("r1", [1], [2])
        bk.invalidate("r1")
        self.assertEqual(bk.pop("r1"), {})

    def test_orphan_sweep(self):
        bk = HiddenCaptureBookkeeper()
        bk.record_rows("r_orphan", [1], [2])
        self.assertEqual(bk.sweep_orphans(ttl_s=-1.0), 1)
        self.assertEqual(bk.pop("r_orphan"), {})


class TestEndToEndPipeline(CustomTestCase):
    """stage -> finalize -> export against a real temp-dir file sink."""

    def _build(self, tmpdir, ring_slots=2, slot_tokens=8, queue_size=8):
        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        ring = _make_ring(num_slots=ring_slots, slot_tokens=slot_tokens)
        sidecar = _make_sidecar()
        finalize = HiddenFinalizeWorker(
            ring=ring, sidecar=sidecar, bookkeeper=bookkeeper, stats=stats
        )
        export = HiddenExportWorker(
            sidecar=sidecar,
            bookkeeper=bookkeeper,
            finalize_worker=finalize,
            stats=stats,
            sink=HiddenFileSink(tmpdir),
            queue_size=queue_size,
            barrier_timeout_s=5.0,
        )
        return stats, bookkeeper, ring, sidecar, finalize, export

    def _stage_and_finalize(self, ring, finalize, rid, kv_slots, tokens, seed=0):
        aux, last = _rows(len(kv_slots), seed=seed)
        slot = ring.try_acquire(1)[0]
        ring.enqueue_segment(
            slot,
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.tensor(kv_slots, dtype=torch.int64),
            tokens=torch.tensor(tokens, dtype=torch.int64),
            req_ranges=[(rid, 0, len(kv_slots))],
        )
        ready = ring.pop_ready()
        finalize.finalize_slot(ready)
        seq = ready.ring_seq
        ring.release(ready)
        finalize.last_finalized_seq = seq
        return aux, last

    def test_export_produces_specforge_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, bookkeeper, ring, sidecar, finalize, export = self._build(tmpdir)
            kv_slots, tokens = [10, 11, 12], [7, 8, 9]
            aux, last = self._stage_and_finalize(ring, finalize, "r1", kv_slots, tokens)

            job = HiddenExportJob(
                rid="r1",
                sample_id="r1",
                tokens=torch.tensor(tokens, dtype=torch.long),
                slots=torch.tensor(kv_slots, dtype=torch.long),
                ring_seq_barrier=0,
            )
            export.export_one(job)
            self.assertEqual(stats.export_ok_ct, 1)

            record = torch.load(os.path.join(tmpdir, "r1.ckpt"), weights_only=True)
            self.assertEqual(
                set(record),
                {"input_ids", "loss_mask", "aux_hidden_state", "hidden_state"},
            )
            self.assertEqual(record["input_ids"].tolist(), tokens)
            self.assertEqual(record["loss_mask"].tolist(), [1, 1, 1])
            self.assertEqual(record["aux_hidden_state"].shape, (1, 3, AUX_WIDTH))
            self.assertEqual(record["hidden_state"].shape, (1, 3, LAST_WIDTH))
            self.assertTrue(torch.equal(record["aux_hidden_state"][0], aux))
            self.assertTrue(torch.equal(record["hidden_state"][0], last))

    def test_missed_request_exports_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, bookkeeper, ring, sidecar, finalize, export = self._build(tmpdir)
            self._stage_and_finalize(ring, finalize, "r1", [1, 2], [5, 6])
            bookkeeper.mark_miss(["r1"])  # e.g. a later forward hit a full ring

            job = HiddenExportJob(
                rid="r1",
                sample_id="r1",
                tokens=torch.tensor([5, 6], dtype=torch.long),
                slots=torch.tensor([1, 2], dtype=torch.long),
                ring_seq_barrier=0,
            )
            export.export_one(job)
            self.assertEqual(stats.export_ok_ct, 0)
            self.assertEqual(os.listdir(tmpdir), [])

    def test_slot_reuse_after_finish_fails_validation(self):
        """ABA: r1 finishes, its slots are re-assigned to r2 and rewritten
        before r1's export runs -> gen mismatch, whole-sample miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, bookkeeper, ring, sidecar, finalize, export = self._build(tmpdir)
            self._stage_and_finalize(ring, finalize, "r1", [1, 2], [5, 6], seed=1)
            job = HiddenExportJob(
                rid="r1",
                sample_id="r1",
                tokens=torch.tensor([5, 6], dtype=torch.long),
                slots=torch.tensor([1, 2], dtype=torch.long),
                ring_seq_barrier=0,
            )
            # r2 reuses the same kv slots before r1's job is processed.
            self._stage_and_finalize(ring, finalize, "r2", [1, 2], [5, 6], seed=2)

            export.export_one(job)
            self.assertEqual(stats.prefix_invalid_miss_ct, 1)
            self.assertEqual(os.listdir(tmpdir), [])

    def test_warm_prefix_rows_from_other_request(self):
        """r2 shares a prefix written by r1; token-id validation admits it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, bookkeeper, ring, sidecar, finalize, export = self._build(tmpdir)
            aux1, last1 = self._stage_and_finalize(
                ring, finalize, "r1", [1, 2], [5, 6], seed=1
            )
            bookkeeper.pop("r1")  # r1 finished unsampled
            # r2: prefix slots 1,2 (cached; no rows of its own) + new slot 3.
            aux2, last2 = self._stage_and_finalize(
                ring, finalize, "r2", [3], [7], seed=2
            )
            job = HiddenExportJob(
                rid="r2",
                sample_id="r2",
                tokens=torch.tensor([5, 6, 7], dtype=torch.long),
                slots=torch.tensor([1, 2, 3], dtype=torch.long),
                ring_seq_barrier=1,
            )
            export.export_one(job)
            self.assertEqual(stats.export_ok_ct, 1)
            record = torch.load(os.path.join(tmpdir, "r2.ckpt"), weights_only=True)
            self.assertTrue(torch.equal(record["aux_hidden_state"][0, :2], aux1))
            self.assertTrue(torch.equal(record["aux_hidden_state"][0, 2:], aux2))
            self.assertTrue(torch.equal(record["hidden_state"][0, :2], last1))

    def test_export_queue_full_is_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, _, _, _, _, export = self._build(tmpdir, queue_size=1)
            job = HiddenExportJob(
                rid="rx",
                sample_id="rx",
                tokens=torch.tensor([1], dtype=torch.long),
                slots=torch.tensor([1], dtype=torch.long),
                ring_seq_barrier=0,
            )
            self.assertTrue(export.submit(job))
            self.assertFalse(export.submit(job))  # queue full -> miss, no block
            self.assertEqual(stats.export_queue_full_miss_ct, 1)

    def test_barrier_timeout_is_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, bookkeeper, ring, sidecar, finalize, export = self._build(tmpdir)
            export._barrier_timeout_s = 0.05
            job = HiddenExportJob(
                rid="r1",
                sample_id="r1",
                tokens=torch.tensor([1], dtype=torch.long),
                slots=torch.tensor([1], dtype=torch.long),
                ring_seq_barrier=99,  # never finalized
            )
            export.export_one(job)
            self.assertEqual(stats.export_timeout_miss_ct, 1)
            self.assertEqual(os.listdir(tmpdir), [])


class TestFileSink(CustomTestCase):
    def test_atomic_write_no_partial_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = HiddenFileSink(tmpdir)
            sink.put("s1", {"input_ids": torch.tensor([1, 2])})
            files = os.listdir(tmpdir)
            self.assertEqual(files, ["s1.ckpt"])  # no .tmp left behind

    def test_fingerprint_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = HiddenFileSink(tmpdir)
            sink.write_fingerprint({"model_path": "a"})
            sink.write_fingerprint({"model_path": "b"})  # DP replica double-write
            import json

            with open(os.path.join(tmpdir, "_fingerprint.json")) as f:
                self.assertEqual(json.load(f)["model_path"], "a")


class TestSamplingDeterminism(CustomTestCase):
    def test_sampled_is_deterministic_and_rate_bounded(self):
        from sglang.srt.state_capturer.hidden_states import HiddenStatesCapturer

        capturer = HiddenStatesCapturer.__new__(HiddenStatesCapturer)
        capturer.sample_rate = 0.5
        rids = [f"rid-{i}" for i in range(2000)]
        picks = [HiddenStatesCapturer._sampled(capturer, rid) for rid in rids]
        # Deterministic across calls.
        self.assertEqual(
            picks, [HiddenStatesCapturer._sampled(capturer, rid) for rid in rids]
        )
        rate = sum(picks) / len(picks)
        self.assertAlmostEqual(rate, 0.5, delta=0.05)

        capturer.sample_rate = 1.0
        self.assertTrue(
            all(HiddenStatesCapturer._sampled(capturer, r) for r in rids[:10])
        )


if __name__ == "__main__":
    unittest.main()
