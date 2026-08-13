"""Unit tests for srt/state_capturer/hidden_host.py and hidden_sink.py.

CPU-only: the staging ring is built with pin_memory=False and null events
(synchronous copies), so the full stage -> finalize -> export pipeline runs
without a GPU.
"""

import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.state_capturer.hidden_host import (
    DeviceTwinPool,
    HiddenCaptureBookkeeper,
    HiddenCaptureStats,
    HiddenFinalizeWorker,
    HiddenHostSidecar,
    HiddenStagingRing,
    HiddenVerifyD2HLauncher,
)
from sglang.srt.state_capturer.hidden_sink import (
    HiddenExportJob,
    HiddenExportWorker,
    HiddenFileSink,
)
from sglang.srt.state_capturer.hidden_states import HiddenStatesCapturer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

AUX_WIDTH = 8
LAST_WIDTH = 4
NUM_SIDECAR_SLOTS = 64
DTYPE = torch.float32


def _make_ring(num_slots=2, slot_tokens=4, stats=None, stats_prefix="prefill"):
    return HiddenStagingRing(
        num_slots=num_slots,
        slot_tokens=slot_tokens,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
        pin_memory=False,
        use_cuda_events=False,
        stats=stats,
        stats_prefix=stats_prefix,
    )


def _make_sidecar(stats=None):
    return HiddenHostSidecar(
        num_slots=NUM_SIDECAR_SLOTS,
        aux_width=AUX_WIDTH,
        last_width=LAST_WIDTH,
        dtype=DTYPE,
        stats=stats,
    )


def _rows(num_rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(num_rows, AUX_WIDTH, generator=g, dtype=DTYPE),
        torch.randn(num_rows, LAST_WIDTH, generator=g, dtype=DTYPE),
    )


class _GateEvent:
    def __init__(self):
        self.ready = threading.Event()

    def record(self):
        pass

    def query(self):
        return self.ready.is_set()


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


class TestSplitRingFinalizeOrder(CustomTestCase):
    """Prefill and verify stage through separate rings (a shared ring let a
    prefill-finalize stall drop whole verify batches: 2 stage_full events =
    75/300 samples lost at saturation). The export barrier stays a single
    number only if the finalize thread settles slots in GLOBAL enqueue order
    across both rings — finalizing seq 2 before seq 1 would let the barrier
    advance past an unsettled slot and the export read torn sidecar rows."""

    @staticmethod
    def _make_pair():
        from sglang.srt.state_capturer.hidden_host import _StagingSeqCounter

        seq = _StagingSeqCounter()
        prefill_ring = HiddenStagingRing(
            num_slots=2,
            slot_tokens=8,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            pin_memory=False,
            use_cuda_events=False,
            seq_counter=seq,
        )
        verify_ring = HiddenStagingRing(
            num_slots=2,
            slot_tokens=8,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            pin_memory=False,
            use_cuda_events=False,
            seq_counter=seq,
        )
        return prefill_ring, verify_ring

    def _enqueue_prefill(self, ring, rid, num_rows=2, seed=0):
        (slot,) = ring.try_acquire(1)
        aux, last = _rows(num_rows, seed=seed)
        ring.enqueue_segment(
            slot,
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.arange(num_rows, dtype=torch.int64),
            tokens=torch.zeros(num_rows, dtype=torch.int64),
            req_ranges=[(rid, 0, num_rows)],
        )

    def _enqueue_verify(self, ring, rid, seed=0):
        from sglang.srt.state_capturer.hidden_host import _DeviceTwin, _NullEvent

        aux, last = _rows(2, seed=seed)
        twin = _DeviceTwin(
            index=0,
            aux=aux,
            last=last,
            cache_loc=torch.arange(2, dtype=torch.int64),
            tokens=torch.arange(2, dtype=torch.int64),
            commit_lens=torch.tensor([2], dtype=torch.int32),
            fence_event=_NullEvent(),
        )
        (slot,) = ring.try_acquire(1)
        ring.enqueue_verify_segment(slot, twin=twin, rids=[rid], stride=2, num_reqs=1)

    def test_finalize_merges_rings_in_global_seq_order(self):
        prefill_ring, verify_ring = self._make_pair()
        finalize = HiddenFinalizeWorker(
            ring=prefill_ring,
            verify_ring=verify_ring,
            sidecar=_make_sidecar(),
            bookkeeper=HiddenCaptureBookkeeper(),
            stats=HiddenCaptureStats(),
        )
        # Interleave: prefill(0), verify(1), prefill(2), verify(3).
        self._enqueue_prefill(prefill_ring, "p0", seed=0)
        self._enqueue_verify(verify_ring, "v1", seed=1)
        self._enqueue_prefill(prefill_ring, "p2", seed=2)
        self._enqueue_verify(verify_ring, "v3", seed=3)

        order = []
        while (popped := finalize._pop_next_ready()) is not None:
            slot, source = popped
            order.append(slot.ring_seq)
            source.release(slot)
        self.assertEqual(order, [0, 1, 2, 3])

    def test_verify_ring_unaffected_by_full_prefill_ring(self):
        """The regression this split exists for: with the prefill ring
        exhausted (finalize stalled on a big memcpy), verify staging must
        still find slots instead of dropping the whole decode batch."""
        prefill_ring, verify_ring = self._make_pair()
        self._enqueue_prefill(prefill_ring, "p0")
        self._enqueue_prefill(prefill_ring, "p1")
        self.assertIsNone(prefill_ring.try_acquire(1))  # prefill exhausted
        self.assertIsNotNone(verify_ring.try_acquire(1))  # verify unaffected


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

    def test_direct_gather_writes_exact_preallocated_views(self):
        stats = HiddenCaptureStats()
        sidecar = _make_sidecar(stats=stats)
        slots = torch.tensor([3, 5, 7], dtype=torch.long)
        tokens = torch.tensor([100, 101, 102], dtype=torch.long)
        aux, last = _rows(3)
        gens = sidecar.write_rows(
            slots=slots, aux_rows=aux, last_rows=last, tokens=tokens
        )
        arena = torch.empty(3 * (AUX_WIDTH + LAST_WIDTH), dtype=DTYPE)
        aux_dst = arena[: 3 * AUX_WIDTH].view(3, AUX_WIDTH)
        last_dst = arena[3 * AUX_WIDTH :].view(3, LAST_WIDTH)
        aux_ptr = aux_dst.data_ptr()
        last_ptr = last_dst.data_ptr()

        self.assertTrue(
            sidecar.read_rows_validated_into(
                slots=slots,
                expected_tokens=tokens,
                own_slot_gens=dict(zip(slots.tolist(), gens.tolist())),
                aux_dst=aux_dst,
                last_dst=last_dst,
            )
        )
        self.assertEqual(aux_dst.data_ptr(), aux_ptr)
        self.assertEqual(last_dst.data_ptr(), last_ptr)
        self.assertTrue(torch.equal(aux_dst, aux))
        self.assertTrue(torch.equal(last_dst, last))
        self.assertEqual(stats.sidecar_direct_gather_rows_ct, 3)

    def test_direct_gather_validates_before_touching_destination(self):
        stats = HiddenCaptureStats()
        sidecar = _make_sidecar(stats=stats)
        slots = torch.tensor([3], dtype=torch.long)
        aux, last = _rows(1)
        sidecar.write_rows(
            slots=slots,
            aux_rows=aux,
            last_rows=last,
            tokens=torch.tensor([100]),
        )
        aux_dst = torch.full((1, AUX_WIDTH), -7.0, dtype=DTYPE)
        last_dst = torch.full((1, LAST_WIDTH), -9.0, dtype=DTYPE)

        self.assertFalse(
            sidecar.read_rows_validated_into(
                slots=slots,
                expected_tokens=torch.tensor([999]),
                own_slot_gens={},
                aux_dst=aux_dst,
                last_dst=last_dst,
            )
        )
        self.assertTrue(torch.equal(aux_dst, torch.full_like(aux_dst, -7.0)))
        self.assertTrue(torch.equal(last_dst, torch.full_like(last_dst, -9.0)))
        self.assertEqual(stats.sidecar_direct_gather_rows_ct, 0)

    def test_direct_gather_rejects_resizable_or_mistyped_destinations(self):
        sidecar = _make_sidecar()
        slots = torch.tensor([1, 2], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "aux_dst must have shape"):
            sidecar.read_rows_validated_into(
                slots=slots,
                expected_tokens=torch.tensor([1, 2]),
                own_slot_gens={},
                aux_dst=torch.empty((1, AUX_WIDTH), dtype=DTYPE),
                last_dst=torch.empty((2, LAST_WIDTH), dtype=DTYPE),
            )
        with self.assertRaisesRegex(ValueError, "last_dst must have shape"):
            sidecar.read_rows_validated_into(
                slots=slots,
                expected_tokens=torch.tensor([1, 2]),
                own_slot_gens={},
                aux_dst=torch.empty((2, AUX_WIDTH), dtype=DTYPE),
                last_dst=torch.empty((2, LAST_WIDTH), dtype=torch.float64),
            )

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
            (
                torch.full((3, AUX_WIDTH), float(v)),
                torch.full((3, LAST_WIDTH), float(v)),
            )
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

    def test_per_request_barrier_ignores_younger_enqueue(self):
        bk = HiddenCaptureBookkeeper()
        bk.record_enqueued(["a"], 3)
        bk.record_enqueued(["b"], 4)
        self.assertEqual(bk.export_barrier("a", global_seq=4), (3, True))
        self.assertEqual(bk.export_barrier("b", global_seq=4), (4, True))

    def test_barrier_tracks_max_seq_and_fully_warm_falls_back_global(self):
        bk = HiddenCaptureBookkeeper()
        bk.record_enqueued(["split"], 2)
        bk.record_enqueued(["split"], 7)
        self.assertEqual(bk.export_barrier("split", global_seq=9), (7, True))
        self.assertEqual(bk.export_barrier("warm", global_seq=9), (9, False))
        bk.invalidate("split")
        self.assertEqual(bk.export_barrier("split", global_seq=9), (9, False))


class TestObservability(CustomTestCase):
    def test_ring_sidecar_and_finalize_counters_close(self):
        stats = HiddenCaptureStats()
        ring = _make_ring(num_slots=2, slot_tokens=4, stats=stats)
        sidecar = _make_sidecar(stats=stats)
        bookkeeper = HiddenCaptureBookkeeper()
        finalize = HiddenFinalizeWorker(
            ring=ring, sidecar=sidecar, bookkeeper=bookkeeper, stats=stats
        )
        aux, last = _rows(3)
        (slot,) = ring.try_acquire(1)
        ring.enqueue_segment(
            slot,
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.tensor([1, 2, 3]),
            tokens=torch.tensor([11, 12, 13]),
            req_ranges=[("r", 0, 3)],
        )
        ready = ring.pop_ready()
        finalize.finalize_slot(ready)
        ring.release(ready)
        out = sidecar.read_rows_validated(
            slots=torch.tensor([1, 2, 3]),
            expected_tokens=torch.tensor([11, 12, 13]),
            own_slot_gens={},
        )
        self.assertIsNotNone(out)
        snapshot = stats.snapshot()
        self.assertEqual(snapshot["prefill_ring_high_water_ct"], 1)
        self.assertEqual(snapshot["prefill_rows_finalized_ct"], 3)
        self.assertEqual(snapshot["sidecar_write_rows_ct"], 3)
        self.assertEqual(snapshot["sidecar_gather_rows_ct"], 3)
        self.assertGreater(snapshot["sidecar_write_bytes_ct"], 0)
        self.assertGreater(snapshot["sidecar_gather_bytes_ct"], 0)
        self.assertGreater(snapshot["finalize_prefill_busy_ns_ct"], 0)

    def test_allocated_resource_bytes_and_bookkeeper_state_are_exact(self):
        ring = _make_ring(num_slots=3, slot_tokens=5)
        per_ring_slot = 5 * (
            (AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize
            + 2 * torch.empty((), dtype=torch.int64).element_size()
        )
        self.assertEqual(ring.allocated_bytes, 3 * per_ring_slot)

        sidecar = _make_sidecar()
        expected_sidecar = NUM_SIDECAR_SLOTS * (
            (AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize
            + torch.empty((), dtype=torch.int32).element_size()
            + torch.empty((), dtype=torch.int64).element_size()
        )
        self.assertEqual(sidecar.allocated_bytes, expected_sidecar)

        twins = DeviceTwinPool(
            num_twins=2,
            twin_tokens=5,
            max_reqs=3,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            device="cpu",
            use_cuda_events=False,
        )
        expected_twin = 2 * (
            5 * (AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize
            + 2 * 5 * torch.empty((), dtype=torch.int64).element_size()
            + 3 * 3 * torch.empty((), dtype=torch.int32).element_size()
            + torch.empty((), dtype=torch.int32).element_size()
        )
        self.assertEqual(twins.allocated_bytes, expected_twin)

        bookkeeper = HiddenCaptureBookkeeper()
        bookkeeper.record_enqueued(["pending"], 3)
        bookkeeper.mark_miss(["missed"])
        self.assertEqual(
            bookkeeper.state_snapshot(),
            {
                "finalize_record_rids": 0,
                "enqueued_rids": 1,
                "missed_rids": 1,
                "touched_rids": 2,
            },
        )


class TestWorkerShutdown(CustomTestCase):
    def test_capture_initialization_failure_disables_only_capture(self):
        server_args = SimpleNamespace(
            enable_hidden_state_capture=True,
            enable_dp_attention=False,
            attn_cp_size=1,
            pp_size=1,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
            disable_overlap_schedule=False,
            dllm_algorithm=None,
            enable_mixed_chunk=False,
            model_path="model",
            revision=None,
            speculative_num_draft_tokens=8,
            chunked_prefill_size=8192,
            max_prefill_tokens=16384,
        )
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=4), dtype=torch.float32
        )
        spec_aux_config = SimpleNamespace(
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=[1],
            eagle_use_aux_hidden_state=False,
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_HIDDEN_CAPTURE_SINK": "file",
                    "SGLANG_HIDDEN_CAPTURE_DIR": "/tmp",
                },
            ),
            mock.patch(
                "sglang.srt.state_capturer.hidden_states.get_parallel",
                return_value=SimpleNamespace(attn_tp_rank=0),
            ),
            mock.patch.object(
                HiddenStatesCapturer,
                "__init__",
                side_effect=RuntimeError("sink unavailable"),
            ),
        ):
            capturer = HiddenStatesCapturer.create(
                server_args=server_args,
                model_config=model_config,
                spec_aux_config=spec_aux_config,
                num_tokens=16,
                max_running_requests=4,
                device="cuda",
            )
        self.assertIsNone(capturer)

    def test_finalize_stop_waits_for_inflight_d2h_then_drains(self):
        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        ring = _make_ring(num_slots=1, slot_tokens=4)
        finalize = HiddenFinalizeWorker(
            ring=ring,
            sidecar=_make_sidecar(),
            bookkeeper=bookkeeper,
            stats=stats,
            poll_interval_s=0.0001,
        )
        aux, last = _rows(2)
        (slot,) = ring.try_acquire(1)
        gate = _GateEvent()
        slot.event = gate
        ring.enqueue_segment(
            slot,
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.tensor([4, 5]),
            tokens=torch.tensor([14, 15]),
            req_ranges=[("d2h", 0, 2)],
        )
        finalize.start()
        finalize.stop(drain=True)
        self.addCleanup(gate.ready.set)
        self.assertFalse(finalize.join(timeout_s=0.02))
        self.assertEqual(finalize.pending_count, 1)
        gate.ready.set()
        self.assertTrue(finalize.join(timeout_s=2.0))
        self.assertEqual(finalize.pending_count, 0)
        self.assertEqual(set(bookkeeper.pop("d2h")), {4, 5})

    def test_finalize_stop_drains_inflight_and_joins(self):
        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        ring = _make_ring(num_slots=1, slot_tokens=4)
        finalize = HiddenFinalizeWorker(
            ring=ring,
            sidecar=_make_sidecar(),
            bookkeeper=bookkeeper,
            stats=stats,
            poll_interval_s=0.0001,
        )
        aux, last = _rows(2)
        (slot,) = ring.try_acquire(1)
        ring.enqueue_segment(
            slot,
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.tensor([4, 5]),
            tokens=torch.tensor([14, 15]),
            req_ranges=[("drain", 0, 2)],
        )
        finalize.start()
        finalize.stop(drain=True)
        self.assertTrue(finalize.join(timeout_s=2.0))
        self.assertEqual(finalize.pending_count, 0)
        self.assertEqual(set(bookkeeper.pop("drain")), {4, 5})

    def test_export_stop_drains_queue_and_rejects_new_admission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = HiddenCaptureStats()
            bookkeeper = HiddenCaptureBookkeeper()
            sidecar = _make_sidecar()
            slots = torch.tensor([7])
            tokens = torch.tensor([17])
            aux, last = _rows(1)
            gens = sidecar.write_rows(
                slots=slots, aux_rows=aux, last_rows=last, tokens=tokens
            )
            bookkeeper.record_rows("queued", [7], gens.tolist())
            finalize = HiddenFinalizeWorker(
                ring=_make_ring(),
                sidecar=sidecar,
                bookkeeper=bookkeeper,
                stats=stats,
            )
            finalize.last_finalized_seq = 0
            export = HiddenExportWorker(
                sidecar=sidecar,
                bookkeeper=bookkeeper,
                finalize_worker=finalize,
                stats=stats,
                sink=HiddenFileSink(tmpdir),
                queue_size=2,
            )
            queued = HiddenExportJob(
                rid="queued",
                sample_id="queued",
                tokens=tokens,
                slots=slots,
                ring_seq_barrier=0,
            )
            self.assertTrue(export.submit(queued))
            export.start()
            export.stop(drain=True)
            self.assertTrue(export.join(timeout_s=2.0))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "queued.ckpt")))

            rejected = HiddenExportJob(
                rid="rejected",
                sample_id="rejected",
                tokens=tokens,
                slots=slots,
                ring_seq_barrier=0,
            )
            self.assertFalse(export.submit(rejected))
            self.assertEqual(stats.shutdown_admission_miss_ct, 1)

    def test_finalize_stop_waits_for_inflight_sidecar_write(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingWriteSidecar(HiddenHostSidecar):
            def write_rows(self, **kwargs):
                entered.set()
                release.wait(timeout=5.0)
                return super().write_rows(**kwargs)

        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        ring = _make_ring(num_slots=1, slot_tokens=4)
        sidecar = BlockingWriteSidecar(
            num_slots=NUM_SIDECAR_SLOTS,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            stats=stats,
        )
        finalize = HiddenFinalizeWorker(
            ring=ring,
            sidecar=sidecar,
            bookkeeper=bookkeeper,
            stats=stats,
            poll_interval_s=0.0001,
        )
        aux, last = _rows(2)
        (slot,) = ring.try_acquire(1)
        ring.enqueue_segment(
            slot,
            aux_rows=aux,
            last_rows=last,
            cache_locs=torch.tensor([6, 7]),
            tokens=torch.tensor([16, 17]),
            req_ranges=[("finalize", 0, 2)],
        )
        finalize.start()
        self.assertTrue(entered.wait(timeout=2.0))
        finalize.stop(drain=True)
        self.addCleanup(release.set)
        self.assertFalse(finalize.join(timeout_s=0.02))
        release.set()
        self.assertTrue(finalize.join(timeout_s=2.0))
        self.assertEqual(set(bookkeeper.pop("finalize")), {6, 7})

    def test_export_stop_waits_for_inflight_gather_then_drains(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingGatherSidecar(HiddenHostSidecar):
            def read_rows_validated(self, **kwargs):
                entered.set()
                release.wait(timeout=5.0)
                return super().read_rows_validated(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats = HiddenCaptureStats()
            bookkeeper = HiddenCaptureBookkeeper()
            sidecar = BlockingGatherSidecar(
                num_slots=NUM_SIDECAR_SLOTS,
                aux_width=AUX_WIDTH,
                last_width=LAST_WIDTH,
                dtype=DTYPE,
                stats=stats,
            )
            slots = torch.tensor([8])
            tokens = torch.tensor([18])
            aux, last = _rows(1)
            gens = sidecar.write_rows(
                slots=slots, aux_rows=aux, last_rows=last, tokens=tokens
            )
            bookkeeper.record_rows("gather", [8], gens.tolist())
            finalize = HiddenFinalizeWorker(
                ring=_make_ring(),
                sidecar=sidecar,
                bookkeeper=bookkeeper,
                stats=stats,
            )
            finalize.last_finalized_seq = 0
            export = HiddenExportWorker(
                sidecar=sidecar,
                bookkeeper=bookkeeper,
                finalize_worker=finalize,
                stats=stats,
                sink=HiddenFileSink(tmpdir),
                queue_size=1,
            )
            self.assertTrue(
                export.submit(
                    HiddenExportJob(
                        rid="gather",
                        sample_id="gather",
                        tokens=tokens,
                        slots=slots,
                        ring_seq_barrier=0,
                    )
                )
            )
            export.start()
            self.assertTrue(entered.wait(timeout=2.0))
            export.stop(drain=True)
            self.addCleanup(release.set)
            self.assertFalse(export.join(timeout_s=0.02))
            release.set()
            self.assertTrue(export.join(timeout_s=2.0))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "gather.ckpt")))

    def test_export_stop_waits_for_slow_put_and_submit_never_blocks(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingSink:
            prefix_enabled = False

            def put(self, sample_id, record):
                entered.set()
                release.wait(timeout=5.0)
                return True

        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        sidecar = _make_sidecar()
        slots = torch.tensor([9])
        tokens = torch.tensor([19])
        aux, last = _rows(1)
        gens = sidecar.write_rows(
            slots=slots, aux_rows=aux, last_rows=last, tokens=tokens
        )
        for rid in ("put-a", "put-b", "put-c"):
            bookkeeper.record_rows(rid, [9], gens.tolist())
        finalize = HiddenFinalizeWorker(
            ring=_make_ring(), sidecar=sidecar, bookkeeper=bookkeeper, stats=stats
        )
        finalize.last_finalized_seq = 0
        export = HiddenExportWorker(
            sidecar=sidecar,
            bookkeeper=bookkeeper,
            finalize_worker=finalize,
            stats=stats,
            sink=BlockingSink(),
            queue_size=1,
        )

        def _job(rid):
            return HiddenExportJob(
                rid=rid,
                sample_id=rid,
                tokens=tokens,
                slots=slots,
                ring_seq_barrier=0,
            )

        self.assertTrue(export.submit(_job("put-a")))
        export.start()
        self.assertTrue(entered.wait(timeout=2.0))
        started = time.monotonic()
        self.assertTrue(export.submit(_job("put-b")))
        self.assertFalse(export.submit(_job("put-c")))
        self.assertLess(time.monotonic() - started, 0.05)
        export.stop(drain=True)
        self.addCleanup(release.set)
        self.assertFalse(export.join(timeout_s=0.02))
        release.set()
        self.assertTrue(export.join(timeout_s=2.0))
        self.assertEqual(stats.export_queue_full_miss_ct, 1)

    def test_verify_launcher_stop_drains_header_wait(self):
        gate = _GateEvent()
        launched = []
        launcher = HiddenVerifyD2HLauncher(
            ring=SimpleNamespace(),
            twin_pool=SimpleNamespace(),
            bookkeeper=HiddenCaptureBookkeeper(),
            stats=HiddenCaptureStats(),
            capture_stream=None,
            capture_launch_lock=threading.Lock(),
            row_bytes=1,
            poll_interval_s=0.0001,
        )
        launcher._launch_payload = lambda slot: launched.append(slot)
        slot = SimpleNamespace(header_event=gate)
        launcher.start()
        launcher.submit(slot)
        launcher.stop(drain=True)
        self.addCleanup(gate.ready.set)
        self.assertFalse(launcher.join(timeout_s=0.02))
        self.assertEqual(launcher.pending_count, 1)
        gate.ready.set()
        self.assertTrue(launcher.join(timeout_s=2.0))
        self.assertEqual(launched, [slot])

    def test_capturer_close_never_releases_sink_before_export_join(self):
        events = []
        export_join_entered = threading.Event()
        release_export = threading.Event()

        class Worker:
            pending_count = 0

            def __init__(self, name, block_join=False):
                self.name = name
                self.block_join = block_join

            def stop(self, *, drain):
                events.append((self.name, "stop", drain))

            def join(self, timeout_s):
                events.append((self.name, "join"))
                if self.block_join:
                    export_join_entered.set()
                    release_export.wait(timeout=5.0)
                return True

        class ExportWorker(Worker):
            def stop_admission(self):
                events.append((self.name, "stop_admission"))

        class Sink:
            def __init__(self):
                self.registered = True

            def close(self):
                events.append(("sink", "close"))
                self.registered = False

        capturer = object.__new__(HiddenStatesCapturer)
        capturer._close_lock = threading.Lock()
        capturer._closed = False
        capturer._accepting = threading.Event()
        capturer._accepting.set()
        capturer.verify_launcher = Worker("launcher")
        capturer.finalize_worker = Worker("finalize")
        capturer.export_worker = ExportWorker("export", block_join=True)
        capturer.sink = Sink()
        capturer.stats = HiddenCaptureStats()

        result = []
        closer = threading.Thread(target=lambda: result.append(capturer.close(2.0)))
        closer.start()
        self.assertTrue(export_join_entered.wait(timeout=2.0))
        self.addCleanup(release_export.set)
        self.assertTrue(capturer.sink.registered)
        self.assertNotIn(("sink", "close"), events)
        release_export.set()
        closer.join(timeout=2.0)
        self.assertFalse(closer.is_alive())
        self.assertEqual(result, [True])
        self.assertFalse(capturer.sink.registered)
        self.assertLess(
            events.index(("export", "join")), events.index(("sink", "close"))
        )

    def test_capturer_close_timeout_keeps_sink_registered(self):
        class Worker:
            pending_count = 2

            def stop(self, *, drain):
                pass

            def join(self, timeout_s):
                return True

        class ExportWorker(Worker):
            def stop_admission(self):
                pass

            def join(self, timeout_s):
                return False

        class Sink:
            registered = True

            def close(self):
                self.registered = False

        capturer = object.__new__(HiddenStatesCapturer)
        capturer._close_lock = threading.Lock()
        capturer._closed = False
        capturer._accepting = threading.Event()
        capturer._accepting.set()
        capturer.verify_launcher = Worker()
        capturer.finalize_worker = Worker()
        capturer.export_worker = ExportWorker()
        capturer.sink = Sink()
        capturer.stats = HiddenCaptureStats()

        self.assertFalse(capturer.close(0.01))
        self.assertTrue(capturer.sink.registered)
        self.assertEqual(capturer.stats.shutdown_export_timeout_miss_ct, 2)


class TestVerifyCommittedRows(CustomTestCase):
    """Verify-window finalize: only rows [i*stride, i*stride+commit_lens[i])
    per request enter the sidecar. Rejected drafts entering the sidecar would
    poison warm-prefix reuse (their kv slots are freed and reused while the
    stale row still carries a plausible token id)."""

    def _make_verify_slot(self, ring, commit_lens, stride, seed=0):
        from sglang.srt.state_capturer.hidden_host import _DeviceTwin, _NullEvent

        num_reqs = len(commit_lens)
        num_rows = num_reqs * stride
        aux, last = _rows(num_rows, seed=seed)
        twin = _DeviceTwin(
            index=0,
            aux=aux,
            last=last,
            cache_loc=torch.arange(num_rows, dtype=torch.int64),
            tokens=torch.arange(100, 100 + num_rows, dtype=torch.int64),
            commit_lens=torch.tensor(commit_lens, dtype=torch.int32),
            fence_event=_NullEvent(),
        )
        (slot,) = ring.try_acquire(1)
        ring.enqueue_verify_segment(
            slot,
            twin=twin,
            rids=[f"r{i}" for i in range(num_reqs)],
            stride=stride,
            num_reqs=num_reqs,
        )
        return ring.pop_ready(), aux, last

    def test_only_committed_prefix_enters_sidecar(self):
        ring = _make_ring(num_slots=1, slot_tokens=16)
        sidecar = _make_sidecar()
        bookkeeper = HiddenCaptureBookkeeper()
        stats = HiddenCaptureStats()
        finalize = HiddenFinalizeWorker(
            ring=ring, sidecar=sidecar, bookkeeper=bookkeeper, stats=stats
        )
        stride = 4
        # r0 commits 2 of 4 rows, r1 commits all 4.
        slot, aux, last = self._make_verify_slot(ring, [2, 4], stride)
        finalize.finalize_slot(slot)

        # Committed rows present with correct payloads.
        committed = [0, 1, 4, 5, 6, 7]  # r0: rows 0-1; r1: rows 4-7
        for row in committed:
            out = sidecar.read_rows_validated(
                slots=torch.tensor([row], dtype=torch.long),
                expected_tokens=torch.tensor([100 + row], dtype=torch.int64),
                own_slot_gens={},
            )
            self.assertIsNotNone(out, f"committed row {row} missing")
            self.assertTrue(torch.equal(out[0][0], aux[row]))
        # Rejected rows (r0's rows 2-3) never written: gen stays 0.
        for row in (2, 3):
            self.assertEqual(int(sidecar.slot_gen[row]), 0)
        self.assertEqual(stats.verify_rows_committed_ct, 6)

        # Bookkeeper attribution is per request.
        self.assertEqual(set(bookkeeper.pop("r0")), {0, 1})
        self.assertEqual(set(bookkeeper.pop("r1")), {4, 5, 6, 7})

    def test_twin_released_after_finalize(self):
        from sglang.srt.state_capturer.hidden_host import DeviceTwinPool

        pool = DeviceTwinPool(
            num_twins=1,
            twin_tokens=16,
            max_reqs=4,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            device="cpu",
            use_cuda_events=False,
        )
        ring = _make_ring(num_slots=1, slot_tokens=16)
        sidecar = _make_sidecar()
        finalize = HiddenFinalizeWorker(
            ring=ring,
            sidecar=sidecar,
            bookkeeper=HiddenCaptureBookkeeper(),
            stats=HiddenCaptureStats(),
            twin_pool=pool,
        )
        twin = pool.try_acquire()
        self.assertIsNone(pool.try_acquire())  # pool exhausted
        twin.cache_loc[:2] = torch.tensor([10, 11], dtype=torch.int64)
        twin.tokens[:2] = torch.tensor([5, 6], dtype=torch.int64)
        twin.commit_lens[:1] = torch.tensor([1], dtype=torch.int32)
        (slot,) = ring.try_acquire(1)
        ring.enqueue_verify_segment(slot, twin=twin, rids=["r0"], stride=2, num_reqs=1)
        # Drive one loop iteration inline (the daemon path).
        ready = ring.pop_ready()
        finalize.finalize_slot(ready)
        if ready.twin is not None:
            pool.release(ready.twin)
        ring.release(ready)
        self.assertIsNotNone(pool.try_acquire())  # twin back in the pool

    def test_twin_reaped_before_finalize(self):
        """Coverage regression (matrix probe: 2x2048 -> 8x512 lifted baseline
        coverage 0.52 -> 0.87 with twin_miss 63 -> 0): a twin is dead the
        moment its slot's D2H event fires; reap_ready_twins must release it
        while the slot still sits in the finalize queue — waiting for the
        finalize memcpy starves fast decode steps of twins."""
        from sglang.srt.state_capturer.hidden_host import DeviceTwinPool

        pool = DeviceTwinPool(
            num_twins=1,
            twin_tokens=16,
            max_reqs=4,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            device="cpu",
            use_cuda_events=False,
        )
        ring = _make_ring(num_slots=1, slot_tokens=16)
        twin = pool.try_acquire()
        twin.cache_loc[:2] = torch.tensor([10, 11], dtype=torch.int64)
        twin.tokens[:2] = torch.tensor([5, 6], dtype=torch.int64)
        twin.commit_lens[:1] = torch.tensor([1], dtype=torch.int32)
        (slot,) = ring.try_acquire(1)
        ring.enqueue_verify_segment(slot, twin=twin, rids=["r0"], stride=2, num_reqs=1)
        self.assertIsNone(pool.try_acquire())

        # D2H done (_NullEvent queries True); slot NOT yet finalized.
        self.assertEqual(ring.reap_ready_twins(pool), 1)
        self.assertIsNotNone(pool.try_acquire())  # recycled early
        # The slot is still in flight with its data intact; finalize must not
        # double-release (twin field was cleared by the reap).
        ready = ring.pop_ready()
        self.assertIsNotNone(ready)
        self.assertIsNone(ready.twin)


class TestVerifyCommittedPack(CustomTestCase):
    @staticmethod
    def _case(device="cpu", compact_last=False):
        from sglang.srt.state_capturer.hidden_pack import (
            pack_committed_verify_rows_into,
        )

        bs, stride = 3, 4
        n = bs * stride
        aux = torch.arange(n * AUX_WIDTH, dtype=DTYPE, device=device).view(n, AUX_WIDTH)
        last_dense = (
            torch.arange(n * LAST_WIDTH, dtype=DTYPE, device=device).view(n, LAST_WIDTH)
            + 1000
        )
        cache_loc = torch.arange(20, 20 + n, dtype=torch.int64, device=device)
        tokens = torch.arange(200, 200 + n, dtype=torch.int64, device=device)
        verify_lens = torch.tensor([4, 3, 2], dtype=torch.int32, device=device)
        commit_lens = torch.tensor([1, 3, 2], dtype=torch.int32, device=device)
        if compact_last:
            last = torch.cat(
                [
                    last_dense[i * stride : i * stride + int(length)]
                    for i, length in enumerate(verify_lens.cpu().tolist())
                ]
            )
            last_strided = None
            last_compact = last
        else:
            last_strided = last_dense
            last_compact = None

        out_aux = torch.empty_like(aux)
        out_last = torch.empty_like(last_dense)
        out_cache = torch.empty_like(cache_loc)
        out_tokens = torch.empty_like(tokens)
        out_lens = torch.empty(bs, dtype=torch.int32, device=device)
        out_offsets = torch.empty(bs, dtype=torch.int32, device=device)
        out_total = torch.empty(1, dtype=torch.int32, device=device)
        out_verify_offsets = torch.empty(bs, dtype=torch.int32, device=device)
        pack_committed_verify_rows_into(
            aux_strided=aux,
            last_strided=last_strided,
            last_compact=last_compact,
            verify_lens=verify_lens if compact_last else None,
            verify_cache_loc=cache_loc,
            verify_tokens=tokens,
            commit_lens=commit_lens,
            bs=bs,
            stride=stride,
            out_aux=out_aux,
            out_last=out_last,
            out_cache_loc=out_cache,
            out_tokens=out_tokens,
            out_commit_lens=out_lens,
            out_commit_offsets=out_offsets,
            out_total_rows=out_total,
            out_verify_offsets=out_verify_offsets,
        )
        keep = torch.tensor([0, 4, 5, 6, 8, 9], dtype=torch.long, device=device)
        return {
            "aux": aux,
            "last": last_dense,
            "cache": cache_loc,
            "tokens": tokens,
            "keep": keep,
            "out_aux": out_aux,
            "out_last": out_last,
            "out_cache": out_cache,
            "out_tokens": out_tokens,
            "out_lens": out_lens,
            "out_offsets": out_offsets,
            "out_total": out_total,
        }

    def _assert_case(self, case):
        total = int(case["out_total"].cpu()[0])
        self.assertEqual(total, 6)
        self.assertEqual(case["out_lens"].cpu().tolist(), [1, 3, 2])
        self.assertEqual(case["out_offsets"].cpu().tolist(), [0, 1, 4])
        keep = case["keep"]
        self.assertTrue(
            torch.equal(case["out_aux"][:total], case["aux"].index_select(0, keep))
        )
        self.assertTrue(
            torch.equal(case["out_last"][:total], case["last"].index_select(0, keep))
        )
        self.assertTrue(
            torch.equal(case["out_cache"][:total], case["cache"].index_select(0, keep))
        )
        self.assertTrue(
            torch.equal(
                case["out_tokens"][:total], case["tokens"].index_select(0, keep)
            )
        )

    def test_cpu_dense_and_compact_source_parity(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                self._assert_case(self._case(compact_last=compact))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_dense_and_compact_source_parity(self):
        # Exercises the Triton kernels, including commit_len=1 and mixed/full
        # request prefixes. Tensor reads below are the test-only sync.
        for compact in (False, True):
            with self.subTest(compact=compact):
                self._assert_case(self._case(device="cuda", compact_last=compact))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_graph_replay_cannot_overwrite_packed_twin(self):
        """The capture pack is queued on the replay stream before the next
        graph replay. Even though the graph writes the same static sources,
        the first capture-owned twin must retain step one's values."""
        from sglang.srt.state_capturer.hidden_pack import (
            pack_committed_verify_rows_into,
        )

        bs, stride = 2, 4
        n = bs * stride
        aux_width, last_width = 8, 4
        graph_input = torch.ones((n, aux_width), device="cuda")
        aux_static = torch.empty_like(graph_input)
        last_static = torch.empty((n, last_width), device="cuda")
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                aux_static.copy_(graph_input)
                last_static.copy_(graph_input[:, :last_width] + 100)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                aux_static.copy_(graph_input)
                last_static.copy_(graph_input[:, :last_width] + 100)

        def alloc(width, dtype=DTYPE):
            return torch.empty((n, width), dtype=dtype, device="cuda")

        out_aux = alloc(aux_width)
        out_last = alloc(last_width)
        out_cache = torch.empty(n, dtype=torch.int64, device="cuda")
        out_tokens = torch.empty(n, dtype=torch.int64, device="cuda")
        out_lens = torch.empty(bs, dtype=torch.int32, device="cuda")
        out_offsets = torch.empty(bs, dtype=torch.int32, device="cuda")
        out_total = torch.empty(1, dtype=torch.int32, device="cuda")
        out_verify_offsets = torch.empty(bs, dtype=torch.int32, device="cuda")
        cache = torch.arange(n, dtype=torch.int64, device="cuda")
        tokens = cache + 1000
        lens = torch.tensor([1, 4], dtype=torch.int32, device="cuda")

        with torch.cuda.stream(capture_stream):
            graph_input.fill_(3)
            graph.replay()
            pack_committed_verify_rows_into(
                aux_strided=aux_static,
                last_strided=last_static,
                last_compact=None,
                verify_lens=None,
                verify_cache_loc=cache,
                verify_tokens=tokens,
                commit_lens=lens,
                bs=bs,
                stride=stride,
                out_aux=out_aux,
                out_last=out_last,
                out_cache_loc=out_cache,
                out_tokens=out_tokens,
                out_commit_lens=out_lens,
                out_commit_offsets=out_offsets,
                out_total_rows=out_total,
                out_verify_offsets=out_verify_offsets,
            )
            # This replay overwrites aux_static/last_static in place, but is
            # ordered after the pack on the same forward stream.
            graph_input.fill_(9)
            graph.replay()
        capture_stream.synchronize()
        self.assertTrue(torch.equal(out_aux[:5].cpu(), torch.full((5, 8), 3.0)))
        self.assertTrue(torch.equal(out_last[:5].cpu(), torch.full((5, 4), 103.0)))

    def test_two_phase_launcher_writes_contiguous_payload(self):
        from sglang.srt.state_capturer.hidden_host import (
            DeviceTwinPool,
            HiddenVerifyD2HLauncher,
        )
        from sglang.srt.state_capturer.hidden_pack import (
            pack_committed_verify_rows_into,
        )

        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        pool = DeviceTwinPool(
            num_twins=1,
            twin_tokens=8,
            max_reqs=2,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            device="cpu",
            use_cuda_events=False,
            stats=stats,
        )
        ring = HiddenStagingRing(
            num_slots=1,
            slot_tokens=8,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            pin_memory=False,
            use_cuda_events=False,
            stats=stats,
            stats_prefix="verify",
            max_verify_reqs=2,
        )
        twin = pool.try_acquire()
        aux, last = _rows(8, seed=19)
        cache = torch.arange(10, 18, dtype=torch.int64)
        tokens = torch.arange(110, 118, dtype=torch.int64)
        lens = torch.tensor([1, 3], dtype=torch.int32)
        pack_committed_verify_rows_into(
            aux_strided=aux,
            last_strided=last,
            last_compact=None,
            verify_lens=None,
            verify_cache_loc=cache,
            verify_tokens=tokens,
            commit_lens=lens,
            bs=2,
            stride=4,
            out_aux=twin.aux,
            out_last=twin.last,
            out_cache_loc=twin.cache_loc,
            out_tokens=twin.tokens,
            out_commit_lens=twin.commit_lens,
            out_commit_offsets=twin.commit_offsets,
            out_total_rows=twin.total_rows,
            out_verify_offsets=twin.verify_offsets,
        )
        twin.pack_start_event.record()
        twin.fence_event.record()
        (slot,) = ring.try_acquire(1)
        slot.commit_lens[:2].copy_(twin.commit_lens[:2])
        slot.total_rows.copy_(twin.total_rows)
        slot.header_event.record()
        seq = ring.reserve_verify_compact(
            slot,
            twin=twin,
            rids=["r0", "r1"],
            stride=4,
            num_reqs=2,
            header_submitted_ns=time.monotonic_ns(),
        )
        bookkeeper.record_enqueued(["r0", "r1"], seq)
        launcher = HiddenVerifyD2HLauncher(
            ring=ring,
            twin_pool=pool,
            bookkeeper=bookkeeper,
            stats=stats,
            capture_stream=None,
            capture_launch_lock=threading.Lock(),
            row_bytes=(AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize + 16,
        )
        launcher.start()
        launcher.submit(slot)
        deadline = time.monotonic() + 2
        ready = None
        while ready is None and time.monotonic() < deadline:
            ready = ring.pop_ready()
            time.sleep(0.001)
        launcher.stop(drain=True)
        self.assertTrue(launcher.join(timeout_s=2))
        self.assertIsNotNone(ready)

        sidecar = _make_sidecar()
        finalize = HiddenFinalizeWorker(
            ring=ring,
            sidecar=sidecar,
            bookkeeper=bookkeeper,
            stats=stats,
            twin_pool=pool,
        )
        finalize.finalize_slot(ready)
        pool.release(ready.twin)
        ring.release(ready)
        self.assertEqual(set(bookkeeper.pop("r0")), {10})
        self.assertEqual(set(bookkeeper.pop("r1")), {14, 15, 16})
        self.assertEqual(stats.verify_candidate_rows_staged_ct, 0)
        self.assertEqual(stats.verify_payload_rows_staged_ct, 4)
        self.assertEqual(stats.verify_rows_committed_ct, 4)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_two_phase_launcher_events_and_timing(self):
        from sglang.srt.state_capturer.hidden_host import (
            DeviceTwinPool,
            HiddenVerifyD2HLauncher,
        )

        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        pool = DeviceTwinPool(
            num_twins=1,
            twin_tokens=8,
            max_reqs=2,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            stats=stats,
        )
        ring = HiddenStagingRing(
            num_slots=1,
            slot_tokens=8,
            aux_width=AUX_WIDTH,
            last_width=LAST_WIDTH,
            dtype=DTYPE,
            stats=stats,
            stats_prefix="verify",
            max_verify_reqs=2,
        )
        twin = pool.try_acquire()
        twin.pack_start_event.record()
        twin.aux[:4].fill_(7)
        twin.last[:4].fill_(8)
        twin.cache_loc[:4].copy_(
            torch.tensor([10, 14, 15, 16], dtype=torch.int64, device="cuda")
        )
        twin.tokens[:4].copy_(
            torch.tensor([110, 114, 115, 116], dtype=torch.int64, device="cuda")
        )
        twin.commit_lens[:2].copy_(
            torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        )
        twin.total_rows.fill_(4)
        twin.fence_event.record()

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_event(twin.fence_event)
        (slot,) = ring.try_acquire(1)
        with torch.cuda.stream(capture_stream):
            slot.commit_lens[:2].copy_(twin.commit_lens[:2], non_blocking=True)
            slot.total_rows.copy_(twin.total_rows, non_blocking=True)
            slot.header_event.record()
        seq = ring.reserve_verify_compact(
            slot,
            twin=twin,
            rids=["r0", "r1"],
            stride=4,
            num_reqs=2,
            header_submitted_ns=time.monotonic_ns(),
        )
        bookkeeper.record_enqueued(["r0", "r1"], seq)
        launcher = HiddenVerifyD2HLauncher(
            ring=ring,
            twin_pool=pool,
            bookkeeper=bookkeeper,
            stats=stats,
            capture_stream=capture_stream,
            capture_launch_lock=threading.Lock(),
            row_bytes=(AUX_WIDTH + LAST_WIDTH) * DTYPE.itemsize + 16,
        )
        launcher.start()
        launcher.submit(slot)
        deadline = time.monotonic() + 5
        ready = None
        while ready is None and time.monotonic() < deadline:
            ready = ring.pop_ready()
            time.sleep(0.001)
        launcher.stop(drain=True)
        self.assertTrue(launcher.join(timeout_s=2))
        self.assertIsNotNone(ready)
        self.assertTrue(torch.equal(ready.aux[:4], torch.full_like(ready.aux[:4], 7)))
        self.assertGreater(stats.verify_pack_ns_ct, 0)
        self.assertGreater(stats.verify_payload_d2h_ns_ct, 0)
        pool.release(ready.twin)
        ring.release(ready)


class TestEndToEndPipeline(CustomTestCase):
    """stage -> finalize -> export against a real temp-dir file sink."""

    def _build(self, tmpdir, ring_slots=2, slot_tokens=8, queue_size=8):
        stats = HiddenCaptureStats()
        bookkeeper = HiddenCaptureBookkeeper()
        ring = _make_ring(num_slots=ring_slots, slot_tokens=slot_tokens, stats=stats)
        sidecar = _make_sidecar(stats=stats)
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
                {
                    "input_ids",
                    "loss_mask",
                    "aux_hidden_state",
                    "hidden_state",
                    "rid",
                    "prompt_len",
                },
            )
            self.assertEqual(record["rid"], "r1")
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

    def test_prefix_sink_owns_direct_gather_path(self):
        """Prefix mode must not first allocate/gather a whole sample in the
        export worker; the sink gathers only its planned suffix into a
        registered segment lane."""

        class PrefixSinkStub:
            prefix_enabled = True

            def __init__(self):
                self.calls = []

            def put_prefix_sample(self, **kwargs):
                rows = int(kwargs["slots"].numel())
                aux_dst = torch.empty((rows, AUX_WIDTH), dtype=DTYPE)
                last_dst = torch.empty((rows, LAST_WIDTH), dtype=DTYPE)
                ok = kwargs["sidecar"].read_rows_validated_into(
                    slots=kwargs["slots"],
                    expected_tokens=kwargs["tokens"],
                    own_slot_gens=kwargs["own_slot_gens"],
                    aux_dst=aux_dst,
                    last_dst=last_dst,
                )
                self.calls.append((kwargs, aux_dst, last_dst))
                return True if ok else None

        with tempfile.TemporaryDirectory() as tmpdir:
            stats, bookkeeper, ring, sidecar, finalize, export = self._build(tmpdir)
            sink = PrefixSinkStub()
            export.sink = sink
            aux, last = self._stage_and_finalize(
                ring, finalize, "r1", [1, 2, 3], [5, 6, 7], seed=21
            )
            export.export_one(
                HiddenExportJob(
                    rid="r1",
                    sample_id="s1",
                    tokens=torch.tensor([5, 6, 7], dtype=torch.long),
                    slots=torch.tensor([1, 2, 3], dtype=torch.long),
                    ring_seq_barrier=0,
                    prompt_len=2,
                )
            )
            self.assertEqual(stats.export_ok_ct, 1)
            self.assertEqual(stats.sidecar_direct_gather_rows_ct, 3)
            self.assertEqual(len(sink.calls), 1)
            kwargs, aux_back, last_back = sink.calls[0]
            self.assertEqual(kwargs["prompt_len"], 2)
            self.assertTrue(torch.equal(aux_back, aux))
            self.assertTrue(torch.equal(last_back, last))

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


class TestSampleIdSafety(CustomTestCase):
    """P2-1 regression: rid is caller-suppliable and flows into the sink path;
    a traversal rid must not escape the sink dir, and a duplicate must not
    silently overwrite the first export."""

    def test_traversal_rid_is_hashed(self):
        from sglang.srt.state_capturer.hidden_states import HiddenStatesCapturer

        for rid in ("../../etc/passwd", "a/b.ckpt", "x" * 200, "rid with space"):
            sid = HiddenStatesCapturer._sample_id_for(rid)
            self.assertNotIn("/", sid)
            self.assertNotIn("..", sid)
            self.assertLessEqual(len(sid), 128)
        # Safe uuid-shaped rids pass through unchanged (readable filenames).
        self.assertEqual(
            HiddenStatesCapturer._sample_id_for("abc-123_DEF"), "abc-123_DEF"
        )

    def test_duplicate_sample_id_keeps_first_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = HiddenFileSink(tmpdir)
            first = {"input_ids": torch.tensor([1])}
            second = {"input_ids": torch.tensor([2])}
            self.assertTrue(sink.put("dup", first))
            self.assertFalse(sink.put("dup", second))
            kept = torch.load(os.path.join(tmpdir, "dup.ckpt"), weights_only=True)
            self.assertEqual(kept["input_ids"].tolist(), [1])


class TestSnapshotKvSlots(CustomTestCase):
    """Regression for the capture-on p99 TPOT tail: the finish hook's kv-slot
    snapshot must go through the pinned+dedicated-stream path (a plain .cpu()
    is a synchronous pageable D2H that serializes behind capture's own staging
    transfers on the copy engine, stalling the scheduler thread for ms)."""

    @staticmethod
    def _make_host(num_tokens=64):
        from sglang.srt.state_capturer.hidden_states import HiddenStatesCapturer

        class _Host:
            _snapshot_kv_slots = HiddenStatesCapturer._snapshot_kv_slots
            _snapshot_kv_slots_batch = HiddenStatesCapturer._snapshot_kv_slots_batch

        host = _Host()
        cuda = torch.cuda.is_available()
        host.snapshot_stream = torch.cuda.Stream() if cuda else None
        host._snapshot_buf = (
            torch.empty((num_tokens,), dtype=torch.int32, pin_memory=True)
            if cuda
            else None
        )
        host._snapshot_event = torch.cuda.Event() if cuda else None
        host.stats = HiddenCaptureStats()
        return host

    def test_cpu_batched_snapshot_values_and_independence(self):
        host = self._make_host(num_tokens=4)
        first, second = host._snapshot_kv_slots_batch(
            [
                torch.tensor([2, 4, 6], dtype=torch.int32),
                torch.tensor([11, 13], dtype=torch.int32),
            ]
        )
        self.assertEqual(first.tolist(), [2, 4, 6])
        self.assertEqual(second.tolist(), [11, 13])
        self.assertEqual(first.dtype, torch.int64)
        self.assertEqual(host.stats.finish_snapshot_batches_ct, 1)
        self.assertEqual(host.stats.finish_snapshot_requests_ct, 2)
        self.assertEqual(host.stats.finish_snapshot_rows_ct, 5)

    def test_collect_batch_builds_jobs_from_one_snapshot_batch(self):
        from sglang.srt.state_capturer.hidden_states import HiddenStatesCapturer

        capturer = object.__new__(HiddenStatesCapturer)
        capturer._accepting = threading.Event()
        capturer._accepting.set()
        capturer.stats = HiddenCaptureStats()
        capturer.bookkeeper = HiddenCaptureBookkeeper()
        capturer.sample_rate = 1.0
        capturer.sink = SimpleNamespace(prefix_enabled=False)
        capturer.snapshot_stream = None
        capturer._snapshot_buf = None
        capturer._snapshot_event = None
        capturer.ring = SimpleNamespace(last_enqueued_seq=-1)
        jobs = []
        capturer.export_worker = SimpleNamespace(
            submit=lambda job: jobs.append(job) or True
        )
        reqs = [
            SimpleNamespace(
                rid="finish-a",
                req_pool_idx=0,
                origin_input_ids=[1, 2, 3],
                output_ids_through_stop=[4, 5],
                extra_key=None,
                lora_id=None,
                input_embeds=None,
                multimodal_inputs=None,
                positional_embed_overrides=None,
            ),
            SimpleNamespace(
                rid="finish-b",
                req_pool_idx=1,
                origin_input_ids=[7, 8],
                output_ids_through_stop=[9, 10, 11],
                extra_key=None,
                lora_id=None,
                input_embeds=None,
                multimodal_inputs=None,
                positional_embed_overrides=None,
            ),
        ]
        pool = SimpleNamespace(
            req_to_token=torch.tensor(
                [[20, 21, 22, 23, -1], [30, 31, 32, 33, -1]],
                dtype=torch.int32,
            )
        )
        capturer.collect_batch_at_finish(reqs, pool)
        self.assertEqual(len(jobs), 2)
        # Both sequences have 5 tokens, hence forwarded rows [0, 4).
        self.assertEqual(jobs[0].slots.tolist(), [20, 21, 22, 23])
        self.assertEqual(jobs[1].slots.tolist(), [30, 31, 32, 33])
        self.assertEqual(jobs[0].tokens.tolist(), [1, 2, 3, 4])
        self.assertEqual(jobs[1].tokens.tolist(), [7, 8, 9, 10])
        self.assertEqual(capturer.stats.finish_snapshot_batches_ct, 1)
        self.assertEqual(capturer.stats.finish_snapshot_requests_ct, 2)

    def test_collect_batch_snapshot_failure_is_fail_soft(self):
        from sglang.srt.state_capturer.hidden_states import HiddenStatesCapturer

        capturer = object.__new__(HiddenStatesCapturer)
        capturer._accepting = threading.Event()
        capturer._accepting.set()
        capturer.stats = HiddenCaptureStats()
        capturer.bookkeeper = HiddenCaptureBookkeeper()
        capturer.sample_rate = 1.0
        capturer.sink = SimpleNamespace(prefix_enabled=False)
        capturer.ring = SimpleNamespace(last_enqueued_seq=-1)
        capturer.export_worker = SimpleNamespace(
            submit=lambda _job: self.fail("snapshot failure must not enqueue a job")
        )
        capturer._snapshot_kv_slots_batch = lambda _ranges: (_ for _ in ()).throw(
            RuntimeError("injected snapshot failure")
        )
        req = SimpleNamespace(
            rid="finish-fail-soft",
            req_pool_idx=0,
            origin_input_ids=[1, 2],
            output_ids_through_stop=[3],
            extra_key=None,
            lora_id=None,
            input_embeds=None,
            multimodal_inputs=None,
            positional_embed_overrides=None,
        )
        pool = SimpleNamespace(
            req_to_token=torch.tensor([[20, 21, -1]], dtype=torch.int32)
        )

        # No exception reaches the scheduler, which remains free to release
        # the request's KV slots after this hook returns.
        capturer.collect_batch_at_finish([req], pool)

        self.assertEqual(capturer.stats.finish_snapshot_failed_miss_ct, 1)
        self.assertTrue(capturer.bookkeeper.is_missed(req.rid))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_gpu_batched_snapshot_single_wait_and_buffer_growth(self):
        host = self._make_host(num_tokens=4)
        snapshots = host._snapshot_kv_slots_batch(
            [
                torch.arange(5, dtype=torch.int32, device="cuda"),
                torch.arange(20, 26, dtype=torch.int32, device="cuda"),
            ]
        )
        self.assertEqual(snapshots[0].tolist(), list(range(5)))
        self.assertEqual(snapshots[1].tolist(), list(range(20, 26)))
        self.assertGreaterEqual(host._snapshot_buf.numel(), 11)
        self.assertEqual(host.stats.finish_snapshot_batches_ct, 1)
        self.assertEqual(host.stats.finish_snapshot_requests_ct, 2)
        self.assertEqual(host.stats.finish_snapshot_d2h_bytes_ct, 11 * 4)
        self.assertEqual(host.stats.finish_snapshot_grow_ct, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_gpu_snapshot_values_and_buffer_reuse(self):
        host = self._make_host()
        # int32 source mirrors req_to_token and lands into same-dtype pinned
        # memory; the returned independent CPU snapshot is converted to long.
        first = host._snapshot_kv_slots(
            torch.arange(10, dtype=torch.int32, device="cuda")
        )
        second = host._snapshot_kv_slots(
            torch.arange(100, 108, dtype=torch.int32, device="cuda")
        )
        # clone() semantics: the first result must survive buffer reuse.
        self.assertEqual(first.tolist(), list(range(10)))
        self.assertEqual(second.tolist(), list(range(100, 108)))
        self.assertEqual(first.dtype, torch.int64)
        self.assertFalse(first.is_cuda)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_gpu_snapshot_correct_while_copy_engine_busy(self):
        """The bug scenario: large in-flight D2H traffic on another stream
        (capture staging) while the hook snapshots. Asserts correctness; the
        latency property (not queuing behind the big copies) is covered by
        the bench bisection in PR#3."""
        host = self._make_host()
        big_src = torch.full((64 * 1024 * 1024,), 3, dtype=torch.uint8, device="cuda")
        big_dst = torch.empty_like(big_src, device="cpu", pin_memory=True)
        busy_stream = torch.cuda.Stream()
        with torch.cuda.stream(busy_stream):
            for _ in range(4):
                big_dst.copy_(big_src, non_blocking=True)
        out = host._snapshot_kv_slots(
            torch.arange(32, dtype=torch.int32, device="cuda")
        )
        self.assertEqual(out.tolist(), list(range(32)))
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
