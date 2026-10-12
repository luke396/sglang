"""ForwardCostProfiler lifecycle with fake CUDA events (CPU only)."""

import os
import tempfile
import types
import unittest
from unittest.mock import patch

import msgspec

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.model_executor import model_runner
from sglang.srt.observability import forward_cost_profiler as fcp
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    """torch.cuda.Event stand-in: records a fake clock; completes when the test says."""

    created = []

    def __init__(self, enable_timing=True):
        self.ms = None
        self.done = False
        _FakeEvent.created.append(self)

    def record(self):
        self.ms = _FakeEvent.clock_ms

    def query(self):
        return self.done

    def elapsed_time(self, end):
        return end.ms - self.ms


def _complete_all():
    for e in _FakeEvent.created:
        e.done = True


class _FakeMode:
    def __init__(self, name):
        self.name = name


class _FakeForwardBatch:
    def __init__(
        self,
        *,
        seq,
        prefix,
        mode="EXTEND",
        global_tokens=None,
        split=0,
        waiting_before=None,
        waiting_after=None,
    ):
        self.forward_mode = _FakeMode(mode)
        self.batch_size = len(seq)
        self.extend_seq_lens_cpu = seq
        self.extend_prefix_lens_cpu = prefix
        self.global_num_tokens_cpu = global_tokens
        self.split_index = split
        self.waiting_reqs_before = waiting_before
        self.waiting_reqs_after = waiting_after


class TestForwardCostProfiler(CustomTestCase):
    def setUp(self):
        _FakeEvent.created = []
        _FakeEvent.clock_ms = 0.0
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(Event=_FakeEvent, synchronize=_complete_all)
        )
        for target in (
            "sglang.srt.utils.device_timer.torch",
            "sglang.srt.observability.forward_cost_profiler.torch",
        ):
            p = patch(target, fake_torch)
            p.start()
            self.addCleanup(p.stop)
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "forward_cost.dp3.jsonl")

    def _records(self):
        decoder = msgspec.json.Decoder(fcp.ForwardCostRecord)
        with open(self.path, "rb") as f:
            return [decoder.decode(line) for line in f]

    def _run(self, profiler, batch, *, start_ms, end_ms, split_forward_count=1):
        _FakeEvent.clock_ms = start_ms
        with fcp.forward_cost_scope(
            profiler, forward_batch=batch, split_forward_count=split_forward_count
        ):
            _FakeEvent.clock_ms = end_ms

    def test_shape_is_snapshot_at_entry_and_written_only_after_completion(self):
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=3)
        batch = _FakeForwardBatch(
            seq=[512, 100],
            prefix=[1024, 0],
            global_tokens=[612, 700],
            waiting_before=5,
            waiting_after=3,
        )
        self._run(profiler, batch, start_ms=10.0, end_ms=35.0)
        # The forward pads these in place, and overlap prepares the next batch.
        batch.extend_seq_lens_cpu[0] = 1
        batch.global_num_tokens_cpu.append(9)
        batch.waiting_reqs_before = 0
        batch.waiting_reqs_after = 0
        profiler.poll()
        self.assertEqual(self._records(), [])

        _complete_all()
        profiler.poll()
        (rec,) = self._records()
        self.assertEqual(rec.dp_rank, 3)
        self.assertAlmostEqual(rec.elapsed_s, 0.025)
        self.assertEqual(rec.shape.extend_seq_lens, [512, 100])
        self.assertEqual(rec.shape.extend_prefix_lens, [1024, 0])
        self.assertEqual(rec.shape.global_num_tokens, [612, 700])
        self.assertEqual(rec.shape.waiting_reqs_before, 5)
        self.assertEqual(rec.shape.waiting_reqs_after, 3)
        profiler.close(drain=True)

    def test_each_interval_keeps_its_own_shape_when_completion_is_out_of_order(self):
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=0)
        self._run(
            profiler, _FakeForwardBatch(seq=[8], prefix=[0]), start_ms=0.0, end_ms=5.0
        )
        self._run(
            profiler, _FakeForwardBatch(seq=[64], prefix=[32]), start_ms=5.0, end_ms=7.0
        )
        _FakeEvent.created[3].done = True  # second interval's end completes first
        profiler.poll()
        self.assertEqual(self._records(), [])  # FIFO: first interval still pending
        _complete_all()
        profiler.poll()
        recs = self._records()
        self.assertEqual([r.shape.extend_seq_lens for r in recs], [[8], [64]])
        self.assertEqual([round(r.elapsed_s, 6) for r in recs], [0.005, 0.002])
        profiler.close(drain=True)

    def test_close_with_drain_writes_the_tail_without_a_later_forward(self):
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=0)
        self._run(
            profiler, _FakeForwardBatch(seq=[8], prefix=[0]), start_ms=0.0, end_ms=4.0
        )
        profiler.close(drain=True)
        self.assertEqual(len(self._records()), 1)
        self.assertTrue(profiler._file.closed)

    def test_close_without_drain_does_not_synchronize_and_reports_loss(self):
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=0)
        self._run(
            profiler, _FakeForwardBatch(seq=[8], prefix=[0]), start_ms=0.0, end_ms=4.0
        )
        with self.assertLogs(fcp.logger, level="WARNING") as logs:
            profiler.close(drain=False)
        self.assertEqual(self._records(), [])
        self.assertIn("1 unwritten", logs.output[0])

    def test_forward_that_raises_is_recorded_as_failed_and_reraised(self):
        """A failed forward's partial interval must not look like a normal sample."""
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=0)
        batch = _FakeForwardBatch(seq=[512], prefix=[1024])
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with fcp.forward_cost_scope(
                profiler, forward_batch=batch, split_forward_count=1
            ):
                raise RuntimeError("boom")
        _complete_all()
        profiler.close(drain=False)
        (rec,) = self._records()
        self.assertTrue(rec.failed)
        self.assertEqual(rec.shape.extend_seq_lens, [512])

    def test_close_still_closes_the_file_when_an_event_query_raises(self):
        """A faulted device makes event.query raise; the file must still close."""
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=0)
        self._run(
            profiler,
            _FakeForwardBatch(seq=[8], prefix=[0]),
            start_ms=0.0,
            end_ms=4.0,
        )

        def faulted():
            raise RuntimeError("CUDA error")

        _FakeEvent.created[1].query = faulted
        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            profiler.close(drain=False)
        self.assertTrue(profiler._file.closed)

    def test_queue_state_does_not_filter_raw_modes_or_default_to_zero(self):
        profiler = fcp.ForwardCostProfiler(path=self.path, dp_rank=0)
        modes = ["EXTEND", "MIXED", "DECODE", "IDLE", "SPLIT_PREFILL"]
        for mode in modes:
            self._run(
                profiler,
                _FakeForwardBatch(seq=[8], prefix=[0], mode=mode),
                start_ms=0.0,
                end_ms=4.0,
            )
        profiler.close(drain=True)
        records = self._records()
        self.assertEqual([r.shape.forward_mode for r in records], modes)
        for rec in records:
            self.assertIsNone(rec.shape.waiting_reqs_before)
            self.assertIsNone(rec.shape.waiting_reqs_after)

    def test_refuses_to_append_to_an_existing_run(self):
        open(self.path, "w").close()
        with self.assertRaises(FileExistsError):
            fcp.ForwardCostProfiler(path=self.path, dp_rank=0)

    def test_disabled_scope_does_not_read_the_batch(self):
        with fcp.forward_cost_scope(
            None, forward_batch=object(), split_forward_count=1
        ):
            pass
        self.assertEqual(_FakeEvent.created, [])

    def test_maybe_create_gating(self):
        # The real ParallelContext: a stand-in hid a rank this branch lacks.
        def create(
            *, is_draft_worker=False, enable_elastic_ep=False, dp_rank=2, **ranks
        ):
            parallel = dict(pp_size=1, attn_tp_rank=0, attn_cp_rank=0)
            parallel.update(ranks)
            with get_parallel().override(**parallel):
                return fcp.ForwardCostProfiler.maybe_create(
                    is_draft_worker=is_draft_worker,
                    enable_elastic_ep=enable_elastic_ep,
                    dp_rank=dp_rank,
                )

        self.assertIsNone(create())  # env unset: off
        with envs.SGLANG_FORWARD_COST_PROFILE_DIR.override(self.dir):
            self.assertIsNone(create(is_draft_worker=True))
            self.assertIsNone(create(attn_tp_rank=1))
            self.assertIsNone(create(attn_cp_rank=1))
            with self.assertRaises(ValueError):
                create(pp_size=2)
            with self.assertRaises(ValueError):
                create(enable_elastic_ep=True)
            create().close(drain=True)
            create(dp_rank=None).close(drain=True)  # no DP controller
        for rank in (2, 0):
            self.assertTrue(
                os.path.exists(os.path.join(self.dir, f"forward_cost.dp{rank}.jsonl"))
            )


class _ReachedModelLoad(Exception):
    pass


class TestModelRunnerCreatesProfilerAfterParallelGroups(CustomTestCase):
    """The leader check reads live attention groups; it must run after they exist."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        groups = (parallel_state._ATTN_TP, parallel_state._ATTN_CP)
        self.addCleanup(
            lambda: (
                setattr(parallel_state, "_ATTN_TP", groups[0])
                or setattr(parallel_state, "_ATTN_CP", groups[1])
            )
        )
        self.server_args = ServerArgs(model_path="dummy", device="cpu")
        publish(self.server_args, role="scheduler")
        self.dir = tempfile.mkdtemp()

    def _construct(self, *, attn_tp_rank):
        def init_torch_distributed(runner):
            # What bootstrap.init_torch_distributed provides to the rank getters.
            parallel_state._ATTN_TP = types.SimpleNamespace(
                rank_in_group=attn_tp_rank, world_size=2
            )
            parallel_state._ATTN_CP = types.SimpleNamespace(
                rank_in_group=0, world_size=1
            )
            runner.tp_group = runner.pp_group = runner.attention_tp_group = None
            runner.pre_model_load_memory = 0

        captured = {}

        def initialize(runner):
            captured["profiler"] = runner.forward_cost_profiler
            raise _ReachedModelLoad

        model_config = types.SimpleNamespace(
            is_generation=True,
            is_multimodal=False,
            is_multimodal_chunked_prefill_supported=False,
            is_hybrid_swa=False,
            is_hybrid_swa_compress=False,
            attention_arch=None,
            attention_chunk_size=None,
        )
        Runner = model_runner.ModelRunner
        parallel_state._ATTN_TP = parallel_state._ATTN_CP = None
        with (
            envs.SGLANG_FORWARD_COST_PROFILE_DIR.override(self.dir),
            patch.object(Runner, "init_torch_distributed", init_torch_distributed),
            patch.object(Runner, "init_shared_mooncake_transfer_engine"),
            # CPU-only OpenMP binding shells out to lscpu.
            patch.object(Runner, "init_threads_binding"),
            patch.object(Runner, "initialize", initialize),
            self.assertRaises(_ReachedModelLoad),
        ):
            Runner(
                model_config=model_config,
                mem_fraction_static=0.5,
                gpu_id=0,
                ps=ParallelState.trivial(dp_rank=None),
                nccl_port=0,
                server_args=self.server_args,
            )
        return captured["profiler"]

    def test_leader_creates_profiler_before_model_load(self):
        profiler = self._construct(attn_tp_rank=0)
        self.assertIsNotNone(profiler)
        profiler.close(drain=False)
        self.assertEqual(os.listdir(self.dir), ["forward_cost.dp0.jsonl"])

    def test_non_leader_reads_its_initialized_group_rank(self):
        self.assertIsNone(self._construct(attn_tp_rank=1))
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
