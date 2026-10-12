import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.managers.schedule_policy as schedule_policy
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
    estimate_prefill_extend_tile_metrics,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.observability.forward_cost_profiler import _snapshot_shape
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingDelayer:
    """Duck-typed stand-in for PrefillDelayerSinglePassExecutor that records
    the local_prefillable value of every negotiate call and returns a fixed
    verdict."""

    def __init__(self, allow: bool):
        self.allow = allow
        self.calls = []

    def negotiate_should_allow_prefill(self, local_prefillable, **kwargs):
        self.calls.append(local_prefillable)
        return self.allow


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_token_allocator = self.create_token_allocator()

    def create_tree_cache(
        self,
        *,
        full_evictable_size: int = 0,
        swa_evictable_size: int = 0,
        evictable_size: int = 0,
    ) -> MagicMock:
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = full_evictable_size
        tree_cache.swa_evictable_size.return_value = swa_evictable_size
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        return tree_cache

    def create_token_allocator(
        self,
        *,
        full_available_size: int = 0,
        swa_available_size: int = 0,
        available_size: int = 0,
        size_swa: int = 1_000_000,
    ) -> MagicMock:
        allocator = MagicMock()
        allocator.full_available_size.return_value = full_available_size
        allocator.swa_available_size.return_value = swa_available_size
        allocator.available_size.return_value = available_size
        allocator.size_swa = size_swa
        return allocator

    def create_running_batch(self, reqs=None) -> MagicMock:
        batch = MagicMock()
        batch.reqs = list(reqs or [])
        batch.release_req.return_value = None
        batch.filter_batch.return_value = None
        return batch

    def scheduling_order(self, *, schedule_low_priority_values_first: bool):
        """State the policy on the context, which is where the scheduler reads
        it: `preempt_to_schedule` takes no record to state it on."""
        override = get_context().override_server_args(
            schedule_low_priority_values_first=schedule_low_priority_values_first
        )
        override.install()
        self.addCleanup(override.restore)

    def create_mock_req(self, rid, priority, max_new_tokens, output_len=0, wait_time=0):
        req = MagicMock(spec=Req)
        req.rid = str(rid)
        req.priority = priority
        req.prefix_indices = []
        req.full_untruncated_fill_ids = []
        req.output_ids = [0] * output_len
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
        req.retracted_stain = False
        req.host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        return req

    def create_adder(self, running_batch, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=self.mock_tree_cache,
            token_to_kv_pool_allocator=self.mock_token_allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        return PrefillAdder(**defaults)

    def _queue_snapshot_scheduler(self, *, waiting, chunked, slots, mode):
        s = Scheduler.__new__(Scheduler)
        s.waiting_queue = list(waiting)
        s.chunked_req = chunked
        s.disaggregation_mode = mode
        s.prefill_delayer = s.min_free_slots_delayer = s.dynamic_chunk_sizer = None
        s.enable_priority_preemption = s.is_hybrid_swa = s.is_mixed_chunk = False
        s.enable_lora = s.enable_hicache_storage = s.enable_lmcache = False
        s.enable_hierarchical_cache = s.enable_unified_cache_external_linker = False
        s.enable_priority_scheduling = False
        s.enable_overlap = True
        s.scheduler_stage_metrics = None
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        s.processed_tokens_counter = 0
        s.page_size = 1
        s.max_prefill_tokens = s.chunked_prefill_size = 512
        s.max_prefill_bs = s.max_running_requests = 64
        s.priority_scheduling_preemption_threshold = 0
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.dllm_config = s.truncation_align_size = None
        s.spec_algorithm = SpeculativeAlgorithm.NONE
        s.tree_cache = self.create_tree_cache()
        s.tree_cache.supports_mamba.return_value = False
        s.tree_cache.supports_fast_match_prefix.return_value = False
        s.token_to_kv_pool_allocator = self.create_token_allocator(available_size=32768)
        s.token_to_kv_pool_allocator.page_size = 1
        s.req_to_token_pool = SimpleNamespace(
            device="cpu", mamba_allocator=None, available_size=lambda: slots
        )
        s.policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=s.tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        s.model_config = SimpleNamespace(
            vocab_size=32768,
            requires_mm_token_modalities=False,
            model_is_mrope=False,
            is_encoder_decoder=False,
        )
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                device="cpu",
                is_draft_worker=False,
                model_config=s.model_config,
                attn_backend=SimpleNamespace(extend_attention_block_m=64),
                prefill_aware_swa=False,
                prefill_attention_backend_str="torch_native",
                kv_index_translator=SimpleNamespace(
                    rebind_write_loc=lambda batch: None
                ),
                ngram_embedding_manager=SimpleNamespace(enabled=False),
                lora_manager=None,
            )
        )
        s.load_inquirer = SimpleNamespace(
            get_waiting_queue=lambda: s.waiting_queue,
            get_chunked_req=lambda: s.chunked_req,
        )
        s.load_inquirer._get_num_pending_tokens = MethodType(
            SchedulerLoadInquirer._get_num_pending_tokens, s.load_inquirer
        )
        return s

    def _queue_snapshot_req(self, rid, n):
        req = self.create_mock_req(rid, priority=0, max_new_tokens=1)
        req.full_untruncated_fill_ids = list(range(n))
        req.origin_input_ids = req.full_untruncated_fill_ids
        req.last_node = MagicMock()
        req.seqlen = n
        req.token_indices_to_pool = req.beam_group = req.token_type_ids = None
        req.lora_id = req.grammar = None
        req.return_logprob = req.is_prefill_only = False
        req.return_hidden_states_mode = CaptureHiddenMode.NULL
        req.inflight_middle_chunks = 0
        req.sampling_params.ignore_eos = False
        req.time_stats.set_forward_entry_time = lambda ts: None
        req.time_stats.set_last_scheduled_time = lambda *args: None
        return req

    def test_waiting_snapshot_tracks_admission_not_remaining_chunk_work(self):
        """Pending chunk tokens and an emptied waiting queue are different states."""
        with get_context().override_server_args(
            device="cpu", pp_max_micro_batch_size=64
        ):
            for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
                for case in ("chunk_only", "all_admitted", "still_waiting"):
                    with self.subTest(mode=mode, case=case):
                        chunked = (
                            self._queue_snapshot_req("long", 4096)
                            if case == "chunk_only"
                            else None
                        )
                        waiting = (
                            []
                            if chunked
                            else [
                                self._queue_snapshot_req(str(i), 128) for i in range(2)
                            ]
                        )
                        s = self._queue_snapshot_scheduler(
                            waiting=waiting,
                            chunked=chunked,
                            slots=1 if case == "still_waiting" else 64,
                            mode=mode,
                        )
                        running = ScheduleBatch(reqs=[])
                        # Allocation/H2D is outside the admission snapshot contract.
                        with patch.object(ScheduleBatch, "prepare_for_extend"):
                            if mode == DisaggregationMode.PREFILL:
                                s.process_pending_chunked_abort = lambda: None
                                s._process_hicache_events = lambda **kwargs: None
                                s.resolve_waiting_queue_bootstrap = lambda: None
                                s.process_prefill_chunk = lambda **kwargs: None
                                s.dp_attn_adapter = SimpleNamespace(
                                    maybe_prepare_mlp_sync_batch=lambda batch: batch
                                )
                                plan = s.get_next_disagg_prefill_batch_to_run(
                                    running_batch=running, last_batch=None
                                )
                            else:
                                plan = s.get_new_batch_prefill(running)
                            batch = plan.batch_to_run
                        stats = batch.prefill_stats
                        expected = {
                            "chunk_only": (0, 0),
                            "all_admitted": (2, 0),
                            "still_waiting": (2, 1),
                        }[case]
                        self.assertEqual(
                            (stats.waiting_reqs_before, stats.waiting_reqs_after),
                            expected,
                        )
                        self.assertEqual(
                            len(batch.reqs), 2 - expected[1] if waiting else 1
                        )
                        if chunked:
                            self.assertGreater(stats.num_pending_tokens, 0)
                            self.assertIs(s.chunked_req, chunked)
                        elif case == "still_waiting":
                            self.assertLess(
                                sum(r.extend_len for r in batch.reqs),
                                s.chunked_prefill_size,
                            )
                        # Later arrivals/departures must not change the scheduled batch.
                        s.waiting_queue = [self._queue_snapshot_req("later", 32)] * 7
                        batch = batch.copy()
                        self.assertEqual(
                            (
                                batch.prefill_stats.waiting_reqs_before,
                                batch.prefill_stats.waiting_reqs_after,
                            ),
                            expected,
                        )
                        self._check_forward_queue_snapshot(batch, s, expected)

    def _check_forward_queue_snapshot(self, batch, scheduler, expected):
        # Supply CPU tensors in place of GPU allocation, then run the real factory.
        batch.forward_mode = ForwardMode.EXTEND
        batch.extend_lens = [r.extend_len for r in batch.reqs]
        batch.prefix_lens = [r.prefix_len for r in batch.reqs]
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.input_ids = torch.zeros(batch.extend_num_tokens, dtype=torch.int64)
        batch.seq_lens = torch.tensor([r.extend_end for r in batch.reqs])
        batch.seq_lens_cpu = batch.seq_lens
        batch.seq_lens_sum = int(batch.seq_lens.sum())
        stats = batch.prefill_stats
        prefill_forwards = []
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
            batch.forward_mode = mode
            prefill_forwards.append(
                ForwardBatch.init_new(
                    batch,
                    scheduler.tp_worker.model_runner,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    return_hidden_states_before_norm=False,
                )
            )

        # Model the prior forward's output and isolate only KV allocation.
        batch.forward_mode = ForwardMode.EXTEND
        batch.input_ids = torch.zeros(len(batch.reqs), dtype=torch.int64)
        batch.orig_seq_lens = batch.seq_lens.clone()
        batch.sampling_info = SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        )
        for req in batch.reqs:
            req.decode_batch_idx = 0
        with patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.zeros(len(batch.reqs), dtype=torch.int64),
        ):
            batch.prepare_for_decode()
        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertIs(batch.prefill_stats, stats)
        decode = ForwardBatch.init_new(
            batch,
            scheduler.tp_worker.model_runner,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            return_hidden_states_before_norm=False,
        )
        decode_shape = _snapshot_shape(forward_batch=decode, split_forward_count=1)
        self.assertEqual(decode_shape.forward_mode, "DECODE")
        self.assertIsNone(decode_shape.waiting_reqs_before)
        self.assertIsNone(decode_shape.waiting_reqs_after)
        self.assertIs(batch.prefill_stats, stats)
        self.assertEqual(
            (stats.waiting_reqs_before, stats.waiting_reqs_after), expected
        )
        for forward in prefill_forwards:
            shape = _snapshot_shape(forward_batch=forward, split_forward_count=1)
            self.assertEqual(
                (shape.waiting_reqs_before, shape.waiting_reqs_after), expected
            )

    def test_storage_prefetch_fulfillment_resolves_at_admission(self):
        adder = self.create_adder(self.create_running_batch())
        req = self.create_mock_req("storage-hit", priority=0, max_new_tokens=1)
        req.host_hit_length = 12
        req.host_loaded_length = 4
        req.storage_hit_length = 8
        req.storage_hit_start = 4
        req.materialized_host_hit_len.return_value = 4
        req.fulfilled_storage_hit_len.return_value = 8
        req.needs_host_load_back.return_value = True

        adder._account_prefill_cache_admission(req, prefix_len=12)

        self.mock_tree_cache.finish_storage_prefetch_admission.assert_called_once_with(
            "storage-hit",
            fulfilled_tokens=8,
            reason=None,
        )
        self.assertEqual(adder.log_device_hit_tokens, 4)
        self.assertEqual(adder.log_host_hit_tokens, 0)
        self.assertEqual(adder.log_storage_hit_tokens, 8)

        self.mock_tree_cache.finish_storage_prefetch_admission.reset_mock()
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        adder._account_prefill_cache_admission(req, prefix_len=0)
        self.mock_tree_cache.finish_storage_prefetch_admission.assert_called_once_with(
            "storage-hit", fulfilled_tokens=0, reason="device_capacity"
        )

    def test_retracted_storage_prefetch_accounting_is_omitted(self):
        adder = self.create_adder(self.create_running_batch())
        req = self.create_mock_req(
            "retracted-storage-hit", priority=0, max_new_tokens=1
        )
        req.retracted_stain = True

        adder._account_prefill_cache_admission(req, prefix_len=8)

        self.mock_tree_cache.discard_storage_prefetch_accounting.assert_called_once_with(
            "retracted-storage-hit"
        )
        self.mock_tree_cache.finish_storage_prefetch_admission.assert_not_called()

    def test_preempt_success_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=False)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req)

        self.assertTrue(success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)  # 50 + 75 + 100 - 50 = 175
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req)

        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 125)  # 50 + 75 + 100 - 100 = 125
        running_batch.release_req.assert_called_once()

    def test_preempt_fail_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=2, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_fail_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=False)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=0, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=-1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_skip_already_preempted_request(self):
        params = [
            ("req_prio_0", 0, 50),
            ("req_prio_1", 1, 75),
            ("req_prio_2", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=False)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225

        # New request preempts req_prio_0
        first_req = self.create_mock_req(
            "new_req_prio_1", priority=1, max_new_tokens=49
        )
        first_success = adder.preempt_to_schedule(first_req)
        self.assertTrue(first_success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)
        running_batch.release_req.assert_called_once()

        # Second call needs more tokens than currently free, so it would need to
        # preempt req_prio_0 again if already-preempted requests were not filtered out.
        second_req = self.create_mock_req(
            "second_new_req_prio_1", priority=1, max_new_tokens=76
        )
        second_success = adder.preempt_to_schedule(second_req)

        self.assertFalse(second_success)
        self.assertEqual(adder.rem_total_token_offset, 175)
        self.assertEqual(adder.preempt_list.count(running_reqs[0]), 1)
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_once(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=75)

        success = adder.preempt_to_schedule(new_req)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 375
        )  # 50 + 75 + 100 + 125 + 125 - 100 = 375
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_twice(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        self.scheduling_order(schedule_low_priority_values_first=True)
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=200)

        success = adder.preempt_to_schedule(new_req)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertIn(running_reqs[3], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 250
        )  # 50 + 75 + 100 + 125 + 125 - 100 - 125 = 250
        self.assertEqual(running_batch.release_req.call_count, 2)

    def test_mixed_chunk_prefill_budgets(self):
        self.mock_token_allocator.available_size.return_value = 1000

        decode_reqs = [
            self.create_mock_req(f"decode_{i}", priority=0, max_new_tokens=50)
            for i in range(8)
        ]
        running_batch = self.create_running_batch(decode_reqs)

        adder = self.create_adder(
            running_batch,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(decode_reqs),
        )

        self.assertEqual(adder.rem_input_tokens, 192)  # 200 - 8
        self.assertEqual(adder.rem_chunk_tokens, 56)  # 64 - 8
        self.assertEqual(adder.rem_total_token_offset, 408)  # 8 + 8 * 50
        self.assertEqual(adder.cur_rem_token_offset, 8)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        # Add a prefill that exactly consumes the chunk budget
        req1 = self.create_mock_req("req1", priority=0, max_new_tokens=64)
        req1.host_hit_length = 0
        req1.prefix_indices = []
        req1.full_untruncated_fill_ids = list(range(56))
        req1.last_node = MagicMock()
        req1.sampling_params.ignore_eos = False
        # add_one_req reads req.extend_range.length after set_extend_range;
        # emulate the real Req writer (a spec=Req mock lacks the attribute).
        req1.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req1, "extend_range", Range(start, end)
            )
        )

        result1 = adder.add_one_req(
            req1, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(adder.rem_chunk_tokens, 0)  # 56 - 56
        self.assertEqual(adder.rem_input_tokens, 136)  # 192 - 56
        self.assertEqual(result1, AddReqResult.OTHER)

        # 3 decode requests finished
        remaining_decode_reqs = decode_reqs[3:]
        running_batch2 = self.create_running_batch(remaining_decode_reqs)

        adder2 = self.create_adder(
            running_batch2,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(remaining_decode_reqs),
        )

        self.assertEqual(adder2.rem_input_tokens, 195)  # 200 - 5
        self.assertEqual(adder2.rem_chunk_tokens, 59)  # 64 - 5
        self.assertEqual(adder2.rem_total_token_offset, 255)  # 5 + 5 * 50
        self.assertEqual(adder2.budget_state(), AddReqResult.CONTINUE)

        # Same prefill no longer exhausts the chunk budget
        req2 = self.create_mock_req("req2", priority=0, max_new_tokens=64)
        req2.host_hit_length = 0
        req2.prefix_indices = []
        req2.full_untruncated_fill_ids = list(range(56))
        req2.last_node = MagicMock()
        req2.sampling_params.ignore_eos = False
        req2.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req2, "extend_range", Range(start, end)
            )
        )

        result2 = adder2.add_one_req(
            req2, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 1)
        self.assertEqual(adder2.rem_chunk_tokens, 3)  # 59 - 56 = 3 remaining
        self.assertEqual(result2, AddReqResult.CONTINUE)

        # Fit last small prefill request
        req3 = self.create_mock_req("req3", priority=0, max_new_tokens=16)
        req3.host_hit_length = 0
        req3.prefix_indices = []
        req3.full_untruncated_fill_ids = list(range(3))
        req3.last_node = MagicMock()
        req3.sampling_params.ignore_eos = False
        req3.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req3, "extend_range", Range(start, end)
            )
        )

        result3 = adder2.add_one_req(
            req3, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 2)
        self.assertEqual(adder2.rem_chunk_tokens, 0)  # 3 - 3 = 0
        self.assertEqual(result3, AddReqResult.OTHER)

    def _build_hybrid_swa_chunked_req(
        self,
        *,
        page_size,
        rem_swa,
        rem_chunk=2048,
        extend_input_len=500,
        is_hybrid_swa=True,
        full_available=100_000,
    ):
        self.mock_token_allocator.swa_available_size.return_value = rem_swa
        self.mock_token_allocator.full_available_size.return_value = full_available
        self.mock_token_allocator.available_size.return_value = full_available
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=rem_chunk,
        )
        adder.is_hybrid_swa = is_hybrid_swa

        req = self.create_mock_req("chunked", priority=0, max_new_tokens=128)
        req.prefix_indices = []
        req.full_untruncated_fill_ids = list(range(extend_input_len))
        # set_extend_range is the only writer of extend_range; the production
        # path reads req.extend_range.length right after calling it, so the mock
        # must actually set the attribute (a spec=Req mock has the method but
        # not the instance attribute).
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        return adder, req

    def test_add_chunked_req_hybrid_swa_reserves_page_for_alloc_extend(self):
        # alloc_extend needs extend_num_tokens + page_size per request. If the
        # scheduler hands out all of rem_swa_tokens, alloc_extend cannot get its
        # extra page and OOMs. With the fix, extend_input_len must cap at
        # rem_swa_tokens - page_size so the page is reserved.
        PAGE_SIZE = 64
        REM_SWA = 100
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=REM_SWA
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)  # truncated → chunked prefill continues
        req.set_extend_range.assert_called_once()
        start, end = req.set_extend_range.call_args.args
        new_len = end - start
        self.assertLessEqual(new_len + PAGE_SIZE, REM_SWA)
        self.assertEqual(new_len, REM_SWA - PAGE_SIZE)

    def test_add_chunked_req_hybrid_swa_defers_when_swa_below_page(self):
        # When rem_swa_tokens <= page_size there is no room to serve even the
        # reservation, so the chunked req must be deferred (returned unchanged)
        # instead of falling back to rem_chunk_tokens and bypassing SWA budget.
        PAGE_SIZE = 64
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=PAGE_SIZE
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)
        req.set_extend_range.assert_not_called()
        self.assertEqual(len(adder.can_run_list), 0)

    def test_swa_budget_for_req(self):
        # budget = max(alloc - window, 0) + min(extend + max_new, window) + page,
        # where alloc = min(extend, rem_chunk). The decode headroom is the SWA the
        # request adds to its own sliding window (extend + decode), capped at the
        # window -- a cached prefix funds the rest -- rather than a constant
        # window, which over-charged short requests into an admission livelock.
        cases = [
            # (extend, max_new, rem_chunk, window, page, expected, label)
            (64, 512, None, 128, 16, 128 + 16, "long_decode_hits_window_floor"),
            (64, 32, None, 128, 16, 96 + 16, "short_req_reserves_below_window"),
            (10, 20, None, 512, 8, 30 + 8, "short_resume_tiny_budget"),
            (300, 512, None, 128, 16, 300 + 16, "big_extend_over_window"),
            (200, 512, 50, 64, 8, 64 + 8, "chunk_capped_alloc_hits_floor"),
            # Multi-chunk: alloc = rem_chunk (1024) caps a huge extend, and the
            # chunk itself exceeds the window, so term1 = alloc - window is driven
            # by the chunk, not the full extend -> budget = chunk + page.
            (2000, 256, 1024, 512, 16, 1024 + 16, "multichunk_alloc_over_window"),
            (0, 512, None, 128, 16, 128 + 16, "extend_zero_long_decode"),
            (0, 40, None, 128, 16, 40 + 16, "extend_zero_short_decode"),
        ]
        for extend, max_new, rem_chunk, window, page, expected, label in cases:
            with self.subTest(label=label):
                self.mock_tree_cache.sliding_window_size = window
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=page,
                    rem_chunk_tokens=rem_chunk,
                )
                self.assertEqual(adder._swa_budget_for_req(extend, max_new), expected)

    def test_swa_admission_admits_short_cached_resume_at_two_window_pool(self):
        # Livelock regression (real incident). At an SWA pool ~= 2 sliding
        # windows, a cached-prefix resume matches >= 1 window (locked, excluded
        # from rem_swa) and has only a short uncached tail + a little decode
        # left. The pre-fix constant-window reservation charged a second full
        # window, so the admission gate (swa_needed >= rem_swa_tokens) rejected
        # it every scheduler iteration while LPM kept it at the queue head --
        # 100% scheduler CPU, idle GPU. Capping the reservation at
        # min(extend + decode, window) admits it.
        WINDOW, PAGE, REM_SWA = 128, 8, 100
        PREFIX, EXTEND = 200, 16  # cached prefix > window; short uncached tail
        self.mock_token_allocator.swa_available_size.return_value = REM_SWA
        self.mock_token_allocator.full_available_size.return_value = 100_000
        self.mock_token_allocator.available_size.return_value = 100_000
        self.mock_tree_cache.sliding_window_size = WINDOW
        self.mock_tree_cache.is_tree_cache.return_value = False
        adder = self.create_adder(self.create_running_batch(), page_size=PAGE)
        adder.is_hybrid_swa = True

        req = self.create_mock_req(
            "resume", priority=0, max_new_tokens=40, output_len=10
        )
        req.prefix_indices = list(range(PREFIX))
        req.full_untruncated_fill_ids = list(range(PREFIX + EXTEND))
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.last_node = MagicMock()
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        req.sampling_params = SimpleNamespace(max_new_tokens=40, ignore_eos=False)

        # Pre-fix: a constant sliding-window reservation rejects the resume.
        with patch.object(adder, "_swa_reserved_tokens", return_value=WINDOW + PAGE):
            self.assertIs(
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                ),
                AddReqResult.NO_TOKEN,
            )
        self.assertEqual(len(adder.can_run_list), 0)

        # Fix: min(extend + decode, window) reservation admits it.
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIn(req, adder.can_run_list)

    def test_swa_new_tokens_clamps_remaining_not_total(self):
        # Remaining decode headroom must be min(max_new - generated, CLIP)
        # (subtract-then-clip). The reversed order (clip-then-subtract) zeroes
        # out a request that has already generated >= CLIP tokens but still has a
        # long decode ahead, under-reserving its SWA window -> OOM risk on resume
        # of a long-generation request.
        from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS as CLIP

        adder = self.create_adder(self.create_running_batch())
        cases = [
            # (max_new, generated, expected, label)
            (100, 10, 90, "below_clip_normal"),
            (40, 100, 0, "already_finished"),
            # clip-then-subtract would give 3996; subtract-then-clip caps at CLIP.
            (CLIP + 6000, 100, CLIP, "long_gen_small_output_caps_at_clip"),
            # clip-then-subtract would give 0; the long decode still needs CLIP.
            (CLIP + 6000, CLIP + 100, CLIP, "long_gen_output_over_clip"),
        ]
        for max_new, generated, expected, label in cases:
            with self.subTest(label=label):
                req = self.create_mock_req(
                    label, priority=0, max_new_tokens=max_new, output_len=generated
                )
                self.assertEqual(adder._swa_new_tokens(req), expected)

    def test_delayer_not_consulted_when_kv_budget_rejects(self):
        """A rank whose first candidate fails the KV-budget gate must NOT
        report local_prefillable=True: add_one_req returns NO_TOKEN before
        negotiating, and finalize() later reports the rank as not
        prefillable. Regression guard: the negotiate used to run at the top
        of add_one_req, so under KV pressure a full rank still claimed
        prefillable=True, the delayer saw "all prefillable" and allowed, the
        full ranks then NO_TOKEN'ed out, and the DP-synced forward mixed
        prefill and decode — the exact pattern the delayer exists to
        prevent."""
        delayer = _RecordingDelayer(allow=True)
        adder = self._create_delayer_adder(available_tokens=10, delayer=delayer)

        result = adder.add_one_req(
            self._create_delayer_req(50),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(delayer.calls, [])
        self.assertEqual(adder.can_run_list, [])

    def test_delayer_not_consulted_when_post_lock_recheck_rejects(self):
        """Locking the request's own prefix converts evictable tokens into
        protected ones, so a request can pass the pre-lock KV gate yet fail
        the post-lock recheck — precisely the high-utilization regime the
        delayer targets. The negotiate must sit after that recheck too, or
        the rank claims prefillable=True and then runs decode."""
        delayer = _RecordingDelayer(allow=True)
        adder = self._create_delayer_adder(available_tokens=10, delayer=delayer)
        # Pre-lock budget is covered by evictable tokens; inc_lock_ref pins
        # them (evictable -> protected), shrinking the budget below demand.
        self.mock_tree_cache.evictable_size.return_value = 1000
        self.mock_tree_cache.full_evictable_size.return_value = 1000

        def _pin_prefix(node):
            self.mock_tree_cache.evictable_size.return_value = 0
            self.mock_tree_cache.full_evictable_size.return_value = 0
            return IncLockRefResult()

        self.mock_tree_cache.inc_lock_ref.side_effect = _pin_prefix

        result = adder.add_one_req(
            self._create_delayer_req(50),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(delayer.calls, [])
        self.assertEqual(adder.can_run_list, [])

    def test_delay_verdict_blocks_admissible_request(self):
        """An admissible request must still be gated by the (relocated)
        negotiate: on a delay verdict it is not admitted, and the rank
        reported prefillable=True exactly once."""
        delayer = _RecordingDelayer(allow=False)
        adder = self._create_delayer_adder(available_tokens=100_000, delayer=delayer)

        result = adder.add_one_req(
            self._create_delayer_req(50),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(delayer.calls, [True])
        self.assertEqual(adder.can_run_list, [])

    def test_allow_verdict_admits_request(self):
        """On an allow verdict the request proceeds through admission — the
        relocated negotiate must not block the commit path."""
        delayer = _RecordingDelayer(allow=True)
        adder = self._create_delayer_adder(available_tokens=100_000, delayer=delayer)
        req = self._create_delayer_req(50)

        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(delayer.calls, [True])
        self.assertIn(req, adder.can_run_list)

    def test_chunked_req_negotiates_prefillable_and_proceeds(self):
        """A rank resuming a chunked prefill runs it this pass regardless of
        the verdict, so add_chunked_req must report prefillable=True (else a
        rank with an empty waiting queue reports False via finalize() and
        peers delay while it prefills alone) and must not drop the chunk on
        a delay verdict (that would leak memory)."""
        delayer = _RecordingDelayer(allow=False)
        adder = self._create_delayer_adder(
            available_tokens=100_000, delayer=delayer, rem_chunk_tokens=500
        )
        req = self._create_delayer_req(200)

        result = adder.add_chunked_req(req)

        self.assertIsNone(result)  # chunk fully admitted, not truncated
        self.assertEqual(delayer.calls, [True])
        self.assertIn(req, adder.can_run_list)

    def _create_delayer_adder(self, *, available_tokens, delayer, **kwargs):
        self.mock_token_allocator.available_size.return_value = available_tokens
        self.mock_token_allocator.full_available_size.return_value = available_tokens
        return self.create_adder(
            self.create_running_batch(),
            prefill_delayer_single_pass=delayer,
            **kwargs,
        )

    def _create_delayer_req(self, num_tokens: int):
        req = self.create_mock_req("delayer_req", priority=0, max_new_tokens=8)
        req.full_untruncated_fill_ids = list(range(num_tokens))
        req.host_hit_length = 0
        req.last_node = MagicMock()
        req.sampling_params.ignore_eos = False
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        return req

    def test_add_chunked_req_non_hybrid_no_swa_reservation(self):
        # Non-hybrid path: the SWA-pool reservation must NOT apply, otherwise
        # the fix would regress non-SWA models.
        PAGE_SIZE = 16
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE,
            rem_swa=10,
            rem_chunk=500,
            extend_input_len=200,
            is_hybrid_swa=False,
            full_available=300,
        )

        result = adder.add_chunked_req(req)
        self.assertIsNone(result)
        req.set_extend_range.assert_called_once_with(0, 200)
        self.assertIn(req, adder.can_run_list)

    def _adder_with_extend_lens(self, extend_lens):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.can_run_list = [
            SimpleNamespace(extend_input_len=length) for length in extend_lens
        ]
        # BLOCK_M is auto-detected from the attention backend in production; the
        # __new__ helper bypasses __init__, so set it explicitly. 64 matches the
        # block_m the tile-count assertions below are computed against.
        adder.prefill_tile_block_m = 64
        return adder

    def test_estimate_prefill_extend_tile_metrics(self):
        metrics = estimate_prefill_extend_tile_metrics([1, 7, 13, 129], block_m=64)

        self.assertEqual(metrics["q_tiles_per_request"], [1, 1, 1, 3])
        self.assertEqual(metrics["legacy_q_tiles_per_head"], 12)
        self.assertEqual(metrics["compact_q_tiles_per_head"], 6)
        self.assertEqual(metrics["saved_q_tiles_per_head"], 6)
        self.assertEqual(metrics["saved_q_tile_ratio"], 0.5)

    def test_compact_prefill_tile_budget_admits_more_than_legacy(self):
        adder = self._adder_with_extend_lens([1, 7, 13])

        # The tile-budget admission is gated on HIP in production; force the gate
        # on so this vendor-neutral admission-math check runs on any CI runner.
        with (
            patch.object(schedule_policy, "_IS_HIP", True),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 6),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET_MODE", "compact"),
        ):
            self.assertIsNone(adder._check_prefill_tile_budget(129))

        with (
            patch.object(schedule_policy, "_IS_HIP", True),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 6),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET_MODE", "legacy"),
        ):
            self.assertEqual(adder._check_prefill_tile_budget(129), AddReqResult.OTHER)

    def test_prefill_tile_budget_always_allows_first_request(self):
        adder = self._adder_with_extend_lens([])

        with (
            patch.object(schedule_policy, "_IS_HIP", True),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 1),
        ):
            self.assertIsNone(adder._check_prefill_tile_budget(4096))

    def test_prefill_tile_budget_disabled_on_non_hip(self):
        # AMD-only: on non-HIP vendors the tile-budget admission must be a no-op
        # even when the budget env is set, so scheduler behavior is unchanged.
        adder = self._adder_with_extend_lens([1, 7, 13])

        with (
            patch.object(schedule_policy, "_IS_HIP", False),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET", 6),
            patch.object(schedule_policy, "PREFILL_TILE_BUDGET_MODE", "legacy"),
        ):
            self.assertIsNone(adder._check_prefill_tile_budget(129))

    # ---- _swa_req_never_fits: the gate on the _swa_chunk_cap escape hatch ----
    # The hatch shrinks a chunk and admits it instead of waiting; it must fire
    # ONLY for a request whose SWA budget can never fit the drained pool (true
    # head-of-line livelock). Admitting transient-pressure requests instead
    # collapses the SWA evictable cushion and causes a retraction storm.

    def create_swa_adder(
        self, *, size_swa: int, sliding_window: int, page_size: int = 16
    ) -> PrefillAdder:
        self.mock_tree_cache.sliding_window_size = sliding_window
        self.mock_token_allocator = self.create_token_allocator(size_swa=size_swa)
        return self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=512,
        )

    def test_swa_never_fits_false_under_transient_pressure(self):
        # Small request against an ample pool: budget << pool, so it would fit
        # once running decodes drain -> must wait, not take the hatch.
        adder = self.create_swa_adder(size_swa=1024, sliding_window=128)
        self.assertFalse(
            adder._swa_req_never_fits(extend_input_len=256, max_new_tokens=64)
        )

    def test_swa_never_fits_true_when_budget_exceeds_whole_pool(self):
        # A large host-hit load-back charge pushes the budget past the entire
        # pool: it can never fit however far the pool drains -> hatch.
        adder = self.create_swa_adder(size_swa=1024, sliding_window=128)
        self.assertTrue(
            adder._swa_req_never_fits(
                extend_input_len=256, max_new_tokens=64, swa_host_hit_length=4096
            )
        )

    def test_swa_never_fits_is_gated_by_pool_capacity(self):
        # Same request; only the pool size changes. Proves the check compares
        # the budget against size_swa (guards against a wrong-accessor bug).
        req = dict(extend_input_len=256, max_new_tokens=64, swa_host_hit_length=600)
        self.assertTrue(
            self.create_swa_adder(size_swa=512, sliding_window=128)._swa_req_never_fits(
                **req
            )
        )
        self.assertFalse(
            self.create_swa_adder(
                size_swa=4096, sliding_window=128
            )._swa_req_never_fits(**req)
        )


if __name__ == "__main__":
    unittest.main()
