import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.ngram_worker import NGRAMWorker

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulerDraftOnlyWeightUpdate(unittest.TestCase):
    def _manager(self, *, with_draft=True):
        target = MagicMock()
        draft = MagicMock() if with_draft else None
        flush_cache = MagicMock(return_value=True)
        manager = SchedulerWeightUpdaterManager(
            tp_worker=target,
            draft_worker=draft,
            tp_cpu_group=MagicMock(),
            memory_saver_adapter=MagicMock(),
            flush_cache=flush_cache,
            is_fully_idle=MagicMock(return_value=True),
        )
        return manager, target, draft, flush_cache

    def test_disk_draft_only_skips_target_and_target_cache(self):
        manager, target, draft, flush_cache = self._manager()
        draft.update_weights_from_disk.return_value = (True, "draft updated")
        req = UpdateWeightFromDiskReqInput(
            model_path="/draft",
            draft_only=True,
            # Even with the legacy True default, target-owned cache is untouched.
            flush_cache=True,
        )

        output = manager.update_weights_from_disk(req)

        self.assertTrue(output.success)
        target.update_weights_from_disk.assert_not_called()
        draft.update_weights_from_disk.assert_called_once_with(req)
        flush_cache.assert_not_called()

    @patch(
        "sglang.srt.managers.scheduler_components.weight_updater."
        "torch.distributed.barrier"
    )
    def test_tensor_draft_only_skips_target_and_target_cache(self, barrier):
        manager, target, draft, flush_cache = self._manager()
        draft.update_weights_from_tensor.return_value = (True, "draft updated")
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"payload"],
            draft_only=True,
            flush_cache=True,
        )

        output = manager.update_weights_from_tensor(req)

        self.assertTrue(output.success)
        target.update_weights_from_tensor.assert_not_called()
        draft.update_weights_from_tensor.assert_called_once()
        call = draft.update_weights_from_tensor.call_args
        self.assertIs(call.args[0], req)
        self.assertEqual(set(call.kwargs), {"phase_timings_ms", "update_status"})
        flush_cache.assert_not_called()
        barrier.assert_called_once_with(group=manager.tp_cpu_group)

    @patch(
        "sglang.srt.managers.scheduler_components.weight_updater."
        "torch.distributed.barrier"
    )
    def test_tensor_rejects_conflicting_routing_flags(self, barrier):
        manager, target, draft, flush_cache = self._manager()
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"payload"],
            draft_only=True,
            disable_draft_model=True,
        )

        output = manager.update_weights_from_tensor(req)

        self.assertFalse(output.success)
        self.assertIn("mutually exclusive", output.message)
        target.update_weights_from_tensor.assert_not_called()
        draft.update_weights_from_tensor.assert_not_called()
        flush_cache.assert_not_called()
        barrier.assert_called_once_with(group=manager.tp_cpu_group)

    def test_disk_draft_only_requires_a_draft_model(self):
        manager, target, _, flush_cache = self._manager(with_draft=False)
        req = UpdateWeightFromDiskReqInput(
            model_path="/draft", draft_only=True, flush_cache=False
        )

        output = manager.update_weights_from_disk(req)

        self.assertFalse(output.success)
        self.assertIn("requires a speculative draft model", output.message)
        target.update_weights_from_disk.assert_not_called()
        flush_cache.assert_not_called()

    @patch(
        "sglang.srt.managers.scheduler_components.weight_updater."
        "torch.distributed.barrier"
    )
    def test_tensor_draft_only_requires_a_draft_model(self, barrier):
        manager, target, _, flush_cache = self._manager(with_draft=False)
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"payload"],
            draft_only=True,
            flush_cache=False,
        )

        output = manager.update_weights_from_tensor(req)

        self.assertFalse(output.success)
        self.assertIn("requires a speculative draft model", output.message)
        target.update_weights_from_tensor.assert_not_called()
        flush_cache.assert_not_called()
        barrier.assert_called_once_with(group=manager.tp_cpu_group)


class TestBaseSpecWorkerWeightUpdate(unittest.TestCase):
    def _worker(self):
        worker = BaseSpecWorker()
        spec_algorithm = MagicMock()
        spec_algorithm.is_ngram.return_value = False
        spec_algorithm.is_frozen_kv_mtp.return_value = False
        spec_algorithm.is_dflash_family.return_value = True

        target_updater = MagicMock()
        target_runner = SimpleNamespace(
            spec_algorithm=spec_algorithm,
            weight_updater=target_updater,
        )
        worker._target_worker = SimpleNamespace(model_runner=target_runner)

        draft_updater = MagicMock()
        draft_runner = SimpleNamespace(weight_updater=draft_updater)
        # DSpark/DFlash use a plain TpModelWorker without draft_runners.
        worker._draft_worker = SimpleNamespace(model_runner=draft_runner)
        worker.ps = SimpleNamespace(tp_rank=0)
        return worker, target_updater, draft_updater

    def test_disk_supports_plain_tp_draft_worker(self):
        worker, target_updater, draft_updater = self._worker()
        draft_updater.update_weights_from_disk.return_value = (True, "ok")
        req = UpdateWeightFromDiskReqInput(
            model_path="/draft",
            load_format="safetensors",
            recapture_cuda_graph=False,
            draft_only=True,
        )

        output = worker.update_weights_from_disk(req)

        self.assertEqual(output, (True, "Succeeded to update model weights."))
        draft_updater.update_weights_from_disk.assert_called_once_with(
            "/draft", "safetensors", recapture_cuda_graph=False
        )
        target_updater.update_weights_from_disk.assert_not_called()

    @patch("sglang.srt.speculative.base_spec_worker.monkey_patch_torch_reductions")
    @patch(
        "sglang.srt.speculative.base_spec_worker."
        "MultiprocessingSerializer.deserialize",
        return_value=[("draft.weight", "tensor")],
    )
    def test_tensor_draft_only_does_not_update_target(self, deserialize, _):
        worker, target_updater, draft_updater = self._worker()
        prepared = SimpleNamespace(
            phase_timings_ms={},
            host_memory_bytes={
                "source_tensor_bytes": 1,
                "staged_tensor_bytes": 1,
                "pinned_tensor_bytes": 0,
            },
        )
        draft_updater.prepare_weights_from_tensor.return_value = prepared
        draft_updater.apply_prepared_weights_from_tensor.return_value = (True, "ok")
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"rank-0"],
            load_format=None,
            draft_only=True,
        )

        output = worker.update_weights_from_tensor(req)

        self.assertEqual(output, (True, "Succeeded to update model weights."))
        deserialize.assert_called_once_with(b"rank-0")
        draft_updater.prepare_weights_from_tensor.assert_called_once_with(
            named_tensors=[("draft.weight", "tensor")],
            load_format=None,
            stream_tensors=True,
            tensors_are_pre_sharded=False,
            pin_memory=False,
            fault_injection_after_tensors=None,
            phase_timings_ms={},
        )
        draft_updater.apply_prepared_weights_from_tensor.assert_called_once()
        apply_call = draft_updater.apply_prepared_weights_from_tensor.call_args
        self.assertIs(apply_call.args[0], prepared)
        self.assertEqual(apply_call.kwargs["load_format"], None)
        self.assertTrue(apply_call.kwargs["stream_tensors"])
        self.assertFalse(apply_call.kwargs["collect_phase_timings"])
        target_updater.update_weights_from_tensor.assert_not_called()

    @patch("sglang.srt.speculative.base_spec_worker.monkey_patch_torch_reductions")
    @patch(
        "sglang.srt.speculative.base_spec_worker."
        "MultiprocessingSerializer.deserialize",
        side_effect=RuntimeError("bad CPU IPC handle"),
    )
    def test_tensor_deserialization_error_does_not_reach_updater(self, deserialize, _):
        worker, target_updater, draft_updater = self._worker()
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"rank-0"],
            load_format=None,
            draft_only=True,
        )

        success, message = worker.update_weights_from_tensor(req)

        self.assertFalse(success)
        self.assertIn("bad CPU IPC handle", message)
        deserialize.assert_called_once_with(b"rank-0")
        draft_updater.prepare_weights_from_tensor.assert_not_called()
        draft_updater.apply_prepared_weights_from_tensor.assert_not_called()
        target_updater.update_weights_from_tensor.assert_not_called()

    def test_tensor_default_route_preserves_target_worker_dispatch(self):
        worker, target_updater, draft_updater = self._worker()
        worker.target_worker.update_weights_from_tensor = MagicMock(
            return_value=(True, "target updated")
        )
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"rank-0"], draft_only=False
        )

        output = worker.update_weights_from_tensor(req)

        self.assertEqual(output, (True, "target updated"))
        worker.target_worker.update_weights_from_tensor.assert_called_once_with(req)
        draft_updater.update_weights_from_tensor.assert_not_called()
        target_updater.update_weights_from_tensor.assert_not_called()

    def test_ngram_draft_only_rejects_without_touching_target(self):
        target = MagicMock()
        worker = SimpleNamespace(target_worker=target)
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"rank-0"], draft_only=True
        )

        output = NGRAMWorker.update_weights_from_tensor(worker, req)

        self.assertFalse(output[0])
        self.assertIn("model-backed speculative draft", output[1])
        target.update_weights_from_tensor.assert_not_called()


class TestDraftOnlyPublicClient(unittest.TestCase):
    @patch(
        "sglang.srt.entrypoints.http_server_engine."
        "MultiprocessingSerializer.serialize",
        side_effect=["rank-0", "rank-1"],
    )
    def test_http_adapter_forwards_draft_only_with_safe_cpu_sharing(self, serialize):
        adapter = HttpServerEngineAdapter.__new__(HttpServerEngineAdapter)
        adapter.server_args = SimpleNamespace(tp_size=2)
        adapter._make_request = MagicMock(return_value={"success": True})
        tensors = [("draft.weight", "cpu-tensor")]

        result = adapter.update_weights_from_tensor(
            tensors, load_format=None, flush_cache=False, draft_only=True
        )

        self.assertEqual(result, {"success": True})
        self.assertEqual(serialize.call_count, 2)
        serialize.assert_any_call(
            tensors,
            output_str=True,
            cpu_sharing_strategy="file_system",
        )
        adapter._make_request.assert_called_once_with(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": ["rank-0", "rank-1"],
                "load_format": None,
                "flush_cache": False,
                "draft_only": True,
            },
        )


class TestTokenizerDraftOnlyModelPath(unittest.IsolatedAsyncioTestCase):
    async def test_successful_disk_update_preserves_target_model_path(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.elastic_worker_count = 1
        manager._update_model_path_info = MagicMock()

        def dispatch(_):
            manager.model_update_result.set_result(
                SimpleNamespace(
                    success=True,
                    message="ok",
                    num_paused_requests=0,
                )
            )

        manager._dispatch_to_scheduler = dispatch
        req = UpdateWeightFromDiskReqInput(model_path="/draft", draft_only=True)

        output = await manager._wait_for_model_update_from_disk(req)

        self.assertEqual(output, (True, "ok", 0))
        manager._update_model_path_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
