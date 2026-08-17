import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.base_spec_worker import PreparedDraftWeightUpdate
from sglang.srt.weight_sync.draft_tensor_preshard import (
    preshard_dflash_named_tensors,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_cpu_updater(model):
    runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=MagicMock(),
        recapture_cuda_graph=MagicMock(),
        get_model_runner=lambda: runner,
    )


class TestDraftTensorPreshard(unittest.TestCase):
    def test_compact_rank_local_storage_and_filtering(self):
        q = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        down = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        norm = torch.arange(4, dtype=torch.float32)
        shared = torch.ones(16, 4)

        ranks, info = preshard_dflash_named_tensors(
            [
                ("layers.0.self_attn.q_proj.weight", q),
                ("layers.0.mlp.down_proj.weight", down),
                ("norm.weight", norm),
                ("embed_tokens.weight", shared),
            ],
            tp_size=2,
        )

        self.assertEqual(info["sharded_tensor_count"], 2)
        self.assertEqual(info["replicated_tensor_count"], 1)
        self.assertEqual(info["per_rank_tensor_count"], [3, 3])
        for rank, items in enumerate(ranks):
            payload = dict(items)
            self.assertNotIn("embed_tokens.weight", payload)
            torch.testing.assert_close(
                payload["layers.0.self_attn.q_proj.weight"],
                q.narrow(0, rank * 4, 4),
            )
            torch.testing.assert_close(
                payload["layers.0.mlp.down_proj.weight"],
                down.narrow(1, rank * 4, 4),
            )
            self.assertEqual(
                payload["layers.0.self_attn.q_proj.weight"].untyped_storage().nbytes(),
                4 * 4 * q.element_size(),
            )
            self.assertEqual(
                payload["layers.0.mlp.down_proj.weight"].untyped_storage().nbytes(),
                4 * 4 * down.element_size(),
            )

    def test_auxiliary_head_projections_are_never_sharded(self):
        # GatedMarkovHead/RNNHead params contain ".gate_proj."/".up_proj."-like
        # substrings but are replicated plain Linear modules; sharding them
        # would corrupt every rank at tp>=2.
        markov_gate = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        confidence = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        layer_gate = torch.arange(32, dtype=torch.float32).reshape(8, 4)

        ranks, info = preshard_dflash_named_tensors(
            [
                ("markov_head.gate_proj.weight", markov_gate),
                ("confidence_head.proj.weight", confidence),
                ("layers.0.mlp.gate_proj.weight", layer_gate),
            ],
            tp_size=2,
        )

        self.assertEqual(info["sharded_tensor_count"], 1)
        self.assertEqual(info["replicated_tensor_count"], 2)
        for rank, items in enumerate(ranks):
            payload = dict(items)
            torch.testing.assert_close(
                payload["markov_head.gate_proj.weight"], markov_gate
            )
            torch.testing.assert_close(
                payload["confidence_head.proj.weight"], confidence
            )
            torch.testing.assert_close(
                payload["layers.0.mlp.gate_proj.weight"],
                layer_gate.narrow(0, rank * 4, 4),
            )

    def test_model_prefixed_layer_projections_are_sharded(self):
        q = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        ranks, info = preshard_dflash_named_tensors(
            [("model.layers.0.self_attn.q_proj.weight", q)], tp_size=2
        )
        self.assertEqual(info["sharded_tensor_count"], 1)
        for rank, items in enumerate(ranks):
            torch.testing.assert_close(
                dict(items)["model.layers.0.self_attn.q_proj.weight"],
                q.narrow(0, rank * 4, 4),
            )


class TestFailClosedLowLevelUpdate(unittest.TestCase):
    def test_validation_failure_touches_no_model_tensor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loaded = False

            def validate_weights(self, weights, **_):
                list(weights)
                raise ValueError("shape rejected before apply")

            def load_weights(self, weights):
                self.loaded = True
                list(weights)

        model = Model()
        status = {}
        success, message = make_cpu_updater(model).update_weights_from_tensor(
            [("weight", torch.ones(2))],
            stream_tensors=True,
            update_status=status,
        )

        self.assertFalse(success)
        self.assertIn("shape rejected before apply", message)
        self.assertFalse(status["partial_update"])
        self.assertFalse(model.loaded)

    def test_mid_apply_failure_is_reported_partial(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loaded = []

            def load_weights(self, weights):
                for name, tensor in weights:
                    self.loaded.append((name, tensor.clone()))

        model = Model()
        status = {}
        phases = {}
        with patch.dict(
            os.environ,
            {"SGLANG_ENABLE_WEIGHT_UPDATE_FAULT_INJECTION": "1"},
        ):
            success, message = make_cpu_updater(model).update_weights_from_tensor(
                [("a", torch.ones(2)), ("b", torch.zeros(2))],
                stream_tensors=True,
                fault_injection_after_tensors=1,
                phase_timings_ms=phases,
                update_status=status,
            )

        self.assertFalse(success)
        self.assertIn("injected mid-update failure", message)
        self.assertTrue(status["partial_update"])
        self.assertEqual([name for name, _ in model.loaded], ["a"])
        self.assertIn("apply_total_ms", phases)

    def test_presharded_loader_mode_is_scoped_to_validation_and_apply(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Module()
                self.proj.use_presharded_weights = False
                self.seen = []

            def validate_weights(self, weights, **_):
                self.seen.append(("validate", self.proj.use_presharded_weights))
                list(weights)

            def load_weights(self, weights):
                self.seen.append(("apply", self.proj.use_presharded_weights))
                list(weights)

        model = Model()
        success, _ = make_cpu_updater(model).update_weights_from_tensor(
            [("weight", torch.ones(2))],
            stream_tensors=True,
            tensors_are_pre_sharded=True,
        )

        self.assertTrue(success)
        self.assertEqual(model.seen, [("validate", True), ("apply", True)])
        self.assertFalse(model.proj.use_presharded_weights)

    def test_dflash_meta_preflight_rejects_shape_dtype_and_unknown_name(self):
        model = DFlashDraftModel.__new__(DFlashDraftModel)
        torch.nn.Module.__init__(model)
        model.fc = torch.nn.Linear(4, 2, bias=False)
        model.num_context_features = 2
        model.config = SimpleNamespace(hidden_size=2)

        with self.assertRaisesRegex(ValueError, "fc.weight shape mismatch"):
            model.validate_weights([("fc.weight", torch.ones(3, 4))])
        with self.assertRaisesRegex(TypeError, "dtype"):
            model.validate_weights([("fc.weight", torch.ones(2, 4, dtype=torch.int64))])
        with self.assertRaisesRegex(ValueError, "unexpected DFlash weight"):
            model.validate_weights([("not_a_weight", torch.ones(1))])


class TestSchedulerCPUStaging(unittest.TestCase):
    @patch(
        "sglang.srt.managers.scheduler_components.weight_updater."
        "torch.distributed.barrier"
    )
    def test_stage_status_commit_is_rank_local_and_observable(self, barrier):
        worker = MagicMock()
        staged = PreparedDraftWeightUpdate(
            runner_updates=[object()],
            phase_timings_ms={"deserialize_ms": 2.0, "host_pin_ms": 4.0},
            host_memory_bytes={"pinned_tensor_bytes": 128},
        )
        worker.prepare_weights_from_tensor.return_value = staged

        def apply(_, __, *, phase_timings_ms, update_status):
            phase_timings_ms.update({"gpu_h2d_ms": 7.0, "gpu_param_copy_ms": 3.0})
            update_status["partial_update"] = False
            return True, "ok"

        worker.apply_prepared_weights_from_tensor.side_effect = apply
        manager = SchedulerWeightUpdaterManager(
            tp_worker=MagicMock(),
            draft_worker=worker,
            tp_cpu_group=MagicMock(),
            memory_saver_adapter=MagicMock(),
            flush_cache=MagicMock(),
            is_fully_idle=MagicMock(return_value=True),
        )
        try:
            stage_req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[b"rank-0"],
                draft_only=True,
                operation="stage",
                update_id="u1",
                pin_memory=True,
            )
            stage_output = manager.update_weights_from_tensor(stage_req)
            self.assertEqual(stage_output.staging_state, "pending")
            manager.staged_tensor_updates["u1"].future.result(timeout=5)

            status_req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[],
                draft_only=True,
                operation="status",
                update_id="u1",
            )
            status_output = manager.update_weights_from_tensor(status_req)
            self.assertEqual(status_output.staging_state, "ready")
            self.assertEqual(status_output.phase_timings_ms["host_pin_ms"], 4.0)

            commit_req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[],
                draft_only=True,
                operation="commit",
                update_id="u1",
            )
            commit_output = manager.update_weights_from_tensor(commit_req)
            self.assertTrue(commit_output.success)
            self.assertEqual(commit_output.phase_timings_ms["gpu_h2d_ms"], 7.0)
            self.assertNotIn("u1", manager.staged_tensor_updates)
            barrier.assert_called_once_with(group=manager.tp_cpu_group)
        finally:
            manager.tensor_stage_executor.shutdown(wait=True)


class TestAtomicFailClosedController(unittest.IsolatedAsyncioTestCase):
    def make_manager(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(dp_size=1, enable_dp_attention=False)
        manager.auto_create_handle_loop = MagicMock()
        manager.abort_request = MagicMock()
        manager.is_pause = False
        manager.is_pause_cond = asyncio.Condition()
        manager.draft_weight_update_unhealthy = False
        manager.draft_weight_update_unhealthy_reason = None
        manager.draft_weight_update_unhealthy_id = None
        manager.atomic_tensor_update_lock = asyncio.Lock()
        manager.atomic_tensor_update_tasks = set()
        manager._update_weight_version_if_provided = MagicMock()
        manager.pause_generation = AsyncMock(
            side_effect=lambda _: setattr(manager, "is_pause", True)
        )

        async def continue_generation(_):
            self.assertFalse(manager.draft_weight_update_unhealthy)
            manager.is_pause = False
            return True, "continued"

        manager.continue_generation = AsyncMock(side_effect=continue_generation)
        state = {"commit_success": False}

        async def communicate(req):
            if req.operation == "stage":
                output = UpdateWeightsFromTensorReqOutput(
                    success=True,
                    message="staging",
                    update_id=req.update_id,
                    staging_state="pending",
                )
            elif req.operation == "status":
                output = UpdateWeightsFromTensorReqOutput(
                    success=True,
                    message="ready",
                    update_id=req.update_id,
                    staging_state="ready",
                    phase_timings_ms={"host_pin_ms": 5.0},
                )
            elif req.operation == "commit":
                output = UpdateWeightsFromTensorReqOutput(
                    success=state["commit_success"],
                    message=("committed" if state["commit_success"] else "copy failed"),
                    update_id=req.update_id,
                    partial_update=not state["commit_success"],
                    phase_timings_ms={"gpu_h2d_ms": 8.0},
                )
            else:
                output = UpdateWeightsFromTensorReqOutput(
                    success=True, message="discarded", update_id=req.update_id
                )
            return [output]

        manager.update_weights_from_tensor_communicator = communicate
        return manager, state

    async def test_commit_failure_stays_down_until_explicit_restore(self):
        manager, state = self.make_manager()
        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"payload"],
            draft_only=True,
            weight_version="bad-version",
        )

        failed = await manager.update_weights_from_tensor_atomic(request)

        self.assertFalse(failed.success)
        self.assertTrue(failed.partial_update)
        self.assertTrue(failed.unhealthy)
        self.assertTrue(manager.is_pause)
        manager.continue_generation.assert_not_called()
        manager._update_weight_version_if_provided.assert_not_called()

        state["commit_success"] = True
        recovery = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"original"],
            draft_only=True,
            recovery=True,
            weight_version="restored-version",
        )
        restored = await manager.update_weights_from_tensor_atomic(recovery)

        self.assertTrue(restored.success)
        self.assertFalse(restored.unhealthy)
        self.assertFalse(manager.is_pause)
        manager.continue_generation.assert_awaited_once()
        manager._update_weight_version_if_provided.assert_called_once_with(
            "restored-version"
        )

    async def test_continue_rejects_unhealthy_instance(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.draft_weight_update_unhealthy = True
        manager.draft_weight_update_unhealthy_reason = "partial copy"
        manager.is_pause_cond = asyncio.Condition()
        manager._async_dispatch_to_scheduler = AsyncMock()

        success, message = await TokenizerManager.continue_generation(
            manager, ContinueGenerationReqInput(torch_empty_cache=False)
        )

        self.assertFalse(success)
        self.assertIn("partial copy", message)
        manager._async_dispatch_to_scheduler.assert_not_called()

    async def test_legacy_update_waits_for_atomic_transaction_lock(self):
        manager, _ = self.make_manager()
        manager.is_pause = True
        manager.update_weights_from_tensor_communicator = AsyncMock(
            return_value=[
                UpdateWeightsFromTensorReqOutput(success=True, message="applied")
            ]
        )
        request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[b"payload"],
            draft_only=True,
        )

        await manager.atomic_tensor_update_lock.acquire()
        task = asyncio.create_task(manager.update_weights_from_tensor_detailed(request))
        try:
            await asyncio.sleep(0)
            manager.update_weights_from_tensor_communicator.assert_not_awaited()
        finally:
            manager.atomic_tensor_update_lock.release()

        result = await task
        self.assertTrue(result.success)
        manager.update_weights_from_tensor_communicator.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
