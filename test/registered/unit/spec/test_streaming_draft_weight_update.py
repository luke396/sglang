import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftMixin, DSparkDraftModel
from sglang.srt.utils import MultiprocessingSerializer

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStreamingWeightUpdater(unittest.TestCase):
    def test_filename_cpu_sharing_is_scoped_to_one_serialization(self):
        original_strategy = torch.multiprocessing.get_sharing_strategy()
        tensor = torch.arange(4, dtype=torch.float32)

        payload = MultiprocessingSerializer.serialize(
            [("weight", tensor)], cpu_sharing_strategy="file_system"
        )

        self.assertEqual(
            torch.multiprocessing.get_sharing_strategy(), original_strategy
        )
        self.assertIn(b"rebuild_storage_filename", payload)
        restored = MultiprocessingSerializer.deserialize(payload)
        torch.testing.assert_close(restored[0][1], tensor)

    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unsupported_derived_weight_cache_error",
        return_value=None,
    )
    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "_unwrap_tensor"
    )
    @patch(
        "sglang.srt.model_executor.model_runner_components.weight_updater."
        "torch.get_device_module"
    )
    def test_draft_tensors_materialize_only_as_loader_consumes_them(
        self, get_device_module, unwrap_tensor, _
    ):
        stream = MagicMock()
        device_module = SimpleNamespace(
            current_device=MagicMock(return_value=0),
            current_stream=MagicMock(return_value=stream),
        )
        get_device_module.return_value = device_module
        unwrap_tensor.side_effect = lambda tensor, **_: f"device-{tensor}"

        class SinglePassModel:
            def __init__(self):
                self.started_with_materialized = None
                self.materialized_counts = []
                self.loaded = []

            def should_materialize_weight(self, name):
                return name != "b"

            def load_weights(self, weights):
                self.started_with_materialized = unwrap_tensor.call_count
                for item in weights:
                    self.materialized_counts.append(unwrap_tensor.call_count)
                    self.loaded.append(item)

        model = SinglePassModel()
        runner = SimpleNamespace(server_args=SimpleNamespace(weight_cache_mode="off"))
        updater = WeightUpdater(
            tp_rank=0,
            device="cuda",
            gpu_id=0,
            model_config=SimpleNamespace(),
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=MagicMock(),
            recapture_cuda_graph=MagicMock(),
            get_model_runner=lambda: runner,
        )

        success, _ = updater.update_weights_from_tensor(
            named_tensors=[("a", "one"), ("b", "two"), ("c", "three")],
            stream_tensors=True,
        )

        self.assertTrue(success)
        self.assertEqual(model.started_with_materialized, 0)
        self.assertEqual(model.materialized_counts, [1, 2])
        self.assertEqual(
            model.loaded,
            [("a", "device-one"), ("c", "device-three")],
        )
        stream.synchronize.assert_called_once_with()


class TestDSparkStreamingLoader(unittest.TestCase):
    def test_backbone_generator_is_consumed_one_tensor_at_a_time(self):
        model = DSparkDraftModel.__new__(DSparkDraftModel)
        torch.nn.Module.__init__(model)
        model.markov_head = None
        model.confidence_head = None
        model._stacked_ctx_kv_cache = False
        events = []

        def weights():
            events.append("yield-a")
            yield "layers.0.a", torch.tensor([1.0])
            events.append("yield-b")
            yield "layers.0.b", torch.tensor([2.0])

        def consume_one(_, name, loaded_weight, params_dict):
            events.append(f"load-{name[-1]}")

        with patch.object(
            DFlashDraftModel, "_load_dflash_weight", autospec=True
        ) as load:
            load.side_effect = consume_one
            model.load_weights(weights())

        self.assertEqual(events, ["yield-a", "load-a", "yield-b", "load-b"])
        self.assertEqual(load.call_count, 2)

    def test_skipped_weights_are_filtered_before_gpu_materialization(self):
        for name in ("embed_tokens.weight", "lm_head.weight", "rotary_emb.inv_freq"):
            self.assertFalse(DSparkDraftMixin.should_materialize_weight(name))
        self.assertTrue(
            DSparkDraftMixin.should_materialize_weight("layers.0.q_proj.weight")
        )

    def test_derived_cache_refresh_preserves_captured_storage(self):
        with torch.inference_mode():
            old_weight = torch.zeros(4)
            old_norm = torch.zeros(2, dtype=torch.float32)
        cached = {
            "weight": old_weight,
            "bias": None,
            "k_norm_weight": old_norm,
            "eps": 1e-5,
        }
        refreshed = {
            "weight": torch.arange(4, dtype=torch.float32),
            "bias": None,
            "k_norm_weight": torch.tensor([7.0, 8.0]),
            "eps": 2e-5,
        }
        model = SimpleNamespace(
            _stacked_ctx_kv_cache=cached,
            _build_stacked_ctx_kv_params=MagicMock(return_value=refreshed),
        )
        weight_ptr = old_weight.data_ptr()
        norm_ptr = old_norm.data_ptr()

        DSparkDraftMixin.refresh_derived_weight_caches(model)

        self.assertEqual(old_weight.data_ptr(), weight_ptr)
        self.assertEqual(old_norm.data_ptr(), norm_ptr)
        torch.testing.assert_close(old_weight, refreshed["weight"])
        torch.testing.assert_close(old_norm, refreshed["k_norm_weight"])
        self.assertEqual(cached["eps"], 2e-5)


if __name__ == "__main__":
    unittest.main()
