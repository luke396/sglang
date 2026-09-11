"""B300 per-commit CI coverage for Kimi-K3 serving recipes.

Runs the Balanced DCP/HiCache and MegaMoE recipes on eight B300 GPUs, retaining
their GSM8K accuracy gates.
"""

import unittest

import requests
from prometheus_client.parser import text_string_to_metric_families

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=578, stage="base-c", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3"
DSPARK_DRAFT_MODEL = "RadixArk/Kimi-K3-DSpark"
MEGAMOE_URL = "http://0.0.0.0:30000"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true, "num_threads": 12}'
MEGAMOE_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
    "SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL": "10",
}
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


def _stop_server(process):
    if process:
        terminate_and_kill_process_tree(process)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


class TestKimiK3B300Balanced(GSM8KMixin, CustomTestCase):
    """TP8/DCP8 Balanced recipe with hierarchical cache."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 98

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--dcp-size",
                "8",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                MODEL_LOADER_EXTRA_CONFIG,
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--mamba-full-memory-ratio",
                "7.21",
                "--enable-hierarchical-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


class TestKimiK3B300MegaMoE(GSM8KMixin, CustomTestCase):
    """TP8/EP8/DCP8 MegaMoE recipe with DSPARK speculation."""

    gsm8k_score_threshold = 0.95
    gsm8k_num_examples = 200
    gsm8k_num_threads = 22

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = MEGAMOE_URL
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--moe-a2a-backend",
                "megamoe",
                "--enable-metrics",
                "--expert-balancedness-report-mode",
                "prometheus",
                "--ep",
                "8",
                "--dcp-size",
                "8",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                MODEL_LOADER_EXTRA_CONFIG,
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--mamba-full-memory-ratio",
                "5.13",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--speculative-dspark-block-size",
                "7",
                "--enable-linear-replayssm-spec",
            ],
            env=MEGAMOE_ENV,
        )

    def test_gsm8k(self):
        # Reuse the existing accuracy workload to populate the recorder.
        super().test_gsm8k()
        # Follow observability/test_metrics.py: inspect the HTTP export after
        # inference, using the Prometheus parser and positive-sample checks.
        response = requests.get(f"{self.base_url}/metrics", timeout=30)
        response.raise_for_status()
        samples = [
            sample
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
        ]
        for name in (
            "sglang:eplb_balancedness_count",
            # Existing Expert Dispatch per GPU Grafana panel consumes this.
            "sglang:eplb_gpu_physical_count_bucket",
        ):
            self.assertTrue(
                any(sample.name == name and sample.value > 0 for sample in samples),
                f"No positive samples for {name}",
            )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


if __name__ == "__main__":
    unittest.main()
