"""CPU-only invariants for the reusable hidden-capture version harness."""

import importlib.util
import math
import os
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[3]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


matrix = _load(
    "hidden_capture_matrix_harness", "test/manual/bench_hidden_capture_matrix.py"
)
versions = _load(
    "hidden_capture_version_harness", "test/manual/bench_hidden_capture_versions.py"
)


class _FakeStore:
    def __init__(self, keys):
        self.keys = set(keys)

    def is_exist(self, key):
        return int(key in self.keys)


class TestHiddenCaptureBenchmarkHarness(unittest.TestCase):
    def test_version_suite_is_frozen_and_paired(self):
        cells = matrix.version_comparison_v1_cells()
        self.assertEqual(len(cells), 2 * len(versions.SUITE_CELLS))
        by_name = {}
        for name, repeat, cell in cells:
            self.assertEqual(repeat, 0)
            base_name = name.split("|", 1)[0]
            by_name.setdefault(base_name, set()).add(cell["capture"])
        self.assertEqual(tuple(by_name), versions.SUITE_CELLS)
        self.assertTrue(all(arms == {False, True} for arms in by_name.values()))

    def test_crossed_schedule_balances_every_concurrent_step(self):
        baseline = {"label": "v5"}
        candidate = {"label": "v6"}
        gpu_a, gpu_b = versions.crossed_schedules(baseline, candidate)
        for step in range(4):
            pair = (gpu_a[step], gpu_b[step])
            self.assertEqual({revision["label"] for revision, _ in pair}, {"v5", "v6"})
            self.assertEqual({capture for _, capture in pair}, {False, True})

    def test_manifest_snapshot_is_non_destructive_and_reports_holes(self):
        prefix = "store/_seq/0"
        store = _FakeStore({f"{prefix}/0", f"{prefix}/2", f"{prefix}/3"})
        snapshot = matrix._manifest_snapshot(store, "store", 6)
        self.assertEqual(snapshot["count"], 3)
        self.assertEqual(snapshot["first_seq"], 0)
        self.assertEqual(snapshot["last_seq"], 3)
        self.assertEqual(snapshot["holes_through_last"], [1])
        self.assertEqual(len(store.keys), 3)

    def test_coverage_fails_closed_on_contamination(self):
        with self.assertRaisesRegex(AssertionError, "outside the measured request set"):
            matrix._validated_export_coverage(
                exported=161,
                completed=160,
                expected_requests=160,
                measured_warmup_requests=0,
                outer_warmup_drained=True,
                measured_drained=True,
            )

    def test_version_suite_requires_one_numeric_visible_gpu(self):
        for value in ("", "0,1", "GPU-deadbeef"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": value}
            ):
                with self.assertRaisesRegex(
                    SystemExit, "exactly one numeric physical GPU"
                ):
                    matrix._require_single_visible_physical_gpu()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            self.assertEqual(matrix._require_single_visible_physical_gpu(), "7")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " 0"}):
            self.assertEqual(matrix._require_single_visible_physical_gpu(), "0")

    def test_row_gate_checks_scheduled_gpu(self):
        row = {
            "status": "ok",
            "server_source": {"git_revision": "sha", "worktree_dirty": False},
            "name": "v1_short_low|cap=0",
            "selected_physical_gpu_ids": ["1"],
            "measured_request_set": {
                "requested": 1,
                "completed": 1,
                "inner_warmup_requests": 0,
                "fingerprint": {"sha256": "fingerprint"},
            },
            "outer_warmup_request_set": {
                "requested": 1,
                "completed": 1,
                "inner_warmup_requests": 0,
                "fingerprint": {"sha256": "warmup-fingerprint"},
            },
            "cell": {"capture": False},
            "cache_hit_rate_pct": 0.0,
            "gpu_resources": {"samples": [{"elapsed_s": 0.0}]},
            "process_resources": {
                "samples": [{"elapsed_s": 0.0}],
                "host_net_pernic_delta": {},
            },
            "gpu_process_settled": True,
            "gpu_process_introduced_after_cleanup": [],
        }
        row["bench"] = {metric: 1 for metric in versions.COMMON_METRICS}
        row["bench"]["completed"] = 1
        self.assertEqual(versions._row_gate(row, "sha", "v1_short_low", "1"), [])
        self.assertIn(
            "selected physical GPU=['1'] expected=['0']",
            versions._row_gate(row, "sha", "v1_short_low", "0"),
        )

    def test_diagnosis_summary_does_not_expand_to_frozen_full_suite(self):
        summary = versions.summarize_attempts(
            [], ("baseline", "candidate"), ("v1_short_low",)
        )
        self.assertEqual(tuple(summary["cells"]), ("v1_short_low",))

    def test_ratio_helpers_use_multiplicative_difference_in_differences(self):
        self.assertAlmostEqual(versions._ratio_pct(99, 100), -1.0)
        # (candidate on/off) / (baseline on/off) = (99/100)/(102/100)
        did = versions._geomean_ratio_pct([(99 * 100, 100 * 102)])
        self.assertTrue(math.isclose(did, -2.941176470588236, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
