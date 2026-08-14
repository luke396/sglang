"""CPU-only invariants for the reusable hidden-capture version harness."""

import importlib.util
import math
import os
import tempfile
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

    def test_default_schedule_keeps_each_cell_on_one_gpu(self):
        baseline = {"label": "v5"}
        candidate = {"label": "v6"}
        waves = versions.comparison_waves(
            ("cell_a", "cell_b"), ("0", "1"), baseline, candidate
        )
        self.assertEqual(len(waves), 4)
        by_cell = {"cell_a": [], "cell_b": []}
        for wave in waves:
            self.assertEqual({arm["gpu"] for arm in wave}, {"0", "1"})
            self.assertEqual({arm["revision"]["label"] for arm in wave}, {"v5", "v6"})
            self.assertEqual({arm["capture"] for arm in wave}, {False, True})
            for arm in wave:
                by_cell[arm["cell"]].append(arm)
        self.assertEqual({arm["gpu"] for arm in by_cell["cell_a"]}, {"0"})
        self.assertEqual({arm["gpu"] for arm in by_cell["cell_b"]}, {"1"})
        for arms in by_cell.values():
            self.assertEqual(
                {(arm["revision"]["label"], arm["capture"]) for arm in arms},
                {("v5", False), ("v5", True), ("v6", False), ("v6", True)},
            )

    def test_default_schedule_shards_cells_across_multiple_gpus(self):
        baseline = {"label": "v5"}
        candidate = {"label": "v6"}
        cells = tuple(f"cell_{index}" for index in range(10))
        waves = versions.comparison_waves(
            cells, ("0", "1", "2", "3"), baseline, candidate
        )
        arms = [arm for wave in waves for arm in wave]
        self.assertEqual(len(arms), 4 * len(cells))
        self.assertEqual(len(waves), 12)
        for cell in cells:
            cell_arms = [arm for arm in arms if arm["cell"] == cell]
            self.assertEqual(len(cell_arms), 4)
            self.assertEqual(len({arm["gpu"] for arm in cell_arms}), 1)
        for wave in waves:
            self.assertEqual(len({arm["gpu"] for arm in wave}), len(wave))

    def test_upstream_baseline_schedule_launches_only_one_off_arm_per_cell(self):
        baseline = {"label": "stock"}
        candidate = {"label": "v6"}
        cells = tuple(f"cell_{index}" for index in range(5))
        waves = versions.comparison_waves(
            cells,
            ("0", "1"),
            baseline,
            candidate,
            scheduled_arms=((baseline, False),),
        )
        arms = [arm for wave in waves for arm in wave]
        self.assertEqual(len(arms), len(cells))
        self.assertEqual(len(waves), 3)
        self.assertTrue(
            all(
                arm["revision"]["label"] == "stock" and not arm["capture"]
                for arm in arms
            )
        )
        for cell in cells:
            self.assertEqual(len([arm for arm in arms if arm["cell"] == cell]), 1)

    def test_manifest_snapshot_is_non_destructive_and_reports_holes(self):
        prefix = "store/_seq/0"
        store = _FakeStore({f"{prefix}/0", f"{prefix}/2", f"{prefix}/3"})
        snapshot = matrix._manifest_snapshot(store, "store", 6)
        self.assertEqual(snapshot["count"], 3)
        self.assertEqual(snapshot["first_seq"], 0)
        self.assertEqual(snapshot["last_seq"], 3)
        self.assertEqual(snapshot["holes_through_last"], [1])
        self.assertEqual(len(store.keys), 3)

    def test_access_log_counts_all_completed_pre_measurement_generates(self):
        with tempfile.NamedTemporaryFile(mode="w") as stream:
            stream.write(
                '[date] INFO: 127.0.0.1 - "POST /generate HTTP/1.1" 200 OK\n'
                '[date] INFO: 127.0.0.1 - "GET /health HTTP/1.1" 200 OK\n'
                '[date] INFO: 127.0.0.1 - "POST /generate HTTP/1.1" 200 OK\n'
            )
            stream.flush()
            self.assertEqual(matrix._completed_generate_requests(stream.name), 2)

    def test_manifest_drain_marks_stable_incomplete_baseline_invalid(self):
        store = _FakeStore({"store/_seq/0/0"})
        with (
            mock.patch.object(matrix, "MANIFEST_POLL_S", 0),
            mock.patch.object(matrix, "MANIFEST_QUIET_POLLS", 0),
            mock.patch.object(matrix, "MANIFEST_EXACT_QUIET_POLLS", 0),
        ):
            snapshot, _, _, drained = matrix._drain_manifest(
                store,
                "store",
                baseline_count=0,
                expected_delta=2,
                scan_limit=4,
                timeout_s=0.1,
            )
            self.assertEqual(snapshot["count"], 1)
            self.assertFalse(drained)

            store.keys.add("store/_seq/0/1")
            snapshot, _, _, drained = matrix._drain_manifest(
                store,
                "store",
                baseline_count=0,
                expected_delta=2,
                scan_limit=4,
                timeout_s=0.1,
            )
            self.assertEqual(snapshot["count"], 2)
            self.assertTrue(drained)

    def test_progress_probes_advance_both_overlap_slots(self):
        response = mock.Mock(status_code=200)
        with (
            mock.patch.object(matrix, "SERVER_URL", "http://server"),
            mock.patch.object(matrix.requests, "get", return_value=response) as get,
        ):
            probes = matrix._run_capture_progress_probes("pre_measurement")
        self.assertEqual(len(probes), 2)
        self.assertTrue(all(probe["status_code"] == 200 for probe in probes))
        self.assertEqual(
            get.call_args_list,
            [
                mock.call("http://server/health_generate", timeout=30),
                mock.call("http://server/health_generate", timeout=30),
            ],
        )

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
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}),
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
            "capture_progress_probes": {
                stage: [{"status_code": 200}, {"status_code": 200}]
                for stage in ("pre_measurement", "post_measurement")
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
        row["capture_progress_probes"] = {}
        self.assertEqual(
            versions._row_gate(
                row,
                "sha",
                "v1_short_low",
                "1",
                require_progress_probes=False,
            ),
            [],
        )

    def test_candidate_off_reference_is_same_gpu_and_regated(self):
        candidate = {"label": "v6", "revision": "candidate-sha"}
        source_row = {"cell": {"capture": False}}
        source_attempt = {
            "attempt_id": "old-attempt",
            "cell": "v1_short_low",
            "gpu": "1",
            "version_label": "old-v6-label",
            "revision": "candidate-sha",
            "capture": False,
            "exit_code": 0,
            "row": source_row,
            "row_gate_failures": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            path.write_text(f"{versions.json.dumps(source_attempt)}\n")
            expected_sha = versions.hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(versions, "_row_gate", return_value=[]) as gate:
                selected, source = versions._load_candidate_off_references(
                    path,
                    candidate=candidate,
                    cells=("v1_short_low",),
                    cell_gpus={"v1_short_low": "1"},
                )
        self.assertEqual(source["sha256"], expected_sha)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["version_label"], "v6")
        self.assertTrue(selected[0]["reused"])
        gate.assert_called_once_with(
            source_row,
            "candidate-sha",
            "v1_short_low",
            "1",
            require_progress_probes=False,
        )

    def test_off_only_summary_requires_both_stock_and_candidate(self):
        row = {
            "bench": {metric: 1.0 for metric in versions.COMMON_METRICS},
            "measured_request_set": {"fingerprint": {"sha256": "requests"}},
            "outer_warmup_request_set": {"fingerprint": {"sha256": "warmup"}},
            "driver_source": {
                "git_revision": "measurement-driver",
                "worktree_dirty": False,
            },
        }
        attempts = [
            {
                "attempt_id": "stock-only",
                "cell": "v1_short_low",
                "gpu": "0",
                "version_label": "stock",
                "capture": False,
                "row": row,
                "row_gate_failures": [],
            }
        ]
        summary = versions.summarize_attempts(
            attempts,
            ("stock", "v6"),
            ("v1_short_low",),
            required_arms=(("stock", False), ("v6", False)),
        )
        cell = summary["cells"]["v1_short_low"]
        self.assertFalse(cell["comparable"])
        self.assertTrue(
            any("missing arms" in failure for failure in cell["gate_failures"])
        )
        candidate_row = {
            **row,
            "driver_source": {
                "git_revision": "different-driver",
                "worktree_dirty": False,
            },
        }
        attempts.append(
            {
                "attempt_id": "candidate-reference",
                "cell": "v1_short_low",
                "gpu": "0",
                "version_label": "v6",
                "capture": False,
                "row": candidate_row,
                "row_gate_failures": [],
            }
        )
        summary = versions.summarize_attempts(
            attempts,
            ("stock", "v6"),
            ("v1_short_low",),
            required_arms=(("stock", False), ("v6", False)),
        )
        self.assertTrue(
            any(
                "measurement driver revisions differ" in failure
                for failure in summary["cells"]["v1_short_low"]["gate_failures"]
            )
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

    def test_fixed_gpu_summary_compares_capture_pairs_without_missing_swaps(self):
        def row(value):
            return {
                "bench": {metric: value for metric in versions.COMMON_METRICS},
                "measured_request_set": {"fingerprint": {"sha256": "requests"}},
                "outer_warmup_request_set": {"fingerprint": {"sha256": "warmup"}},
                "driver_source": {
                    "git_revision": "measurement-driver",
                    "worktree_dirty": False,
                },
            }

        attempts = []
        for gpu, label, capture, value in (
            ("0", "v5", False, 100.0),
            ("0", "v5", True, 99.0),
            ("0", "v6", True, 98.0),
            ("0", "v6", False, 100.0),
        ):
            attempts.append(
                {
                    "attempt_id": f"{gpu}-{label}-{capture}",
                    "cell": "v1_short_low",
                    "gpu": gpu,
                    "version_label": label,
                    "capture": capture,
                    "row": row(value),
                    "row_gate_failures": [],
                }
            )
        summary = versions.summarize_attempts(
            attempts,
            ("v5", "v6"),
            ("v1_short_low",),
            scheduling_mode="fixed-cell-gpu",
        )
        cell = summary["cells"]["v1_short_low"]
        metric = cell["metrics"]["request_throughput"]
        self.assertTrue(cell["comparable"])
        self.assertIsNone(metric["crossed_geomean"])
        self.assertAlmostEqual(
            metric["aggregate_geomean"]["baseline_capture_cost_pct"], -1.0
        )
        self.assertAlmostEqual(
            metric["aggregate_geomean"]["candidate_capture_cost_pct"], -2.0
        )


if __name__ == "__main__":
    unittest.main()
