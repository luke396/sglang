import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import requests
from sglang.benchmark.dspark_sps_profiler import (
    LoadInfo,
    ProfileEndpoints,
    RoundSettings,
    ServerContext,
    SpsRow,
    _ensure_decode_idle,
    _stop_load,
    build_request_count_sweep,
    build_table_from_summaries,
    count_aligned_steps,
    fetch_server_context,
    postprocess_round,
    resolve_cuda_graph_max_bs,
    resolve_profile_endpoints,
    round_summary_dict,
    run_one_round,
    validate_sweep_against_server,
    write_manifest,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def make_load_info() -> LoadInfo:
    return LoadInfo(
        num_requests=4, max_new_tokens=1200, wall_seconds=1.0, reached_target=True
    )


def make_rows(
    *,
    num_rows: int = 30,
    num_running_reqs: int = 4,
    num_verify_tokens: int = 32,
    step_time: float = 0.01,
    first_forward_ct: int = 0,
) -> list[SpsRow]:
    return [
        SpsRow(
            forward_ct=first_forward_ct + index,
            num_running_reqs=num_running_reqs,
            num_verify_tokens=num_verify_tokens,
            step_time=step_time,
        )
        for index in range(num_rows)
    ]


def make_context(**overrides) -> ServerContext:
    values = {
        "control_base_url": "http://localhost:30000",
        "tokenizer_path": "dummy",
        "tp_size": 4,
        "dp_size": 1,
        "verify_num_draft_tokens": 8,
        "simulate_acc_len": 1.0,
        "cuda_graph_max_bs": 128,
        "skip_max_running_requests_threshold": float("inf"),
        "skip_token_capacity_threshold": float("inf"),
    }
    values.update(overrides)
    return ServerContext(**values)


class TestProfileEndpoints(CustomTestCase):
    def test_legacy_base_url_maps_load_and_control_to_one_server(self):
        endpoints = resolve_profile_endpoints(base_url="http://localhost:30000/")
        self.assertEqual(
            endpoints,
            ProfileEndpoints(
                load_base_url="http://localhost:30000",
                control_base_url="http://localhost:30000",
            ),
        )
        self.assertFalse(endpoints.is_separate)

    def test_separate_endpoints_require_a_complete_unambiguous_pair(self):
        endpoints = resolve_profile_endpoints(
            base_url="",
            load_base_url="http://router:8000/",
            control_base_url="http://decode:30001/",
        )
        self.assertEqual(endpoints.load_base_url, "http://router:8000")
        self.assertEqual(endpoints.control_base_url, "http://decode:30001")
        self.assertTrue(endpoints.is_separate)

        invalid = [
            {"base_url": "", "load_base_url": "http://router:8000"},
            {"base_url": "", "control_base_url": "http://decode:30001"},
            {
                "base_url": "http://legacy:30000",
                "load_base_url": "http://router:8000",
                "control_base_url": "http://decode:30001",
            },
            {"base_url": ""},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_profile_endpoints(**kwargs)


class TestPdEndpointRouting(CustomTestCase):
    def test_separate_round_sends_load_to_router_and_control_to_decode(self):
        endpoints = ProfileEndpoints(
            load_base_url="http://router:8000",
            control_base_url="http://decode:30001",
        )
        context = make_context(
            control_base_url=endpoints.control_base_url,
            disaggregation_mode="decode",
        )
        settings = RoundSettings(
            input_len=16,
            temperature=1.0,
            min_steady_steps=16,
            min_steady_seconds=0.0,
            round_timeout_seconds=5.0,
        )
        load_thread = Mock()
        load_thread.is_alive.return_value = False
        rows = make_rows(num_rows=30)
        module = "sglang.benchmark.dspark_sps_profiler"

        with (
            patch(
                f"{module}.should_skip_due_to_max_running_requests",
                return_value=False,
            ),
            patch(f"{module}.should_skip_due_to_token_capacity", return_value=False),
            patch(f"{module}.set_forced_budget_frac") as set_budget,
            patch(f"{module}._ensure_decode_idle") as ensure_idle,
            patch(f"{module}.fetch_rank_rows", side_effect=[[[]], [rows]]) as fetch,
            patch(f"{module}.start_load", return_value=load_thread) as start,
            patch(f"{module}.wait_for_aligned_steps", return_value=True),
            patch(f"{module}.abort_all_requests") as abort,
        ):
            outcome = run_one_round(
                context=context,
                endpoints=endpoints,
                vocab_size=4096,
                batch_size_per_rank=4,
                settings=settings,
                rng=random.Random(42),
                frac=0.5,
            )

        self.assertIsNotNone(outcome)
        self.assertEqual(start.call_args.kwargs["base_url"], endpoints.load_base_url)
        set_budget.assert_called_once_with(
            base_url=endpoints.control_base_url, frac=0.5
        )
        self.assertEqual(
            ensure_idle.call_args_list,
            [
                call(control_base_url=endpoints.control_base_url),
                call(control_base_url=endpoints.control_base_url),
            ],
        )
        self.assertEqual(
            fetch.call_args_list,
            [
                call(base_url=endpoints.control_base_url),
                call(base_url=endpoints.control_base_url),
            ],
        )
        abort.assert_called_once_with(base_url=endpoints.control_base_url)

    def test_separate_round_stops_if_router_load_does_not_exit_after_decode_abort(
        self,
    ):
        load_thread = Mock()
        load_thread.is_alive.return_value = True
        module = "sglang.benchmark.dspark_sps_profiler"
        with (
            patch(f"{module}.abort_all_requests") as abort,
            patch(f"{module}.LOAD_JOIN_TIMEOUT_SECONDS", 0.01),
            self.assertRaisesRegex(RuntimeError, "did not return"),
        ):
            _stop_load(
                load_thread=load_thread,
                control_base_url="http://decode:30001",
                batch_size=4,
                require_join=True,
            )
        abort.assert_called_once_with(base_url="http://decode:30001")
        load_thread.join.assert_called_once_with(timeout=0.01)

    def test_decode_idle_barrier_rejects_pending_requests(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "still busy"
        )
        module = "sglang.benchmark.dspark_sps_profiler"
        with (
            patch(f"{module}.requests.post", return_value=response) as post,
            patch(f"{module}.LOAD_JOIN_TIMEOUT_SECONDS", 7),
            self.assertRaisesRegex(RuntimeError, "did not become idle"),
        ):
            _ensure_decode_idle(control_base_url="http://decode:30001")
        self.assertEqual(post.call_args.args[0], "http://decode:30001/flush_cache")
        self.assertEqual(post.call_args.kwargs["params"], {"timeout": 7})

    def test_separate_control_endpoint_must_report_pd_decode_role(self):
        response = Mock()
        response.json.return_value = {"disaggregation_mode": "prefill"}
        module = "sglang.benchmark.dspark_sps_profiler"
        with (
            patch(f"{module}.requests.get", return_value=response) as get,
            self.assertRaisesRegex(ValueError, "PD decode server"),
        ):
            fetch_server_context(
                control_base_url="http://prefill:30000",
                local_tokenizer_path=None,
                require_pd_decode=True,
            )
        self.assertEqual(get.call_args.args[0], "http://prefill:30000/server_info")

    def test_manifest_preserves_legacy_base_url_and_records_both_endpoints(self):
        endpoints = ProfileEndpoints(
            load_base_url="http://router:8000",
            control_base_url="http://decode:30001",
        )
        context = make_context(
            control_base_url=endpoints.control_base_url,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        settings = RoundSettings(
            input_len=16,
            temperature=1.0,
            min_steady_steps=16,
            min_steady_seconds=1.0,
            round_timeout_seconds=5.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "sps.json.manifest.json"
            write_manifest(
                manifest_path=manifest_path,
                records_path=Path(tmpdir) / "sps.records.jsonl",
                rounds_path=Path(tmpdir) / "sps.rounds.jsonl",
                context=context,
                endpoints=endpoints,
                batch_sizes=[1, 4],
                settings=settings,
                repeats=1,
                rounds=[],
                fracs=None,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["base_url"], endpoints.control_base_url)
        self.assertEqual(manifest["load_base_url"], endpoints.load_base_url)
        self.assertEqual(manifest["control_base_url"], endpoints.control_base_url)
        self.assertEqual(manifest["endpoint_layout"], "separate")
        self.assertEqual(manifest["disaggregation_mode"], "decode")
        self.assertEqual(manifest["disaggregation_transfer_backend"], "nixl")

    def test_manifest_omits_unreported_disaggregation_fields(self):
        endpoints = ProfileEndpoints(
            load_base_url="http://localhost:30000",
            control_base_url="http://localhost:30000",
        )
        context = make_context()
        settings = RoundSettings(
            input_len=16,
            temperature=1.0,
            min_steady_steps=16,
            min_steady_seconds=1.0,
            round_timeout_seconds=5.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "sps.json.manifest.json"
            write_manifest(
                manifest_path=manifest_path,
                records_path=Path(tmpdir) / "sps.records.jsonl",
                rounds_path=Path(tmpdir) / "sps.rounds.jsonl",
                context=context,
                endpoints=endpoints,
                batch_sizes=[1],
                settings=settings,
                repeats=1,
                rounds=[],
                fracs=None,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertNotIn("disaggregation_mode", manifest)
        self.assertNotIn("disaggregation_transfer_backend", manifest)


class TestPostprocessRound(CustomTestCase):
    def test_single_rank_round_builds_probe_from_median_step_time(self):
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01)],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)
        self.assertEqual(outcome.match_fraction, 1.0)

    def test_round_warmup_steps_are_dropped_from_timing(self):
        slow_head = make_rows(num_rows=8, step_time=0.5, first_forward_ct=0)
        steady_tail = make_rows(num_rows=20, step_time=0.01, first_forward_ct=8)
        outcome = postprocess_round(
            rank_rows=[slow_head + steady_tail],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 100.0)

    def test_off_target_batch_rows_are_filtered_out(self):
        ramp = make_rows(num_rows=10, num_running_reqs=2, num_verify_tokens=16)
        steady = make_rows(num_rows=30, first_forward_ct=10, step_time=0.02)
        outcome = postprocess_round(
            rank_rows=[ramp + steady],
            batch_size_per_rank=4,
            dp_size=1,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertAlmostEqual(outcome.match_fraction, 1.0)

    def test_mid_round_instability_raises(self):
        head = make_rows(num_rows=15)
        gap = make_rows(
            num_rows=40, num_running_reqs=3, num_verify_tokens=24, first_forward_ct=15
        )
        tail = make_rows(num_rows=15, first_forward_ct=55)
        with self.assertRaisesRegex(RuntimeError, "unstable mid-round"):
            postprocess_round(
                rank_rows=[head + gap + tail],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_round_that_never_stabilizes_raises(self):
        rows = make_rows(num_rows=50, num_running_reqs=3, num_verify_tokens=24)
        rows += make_rows(num_rows=2, first_forward_ct=50)
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[rows],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )


class TestPostprocessRoundCrossRank(CustomTestCase):
    def test_two_uniform_ranks_average_their_step_times(self):
        outcome = postprocess_round(
            rank_rows=[make_rows(step_time=0.01), make_rows(step_time=0.03)],
            batch_size_per_rank=4,
            dp_size=2,
            verify_num_draft_tokens=8,
            min_steady_steps=16,
            load_info=make_load_info(),
        )
        self.assertEqual(outcome.batch_size_per_rank, 4)
        self.assertEqual(outcome.batch_tokens, 32)
        self.assertAlmostEqual(outcome.steps_per_sec, 50.0)
        self.assertEqual(len(outcome.per_rank_median_step_time), 2)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[0], 0.01)
        self.assertAlmostEqual(outcome.per_rank_median_step_time[1], 0.03)

    def test_rank_with_no_new_records_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no new decode-step records"):
            postprocess_round(
                rank_rows=[make_rows(), []],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_disjoint_forward_ct_ranges_raise(self):
        with self.assertRaisesRegex(RuntimeError, "no common forward_ct"):
            postprocess_round(
                rank_rows=[
                    make_rows(first_forward_ct=0),
                    make_rows(first_forward_ct=1000),
                ],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_rank_below_expected_verify_tokens_raises(self):
        # A rank reporting fewer verify tokens than bs_per_rank * K is not
        # running the uniform static verify; above-expected counts are
        # tolerated (the recorded count is the replayed graph tier).
        with self.assertRaisesRegex(RuntimeError, "num_verify_tokens"):
            postprocess_round(
                rank_rows=[make_rows(), make_rows(num_verify_tokens=24)],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )

    def test_rank_count_mismatch_raises(self):
        with self.assertRaisesRegex(RuntimeError, "DP ranks"):
            postprocess_round(
                rank_rows=[make_rows()],
                batch_size_per_rank=4,
                dp_size=2,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )


class TestTableAssembly(CustomTestCase):
    def test_repeats_take_the_median_per_batch_tokens(self):
        rounds = [
            postprocess_round(
                rank_rows=[make_rows(step_time=step_time)],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=16,
                load_info=make_load_info(),
            )
            for step_time in (0.01, 0.02, 0.04)
        ]
        table = build_table_from_summaries(
            summaries=[
                round_summary_dict(outcome=outcome, repeat=repeat)
                for repeat, outcome in enumerate(rounds)
            ],
            max_batch_tokens=None,
            offdiag=False,
        )
        self.assertEqual(table.sample_batch_tokens, [32])
        self.assertAlmostEqual(table.sample_steps_per_sec[0], 50.0)


class TestSweepHelpers(CustomTestCase):
    def test_request_count_sweep_tapers_and_hits_the_max(self):
        sweep = build_request_count_sweep(100)
        self.assertEqual(sweep[:4], [1, 2, 4, 8])
        self.assertEqual(sweep[-1], 100)
        self.assertIn(64, sweep)

    def test_sweep_beyond_captured_cuda_graphs_raises(self):
        with self.assertRaisesRegex(ValueError, "cuda graphs"):
            validate_sweep_against_server(
                context=make_context(cuda_graph_max_bs=64),
                batch_sizes=[8, 128],
            )

    def test_sweep_within_captured_cuda_graphs_passes(self):
        validate_sweep_against_server(
            context=make_context(cuda_graph_max_bs=64, dp_size=2),
            batch_sizes=[8, 64],
        )

    def test_resolve_cuda_graph_max_bs_prefers_captured_list(self):
        internal_state = {
            "cuda_graph_config": {"decode": {"bs": [1, 2, 160], "max_bs": 128}}
        }
        self.assertEqual(resolve_cuda_graph_max_bs(internal_state=internal_state), 160)

    def test_resolve_cuda_graph_max_bs_handles_missing_config(self):
        self.assertIsNone(resolve_cuda_graph_max_bs(internal_state={}))


class TestCountAlignedSteps(CustomTestCase):
    def test_off_target_steps_are_not_counted(self):
        rows = make_rows(num_rows=10, num_running_reqs=3)
        self.assertEqual(
            count_aligned_steps(rank_rows=[rows], batch_size_per_rank=4), 0
        )


class TestMinSteadySteps(CustomTestCase):
    def test_min_steady_steps_rejects_thin_probes(self):
        with self.assertRaisesRegex(RuntimeError, "never stabilized"):
            postprocess_round(
                rank_rows=[make_rows(num_rows=20)],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=8,
                min_steady_steps=32,
                load_info=make_load_info(),
            )


if __name__ == "__main__":
    unittest.main()
