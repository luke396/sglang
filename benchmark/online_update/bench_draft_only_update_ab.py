#!/usr/bin/env python3
"""Reusable baseline/candidate A/B runner for draft-only from-tensor updates.

This is the single top-level invocation for the comparison.  It owns server
lifecycle, GPU freshness checks, workload ordering, child benchmark commands,
and the combined artifact so results are not assembled from manual commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROFILE_NAME = "qwen3-8b-dspark-from-tensor-production-v1"

MODEL_PATH = Path("/data/models/Qwen3-8B")
DRAFT_MODEL_PATH = Path("/data/models/dspark_qwen3_8b_block7")
ORIGINAL_CHECKPOINT = DRAFT_MODEL_PATH / "model.safetensors"
UPDATED_CHECKPOINT = Path(
    "/tmp/specloop-perturbed-drafts/dspark-rms-noise-1em03/model.safetensors"
)

SERVER_ENV = {
    "HF_HUB_OFFLINE": "1",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
}
BENCHMARK_ENV = {"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"}

WORKLOAD = {
    "traffic_concurrency": 12,
    "output_tokens": 256,
    "min_phase_samples": 24,
    "dwell_seconds": 0.0,
    "phase_timeout_seconds": 180.0,
    "request_timeout_seconds": 180.0,
    "nvml_interval_seconds": 0.002,
}

# ABBA across TP shapes counterbalances which implementation runs first.
CASE_TEMPLATES = (
    {"role": "baseline", "tp_size": 1, "gpus": [6], "port": 21761, "repeats": 20},
    {"role": "candidate", "tp_size": 1, "gpus": [6], "port": 21762, "repeats": 20},
    {"role": "candidate", "tp_size": 2, "gpus": [3, 4], "port": 21763, "repeats": 10},
    {"role": "baseline", "tp_size": 2, "gpus": [3, 4], "port": 21764, "repeats": 10},
)

GPU_IDLE_MEMORY_LIMIT_MIB = 64
STARTUP_TIMEOUT_SECONDS = 900
BENCHMARK_TIMEOUT_SECONDS = 1200
GPU_RELEASE_TIMEOUT_SECONDS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_artifact(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def nvidia_smi(args: list[str]) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *args], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def gpu_inventory() -> dict[str, Any]:
    gpu_output = nvidia_smi(
        [
            "--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    app_output = nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    gpus: dict[str, Any] = {}
    uuid_to_index: dict[str, str] = {}
    for line in gpu_output.splitlines():
        index, uuid, memory, utilization, temperature = (
            item.strip() for item in line.split(",", 4)
        )
        gpus[index] = {
            "uuid": uuid,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "temperature_c": int(temperature),
            "compute_processes": [],
        }
        uuid_to_index[uuid] = index

    for line in app_output.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, memory = (item.strip() for item in line.split(",", 3))
        index = uuid_to_index.get(uuid)
        if index is not None:
            gpus[index]["compute_processes"].append(
                {
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mib": int(memory),
                }
            )
    return gpus


def selected_gpu_inventory(indices: list[int]) -> dict[str, Any]:
    inventory = gpu_inventory()
    return {str(index): inventory[str(index)] for index in indices}


def gpu_is_idle(state: dict[str, Any]) -> bool:
    return (
        state["memory_used_mib"] <= GPU_IDLE_MEMORY_LIMIT_MIB
        and not state["compute_processes"]
    )


def require_fresh_gpus(indices: list[int]) -> dict[str, Any]:
    selected = selected_gpu_inventory(indices)
    busy = {index: state for index, state in selected.items() if not gpu_is_idle(state)}
    if busy:
        raise RuntimeError(f"assigned GPUs are not fresh: {busy}")
    return selected


def wait_for_gpu_release(indices: list[int]) -> dict[str, Any]:
    deadline = time.monotonic() + GPU_RELEASE_TIMEOUT_SECONDS
    while True:
        selected = selected_gpu_inventory(indices)
        if all(gpu_is_idle(state) for state in selected.values()):
            return selected
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs did not return idle: {selected}")
        time.sleep(1.0)


def server_command(source_root: Path, case: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(MODEL_PATH),
        "--trust-remote-code",
    ]
    if case["tp_size"] > 1:
        command.extend(["--tp", str(case["tp_size"])])
    command.extend(
        [
            "--attention-backend",
            "fa3",
            "--speculative-draft-attention-backend",
            "fa3",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            str(DRAFT_MODEL_PATH),
            "--mem-fraction-static",
            "0.7",
            "--page-size",
            "1",
            "--chunked-prefill-size",
            "8192",
            "--cuda-graph-max-bs-decode",
            "64",
            "--enable-hidden-state-capture",
            "--device",
            "cuda",
            "--host",
            "127.0.0.1",
            "--port",
            str(case["port"]),
        ]
    )
    return command


def benchmark_command(
    harness_root: Path,
    case: dict[str, Any],
    server_log: Path,
    child_artifact: Path,
    server_sha: str,
) -> list[str]:
    command = [
        sys.executable,
        str(
            harness_root / "benchmark" / "online_update" / "bench_draft_only_update.py"
        ),
        "--base-url",
        f"http://127.0.0.1:{case['port']}",
        "--tp-size",
        str(case["tp_size"]),
        "--update-mode",
        case["update_mode"],
        "--gpu-indices",
        *(str(index) for index in case["gpus"]),
        "--original-checkpoint",
        str(ORIGINAL_CHECKPOINT),
        "--updated-checkpoint",
        str(UPDATED_CHECKPOINT),
        "--repeats",
        str(case["repeats"]),
        "--traffic-concurrency",
        str(WORKLOAD["traffic_concurrency"]),
        "--output-tokens",
        str(WORKLOAD["output_tokens"]),
        "--min-phase-samples",
        str(WORKLOAD["min_phase_samples"]),
        "--dwell-seconds",
        str(WORKLOAD["dwell_seconds"]),
        "--phase-timeout-seconds",
        str(WORKLOAD["phase_timeout_seconds"]),
        "--request-timeout-seconds",
        str(WORKLOAD["request_timeout_seconds"]),
        "--nvml-interval-seconds",
        str(WORKLOAD["nvml_interval_seconds"]),
        "--weight-version-prefix",
        f"from-tensor-ab-{case['label']}-{server_sha[:9]}",
        "--server-log",
        str(server_log),
        "--artifact",
        str(child_artifact),
    ]
    if case.get("fault_injection", False):
        command.append("--fault-injection")
    return command


def wait_for_server(base_url: str, process: subprocess.Popen[Any], log: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = "not contacted"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited with {return_code} before ready\n{tail(log)}"
            )
        try:
            response = requests.get(f"{base_url}/server_info", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as error:
            last_error = repr(error)
        time.sleep(1.0)
    raise TimeoutError(f"server readiness timeout: {last_error}\n{tail(log)}")


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 15),
        (signal.SIGKILL, 5),
    ):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def tail(path: Path, line_count: int = 80) -> str:
    if not path.exists():
        return "<log missing>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-line_count:])


def compact_distribution(distribution: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "count",
        "mean",
        "median",
        "p95",
        "p99",
        "max",
        "population_variance",
        "population_stdev",
    )
    return {key: distribution.get(key) for key in keys}


def summarize_child(payload: dict[str, Any]) -> dict[str, Any]:
    request_summary = {}
    for phase in ("baseline", "updated", "restored"):
        phase_payload = payload["request_metrics"][phase]
        request_summary[phase] = {
            "request_count": phase_payload["request_count"],
            "ttft_ms": compact_distribution(phase_payload["ttft_ms"]),
            "tpot_ms": compact_distribution(phase_payload["tpot_ms"]),
            "e2e_latency_ms": compact_distribution(phase_payload["e2e_latency_ms"]),
            "accept_length": phase_payload["spec"]["accept_length"],
            "accept_rate": phase_payload["spec"]["accept_rate"],
        }

    transitions = payload["transition_distributions"]
    memory = {
        key: compact_distribution(value)
        for key, value in transitions.items()
        if "peak_delta_mib" in key
    }
    return {
        "status": payload["status"],
        "duration_seconds": payload["duration_seconds"],
        "pause": {
            "update_ms": compact_distribution(transitions["update_pause_window_ms"]),
            "restore_ms": compact_distribution(transitions["restore_pause_window_ms"]),
        },
        "update_interface": {
            "update_ms": compact_distribution(transitions["update_ms"]),
            "restore_ms": compact_distribution(transitions["restore_ms"]),
        },
        "requests": request_summary
        | {
            "total_request_count": payload["request_metrics"]["total_request_count"],
            "cross_transition_request_count": payload["request_metrics"][
                "cross_transition_request_count"
            ],
            "errors": payload["request_metrics"]["errors"],
        },
        "gpu_peak_delta_mib": memory,
        "memory_leak_analysis": payload["memory_leak_analysis"],
        "host_memory_leak_analysis": payload.get("host_memory_leak_analysis"),
        "phase_timing_distributions_ms": payload.get(
            "phase_timing_distributions_ms", {}
        ),
        "host_memory_distributions_bytes": payload.get(
            "host_memory_distributions_bytes", {}
        ),
        "fault_injection": payload.get("fault_injection"),
        "graph_log_before": payload["graph_log_before"],
        "graph_log_after": payload["graph_log_after"],
        "invariant_failures": payload["invariant_failures"],
    }


def metric_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ("mean", "median", "p95", "p99", "max"):
        first = baseline[key]
        second = candidate[key]
        result[key] = {
            "baseline": first,
            "candidate": second,
            "candidate_minus_baseline": second - first,
            "candidate_over_baseline": second / first if first else None,
        }
    return result


def build_comparison(
    cases: list[dict[str, Any]], variants: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparison = {
        "baseline_name": variants["baseline"]["name"],
        "candidate_name": variants["candidate"]["name"],
    }
    for tp_size in (1, 2):
        baseline = next(
            case["summary"]
            for case in cases
            if case["role"] == "baseline" and case["tp_size"] == tp_size
        )
        candidate = next(
            case["summary"]
            for case in cases
            if case["role"] == "candidate" and case["tp_size"] == tp_size
        )
        comparison[f"tp{tp_size}"] = {
            "update_pause_ms": metric_delta(
                baseline["pause"]["update_ms"], candidate["pause"]["update_ms"]
            ),
            "restore_pause_ms": metric_delta(
                baseline["pause"]["restore_ms"], candidate["pause"]["restore_ms"]
            ),
            "update_interface_ms": metric_delta(
                baseline["update_interface"]["update_ms"],
                candidate["update_interface"]["update_ms"],
            ),
            "restore_interface_ms": metric_delta(
                baseline["update_interface"]["restore_ms"],
                candidate["update_interface"]["restore_ms"],
            ),
        }
    return comparison


def validate_variant_name(name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError(
            f"invalid variant name {name!r}; use letters, digits, dot, dash, underscore"
        )


def expand_cases(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for template in CASE_TEMPLATES:
        variant = variants[template["role"]]
        cases.append(
            template
            | {
                "version": variant["name"],
                "label": f"{variant['name']}-tp{template['tp_size']}",
                "update_mode": variant["update_mode"],
                "fault_injection": bool(
                    variant["fault_injection"] and template["tp_size"] == 1
                ),
            }
        )
    return cases


def validate_inputs(variants: dict[str, dict[str, Any]]) -> dict[str, str]:
    if (
        variants["candidate"]["fault_injection"]
        and variants["candidate"]["update_mode"] != "atomic-presharded"
    ):
        raise ValueError(
            "candidate fault injection requires atomic-presharded update mode"
        )
    if variants["baseline"]["name"] == variants["candidate"]["name"]:
        raise ValueError("baseline and candidate names must differ")
    actual = {}
    for role, variant in variants.items():
        validate_variant_name(variant["name"])
        root = variant["source_root"]
        expected = variant["expected_sha"]
        if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
            raise ValueError(f"{role} expected SHA must be a full lowercase SHA")
        if not root.is_dir():
            raise FileNotFoundError(f"missing {role} source root: {root}")
        actual[role] = git_head(root)
        if actual[role] != expected:
            raise RuntimeError(
                f"{role} source mismatch: expected {expected}, found {actual[role]}"
            )
    for path in (
        MODEL_PATH,
        DRAFT_MODEL_PATH,
        ORIGINAL_CHECKPOINT,
        UPDATED_CHECKPOINT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    return actual


def run_case(
    *,
    harness_root: Path,
    source_root: Path,
    server_sha: str,
    case: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    label = case["label"]
    server_log = artifact_dir / f"{label}-server.log"
    benchmark_log = artifact_dir / f"{label}-benchmark.stdout.log"
    child_artifact = artifact_dir / f"{label}.json"
    before = require_fresh_gpus(case["gpus"])

    server_cmd = server_command(source_root, case)
    benchmark_cmd = benchmark_command(
        harness_root, case, server_log, child_artifact, server_sha
    )
    server_env_record = SERVER_ENV | {
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in case["gpus"]),
        "PYTHONPATH": str(source_root / "python"),
    }
    if case.get("fault_injection", False):
        server_env_record["SGLANG_ENABLE_WEIGHT_UPDATE_FAULT_INJECTION"] = "1"
    benchmark_env_record = BENCHMARK_ENV | {
        "CUDA_VISIBLE_DEVICES": server_env_record["CUDA_VISIBLE_DEVICES"],
        "PYTHONPATH": str(harness_root / "python"),
    }
    case_result: dict[str, Any] = {
        **case,
        "server_sha": server_sha,
        "source_root": str(source_root),
        "started_unix_seconds": time.time(),
        "gpu_before": before,
        "server_command": server_cmd,
        "server_environment": server_env_record,
        "benchmark_command": benchmark_cmd,
        "benchmark_environment": benchmark_env_record,
        "server_log": str(server_log),
        "benchmark_stdout_log": str(benchmark_log),
        "child_artifact": str(child_artifact),
        "status": "running",
    }

    server_env = os.environ.copy() | server_env_record
    benchmark_env = os.environ.copy() | benchmark_env_record
    process: subprocess.Popen[Any] | None = None
    try:
        with server_log.open("w") as log_file:
            process = subprocess.Popen(
                server_cmd,
                cwd=source_root,
                env=server_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            wait_for_server(f"http://127.0.0.1:{case['port']}", process, server_log)
            case_result["server_ready_unix_seconds"] = time.time()
            with benchmark_log.open("w") as output:
                completed = subprocess.run(
                    benchmark_cmd,
                    cwd=harness_root,
                    env=benchmark_env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=BENCHMARK_TIMEOUT_SECONDS,
                )
            case_result["benchmark_returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark failed with {completed.returncode}\n{tail(benchmark_log)}"
                )
            payload = json.loads(child_artifact.read_text())
            if payload.get("status") != "passed":
                raise RuntimeError(
                    f"child artifact did not pass: {payload.get('status')}"
                )
            if len(payload.get("cycles", [])) != case["repeats"]:
                raise RuntimeError("child artifact repeat count mismatch")
            case_result["summary"] = summarize_child(payload)
            case_result["status"] = "passed"
    except BaseException as error:
        case_result["status"] = "failed"
        case_result["error"] = repr(error)
    finally:
        if process is not None:
            stop_process_group(process)
        for key, path in (
            ("child_artifact_sha256", child_artifact),
            ("server_log_sha256", server_log),
            ("benchmark_stdout_log_sha256", benchmark_log),
        ):
            if path.exists():
                case_result[key] = sha256_file(path)
        try:
            case_result["gpu_after"] = wait_for_gpu_release(case["gpus"])
        except BaseException as error:
            case_result["status"] = "failed"
            case_result["gpu_release_error"] = repr(error)
            case_result["gpu_after"] = selected_gpu_inventory(case["gpus"])
        case_result["finished_unix_seconds"] = time.time()
    return case_result


def variants_from_args(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "baseline": {
            "name": args.baseline_name,
            "source_root": args.baseline_source_root.resolve(),
            "expected_sha": args.baseline_sha,
            "update_mode": args.baseline_update_mode,
            "fault_injection": False,
        },
        "candidate": {
            "name": args.candidate_name,
            "source_root": args.candidate_source_root.resolve(),
            "expected_sha": args.candidate_sha,
            "update_mode": args.candidate_update_mode,
            "fault_injection": args.candidate_fault_injection,
        },
    }


def main(args: argparse.Namespace) -> int:
    harness_root = Path(__file__).resolve().parents[2]
    variants = variants_from_args(args)
    source_shas = validate_inputs(variants)
    cases = expand_cases(variants)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = args.artifact_dir / "from-tensor-ab.json"

    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_unix_seconds": time.time(),
        "execution": {
            "argv": sys.argv,
            "harness_root": str(harness_root),
            "harness_git_sha": git_head(harness_root),
            "harness_sha256": sha256_file(Path(__file__)),
            "child_benchmark_sha256": sha256_file(
                harness_root
                / "benchmark"
                / "online_update"
                / "bench_draft_only_update.py"
            ),
        },
        "versions": {
            role: {
                "name": variant["name"],
                "expected_sha": variant["expected_sha"],
                "actual_sha": source_shas[role],
                "source_root": str(variant["source_root"]),
                "update_mode": variant["update_mode"],
                "fault_injection": variant["fault_injection"],
            }
            for role, variant in variants.items()
        },
        "profile": {
            "name": PROFILE_NAME,
            "model_path": str(MODEL_PATH),
            "draft_model_path": str(DRAFT_MODEL_PATH),
            "original_checkpoint": str(ORIGINAL_CHECKPOINT),
            "updated_checkpoint": str(UPDATED_CHECKPOINT),
            "server_environment": SERVER_ENV,
            "benchmark_environment": BENCHMARK_ENV,
            "workload": WORKLOAD,
            "case_order": [case["label"] for case in cases],
            "case_templates": CASE_TEMPLATES,
            "gpu_idle_memory_limit_mib": GPU_IDLE_MEMORY_LIMIT_MIB,
        },
        "cases": [],
    }
    write_artifact(artifact_path, result)

    try:
        for case in cases:
            print(f"starting {case['label']}", flush=True)
            role = case["role"]
            case_result = run_case(
                harness_root=harness_root,
                source_root=variants[role]["source_root"],
                server_sha=source_shas[role],
                case=case,
                artifact_dir=args.artifact_dir,
            )
            result["cases"].append(case_result)
            write_artifact(artifact_path, result)
            if case_result["status"] != "passed":
                raise RuntimeError(
                    f"{case['label']} failed: {case_result.get('error')}"
                )
            print(f"completed {case['label']}", flush=True)
        result["comparison"] = build_comparison(result["cases"], variants)
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        result["finished_unix_seconds"] = time.time()
        result["final_gpu_inventory"] = selected_gpu_inventory([3, 4, 6])
        write_artifact(artifact_path, result)
    print(artifact_path, flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen draft-only from-tensor profile against arbitrary "
            "baseline and candidate source revisions."
        )
    )
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument(
        "--baseline-update-mode",
        choices=("legacy", "atomic-presharded"),
        default="legacy",
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument(
        "--candidate-update-mode",
        choices=("legacy", "atomic-presharded"),
        default="legacy",
    )
    parser.add_argument("--candidate-fault-injection", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
