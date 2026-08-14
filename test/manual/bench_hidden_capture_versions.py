#!/usr/bin/env python3
"""Reproducible comparison of two hidden-capture revisions.

This orchestrator deliberately runs the repo-resident
``bench_hidden_capture_matrix.py`` as a script for every leg.  The driver
revision stays fixed while only the child server worktree changes.  By default,
each cell stays on one GPU for all four version/capture arms.  Different cells
are sharded across the selected GPUs, with complementary orders balancing
capture and version load in each concurrent wave::

    GPU A, cell X: baseline/off, baseline/on, candidate/off, candidate/on
    GPU B, cell Y: candidate/on, candidate/off, baseline/on, baseline/off

Use ``--cross-gpu-repeat`` only when a material card-dependent discrepancy
needs confirmation.  It uses the original two-GPU, eight-arm protocol for each
cell::

    GPU A: baseline/off, baseline/on, candidate/off, candidate/on
    GPU B: candidate/on, candidate/off, baseline/on, baseline/off

The frozen ``version-comparison-v1`` cells live in the matrix module.  Changing
their workload is a protocol change and requires a new suite name.

Run a formal comparison from a clean harness checkout (diagnostic subsets add
``--only-cells`` but are never labelled a formal full suite)::

    python3 test/manual/bench_hidden_capture_versions.py \
      --baseline-root /clean/worktree/a --baseline-revision <sha-a> \
      --baseline-label version_a \
      --candidate-root /clean/worktree/b --candidate-revision <sha-b> \
      --candidate-label version_b \
      --out-dir /new/artifact/directory --gpus 0,1 --numa-node 0 \
      --mooncake-python-root /compatible/mooncake

The script refuses dirty source trees, duplicate GPUs, ambiguous labels, wrong
revisions/Mooncake bindings, request-fingerprint drift, incomplete coverage or
drain, and selected-GPU residue.  It never overwrites a prior output directory.
``run_manifest.json`` freezes sources, scripts, hardware and scheduling;
``attempts.jsonl`` retains every arm; ``comparison_summary.json`` contains
capture deltas and aggregate geometric means; ``artifact_index.json`` hashes
all raw evidence.  Future comparisons should change only roots, exact SHAs,
labels, GPU selection and the new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

SUITE = "version-comparison-v1"
SUITE_CELLS = (
    "v1_cold_low",
    "v1_cold_high",
    "v1_warm_low",
    "v1_warm_high",
    "v1_warm_saturation",
    "v1_long_chunked",
    "v1_short_low",
    "v1_graph_off_high",
    "v1_chunked_off_high",
    "v1_bare_high",
)

COMMON_METRICS = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "median_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "accept_length",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict:
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True
        ).strip()
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
    ).splitlines()
    digest = hashlib.sha256(diff)
    for relative in sorted(untracked):
        path = root / relative
        digest.update(relative.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "root": str(root),
        "revision": revision,
        "dirty": bool(diff or untracked),
        "diff_sha256": digest.hexdigest(),
        "untracked": sorted(untracked),
    }


def _hardware_inventory() -> dict:
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id,"
            "power.limit,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ],
        "gpu_topology": ["nvidia-smi", "topo", "-m"],
        "numa_hardware": ["numactl", "--hardware"],
        "uname": ["uname", "-a"],
    }
    inventory = {}
    for name, command in commands.items():
        try:
            inventory[name] = subprocess.check_output(
                command, text=True, stderr=subprocess.STDOUT, timeout=30
            )
        except Exception as error:
            inventory[name] = f"readback-error: {error}"
    return inventory


def _mooncake_identity(python_root: Path) -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(python_root), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    }
    code = """
import hashlib, importlib.metadata, json, pathlib
import mooncake
from mooncake.store import ReplicateConfig
master = pathlib.Path(mooncake.__file__).parent / 'mooncake_master'
digest = hashlib.sha256(master.read_bytes()).hexdigest()
print(json.dumps({
    'version': importlib.metadata.version('mooncake-transfer-engine'),
    'module_path': mooncake.__file__,
    'master_path': str(master),
    'master_sha256': digest,
    'replicate_config_fields': sorted(
        name for name in dir(ReplicateConfig()) if not name.startswith('_')
    ),
}))
"""
    return json.loads(
        subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    )


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


def _schedule(baseline: dict, candidate: dict) -> tuple[tuple[dict, bool], ...]:
    return (
        (baseline, False),
        (baseline, True),
        (candidate, False),
        (candidate, True),
    )


def crossed_schedules(
    baseline: dict, candidate: dict
) -> tuple[tuple[tuple[dict, bool], ...], tuple[tuple[dict, bool], ...]]:
    return (
        _schedule(baseline, candidate),
        tuple(reversed(_schedule(baseline, candidate))),
    )


def comparison_waves(cells, gpus, baseline, candidate, cross_gpu_repeat=False):
    """Build deterministic waves while keeping a default cell on one GPU."""
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("comparison waves require distinct GPUs")
    if cross_gpu_repeat:
        if len(gpus) != 2:
            raise ValueError("cross-GPU repetition requires exactly two GPUs")
        schedules = crossed_schedules(baseline, candidate)
        return tuple(
            tuple(
                {
                    "cell_index": cell_index,
                    "cell": cell,
                    "step": step,
                    "worker": worker,
                    "gpu": gpu,
                    "revision": schedules[worker][step][0],
                    "capture": schedules[worker][step][1],
                }
                for worker, gpu in enumerate(gpus)
            )
            for cell_index, cell in enumerate(cells)
            for step in range(len(schedules[0]))
        )

    forward = _schedule(baseline, candidate)
    reverse = tuple(reversed(forward))
    waves = []
    for batch_start in range(0, len(cells), len(gpus)):
        batch = tuple(enumerate(cells[batch_start : batch_start + len(gpus)]))
        for step in range(len(forward)):
            wave = []
            for worker, (batch_offset, cell) in enumerate(batch):
                schedule = forward if worker % 2 == 0 else reverse
                revision, capture = schedule[step]
                wave.append(
                    {
                        "cell_index": batch_start + batch_offset,
                        "cell": cell,
                        "step": step,
                        "worker": worker,
                        "gpu": gpus[worker],
                        "revision": revision,
                        "capture": capture,
                    }
                )
            waves.append(tuple(wave))
    return tuple(waves)


def _load_single_row(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise AssertionError(f"expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def _row_gate(
    row: dict, expected_revision: str, expected_cell: str, expected_gpu: str
) -> list[str]:
    failures = []
    if row.get("status") != "ok":
        failures.append(f"status={row.get('status')}")
    source = row.get("server_source") or row.get("source") or {}
    if source.get("git_revision") != expected_revision:
        failures.append(
            f"revision={source.get('git_revision')} expected={expected_revision}"
        )
    if source.get("worktree_dirty"):
        failures.append("server source is dirty")
    if row.get("name", "").split("|", 1)[0] != expected_cell:
        failures.append(f"cell={row.get('name')} expected={expected_cell}")
    if row.get("selected_physical_gpu_ids") != [expected_gpu]:
        failures.append(
            "selected physical GPU="
            f"{row.get('selected_physical_gpu_ids')} expected={[expected_gpu]}"
        )
    measured = row.get("measured_request_set") or {}
    if measured.get("completed") != measured.get("requested"):
        failures.append(
            f"completed={measured.get('completed')}/{measured.get('requested')}"
        )
    if measured.get("inner_warmup_requests") != 0:
        failures.append("measured inner warmup is not zero")
    if not (measured.get("fingerprint") or {}).get("sha256"):
        failures.append("missing request-set fingerprint")
    bench = row.get("bench") or {}
    if bench.get("completed") != measured.get("completed"):
        failures.append(
            f"bench.completed={bench.get('completed')} "
            f"measured.completed={measured.get('completed')}"
        )
    missing_metrics = [metric for metric in COMMON_METRICS if bench.get(metric) is None]
    if missing_metrics:
        failures.append(f"missing benchmark metrics={missing_metrics}")
    if row.get("cache_hit_rate_pct") is None:
        failures.append("missing cache hit rate")
    if not (row.get("gpu_resources") or {}).get("samples"):
        failures.append("missing measured GPU resource samples")
    process_resources = row.get("process_resources") or {}
    if not process_resources.get("samples"):
        failures.append("missing measured process resource samples")
    if not isinstance(process_resources.get("host_net_pernic_delta"), dict):
        failures.append("missing per-NIC network readback")
    outer_warmup = row.get("outer_warmup_request_set") or {}
    if outer_warmup.get("completed") != outer_warmup.get("requested"):
        failures.append(
            "outer warmup completed="
            f"{outer_warmup.get('completed')}/{outer_warmup.get('requested')}"
        )
    if outer_warmup.get("inner_warmup_requests") != 0:
        failures.append("outer warmup inner warmup is not zero")
    if not (outer_warmup.get("fingerprint") or {}).get("sha256"):
        failures.append("missing outer warmup request-set fingerprint")
    if row.get("cell", {}).get("capture"):
        pre_measurement_requests = outer_warmup.get(
            "completed_generate_requests_before_measurement"
        )
        if not isinstance(pre_measurement_requests, int):
            failures.append("missing pre-measurement access-log request count")
        elif pre_measurement_requests < outer_warmup.get("completed", 0):
            failures.append(
                "pre-measurement access-log request count is smaller than "
                f"outer warmup: {pre_measurement_requests}"
            )
        manifest_baseline_count = (row.get("manifest_baseline") or {}).get("count")
        if manifest_baseline_count != pre_measurement_requests:
            failures.append(
                "manifest baseline is not bound to the pre-measurement "
                f"access-log request set: manifest={manifest_baseline_count}, "
                f"access_log={pre_measurement_requests}"
            )
        if row.get("coverage_source") != "mooncake_manifest_delta":
            failures.append(f"coverage_source={row.get('coverage_source')}")
        if row.get("export_coverage_frac") != 1.0:
            failures.append(f"coverage={row.get('export_coverage_frac')}")
        if not row.get("warmup_drained"):
            failures.append("outer warmup did not drain")
        if not row.get("measured_drained"):
            failures.append("measured export did not drain")
        if not (row.get("drain_gpu_resources") or {}).get("samples"):
            failures.append("missing drain GPU resource samples")
        if not (row.get("drain_process_resources") or {}).get("samples"):
            failures.append("missing drain process resource samples")
    if not row.get("gpu_process_settled"):
        failures.append("selected GPU process set did not settle")
    if row.get("gpu_process_introduced_after_cleanup"):
        failures.append(
            "selected GPU residue=" f"{row.get('gpu_process_introduced_after_cleanup')}"
        )
    return failures


def _artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _build_command(
    *,
    matrix_script: Path,
    driver_root: Path,
    revision: dict,
    capture: bool,
    cell: str,
    gpu: str,
    shard: int,
    out: Path,
    artifact_dir: Path,
    numa_node: int | None,
    mooncake_python_root: Path,
    expected_mooncake_version: str,
) -> tuple[list[str], dict]:
    command = [
        sys.executable,
        str(matrix_script),
        "--out",
        str(out),
        "--artifact-dir",
        str(artifact_dir),
        "--cells",
        SUITE,
        "--only",
        cell,
        "--capture-only",
        "on" if capture else "off",
        "--repeat-only",
        "0",
        "--shard",
        str(shard),
        "--server-source-root",
        revision["root"],
        "--expected-server-revision",
        revision["revision"],
        "--expected-mooncake-version",
        expected_mooncake_version,
    ]
    if numa_node is not None:
        if not shutil.which("numactl"):
            raise RuntimeError("--numa-node requested but numactl is unavailable")
        command = [
            "numactl",
            f"--cpunodebind={numa_node}",
            *command,
        ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": os.pathsep.join(
            [
                str(driver_root / "python"),
                str(mooncake_python_root),
                os.environ.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep),
        "MOONCAKE_MASTER_BIN": str(mooncake_python_root / "mooncake/mooncake_master"),
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return command, env


def _launch_attempt(spec: dict, *, resume: bool, dry_run: bool) -> dict:
    out = Path(spec["out"])
    if out.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite attempt artifact: {out}")
        row = _load_single_row(out)
        return {
            **spec,
            "reused": True,
            "exit_code": 0,
            "row": row,
            "row_gate_failures": _row_gate(
                row, spec["revision"], spec["cell"], spec["gpu"]
            ),
            "output": _artifact_record(out),
        }
    if dry_run:
        return {**spec, "reused": False, "dry_run": True}

    stdout_path = Path(spec["stdout"])
    stderr_path = Path(spec["stderr"])
    stdout_file = stdout_path.open("x")
    stderr_file = stderr_path.open("x")
    started = time.time()
    process = subprocess.Popen(
        spec["command"],
        env=spec["env"],
        cwd=spec["driver_root"],
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
    )
    return {
        **spec,
        "reused": False,
        "process": process,
        "stdout_file": stdout_file,
        "stderr_file": stderr_file,
        "started_unix_s": started,
    }


def _finish_attempt(running: dict) -> dict:
    if "process" not in running:
        return running
    process = running.pop("process")
    stdout_file = running.pop("stdout_file")
    stderr_file = running.pop("stderr_file")
    exit_code = process.wait()
    stdout_file.close()
    stderr_file.close()
    finished = time.time()
    out = Path(running["out"])
    result = {
        **running,
        "exit_code": exit_code,
        "finished_unix_s": finished,
        "wall_s": finished - running["started_unix_s"],
        "stdout_artifact": _artifact_record(Path(running["stdout"])),
        "stderr_artifact": _artifact_record(Path(running["stderr"])),
    }
    if out.exists():
        result["output"] = _artifact_record(out)
        try:
            row = _load_single_row(out)
            result["row"] = row
            result["row_gate_failures"] = _row_gate(
                row, running["revision"], running["cell"], running["gpu"]
            )
        except Exception as error:
            result["row_gate_failures"] = [f"row parse: {error}"]
    else:
        result["row_gate_failures"] = ["missing JSONL output"]
    if exit_code != 0:
        result["row_gate_failures"].append(f"driver exit={exit_code}")
    return result


def _attempt_public(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"env"} and not key.startswith("_")
    }


def _ratio_pct(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return (float(numerator) / float(denominator) - 1.0) * 100.0


def _geomean_ratio_pct(pairs):
    ratios = [float(a) / float(b) for a, b in pairs if a and b]
    if not ratios:
        return None
    return (math.exp(sum(math.log(value) for value in ratios) / len(ratios)) - 1) * 100


def _geomean(values):
    usable = [float(value) for value in values if value not in (None, 0)]
    if not usable:
        return None
    return math.exp(sum(math.log(value) for value in usable) / len(usable))


def _ratio_of_geomeans_pct(numerators, denominators):
    numerator = _geomean(numerators)
    denominator = _geomean(denominators)
    return _ratio_pct(numerator, denominator)


def _capture_did_pct(candidate_pairs, baseline_pairs):
    candidate_ratio = _geomean(
        float(on) / float(off) for on, off in candidate_pairs if on and off
    )
    baseline_ratio = _geomean(
        float(on) / float(off) for on, off in baseline_pairs if on and off
    )
    return _ratio_pct(candidate_ratio, baseline_ratio)


def summarize_attempts(
    attempts: list[dict],
    labels: tuple[str, str],
    cells: tuple[str, ...] = SUITE_CELLS,
    scheduling_mode: str = "fixed-cell-gpu",
) -> dict:
    baseline_label, candidate_label = labels
    summary = {"suite": SUITE, "cells": {}, "all_gates_pass": True}
    for cell in cells:
        rows = {}
        failures = []
        for attempt in attempts:
            if attempt.get("cell") != cell or "row" not in attempt:
                continue
            key = (
                attempt["gpu"],
                attempt["version_label"],
                bool(attempt["capture"]),
            )
            rows[key] = attempt["row"]
            failures.extend(
                f"{attempt['attempt_id']}: {failure}"
                for failure in attempt.get("row_gate_failures", [])
            )
        fingerprints = {
            (row.get("measured_request_set") or {}).get("fingerprint", {}).get("sha256")
            for row in rows.values()
        }
        fingerprints.discard(None)
        if len(fingerprints) != 1:
            failures.append(f"request fingerprints differ: {sorted(fingerprints)}")
        outer_warmup_fingerprints = {
            (row.get("outer_warmup_request_set") or {})
            .get("fingerprint", {})
            .get("sha256")
            for row in rows.values()
        }
        outer_warmup_fingerprints.discard(None)
        if len(outer_warmup_fingerprints) != 1:
            failures.append(
                "outer warmup request fingerprints differ: "
                f"{sorted(outer_warmup_fingerprints)}"
            )
        expected_keys = {
            (
                attempt["gpu"],
                attempt["version_label"],
                bool(attempt["capture"]),
            )
            for attempt in attempts
            if attempt.get("cell") == cell
        }
        missing = sorted(expected_keys - set(rows))
        if missing:
            failures.append(f"missing arms: {missing}")

        cell_summary = {
            "comparable": not failures,
            "scheduling_mode": scheduling_mode,
            "gate_failures": failures,
            "request_fingerprint_sha256": next(iter(fingerprints), None),
            "outer_warmup_fingerprint_sha256": next(
                iter(outer_warmup_fingerprints), None
            ),
            "metrics": {},
            "coverage": {},
            "resources": {},
        }
        if failures:
            summary["all_gates_pass"] = False

        gpus = sorted({key[0] for key in rows})
        for metric in COMMON_METRICS:
            per_gpu = {}
            for gpu in gpus:

                def value(label, capture):
                    return (rows.get((gpu, label, capture), {}).get("bench") or {}).get(
                        metric
                    )

                bo = value(baseline_label, False)
                bn = value(baseline_label, True)
                co = value(candidate_label, False)
                cn = value(candidate_label, True)
                per_gpu[gpu] = {
                    "baseline_off": bo,
                    "baseline_on": bn,
                    "candidate_off": co,
                    "candidate_on": cn,
                    "baseline_capture_cost_pct": _ratio_pct(bn, bo),
                    "candidate_capture_cost_pct": _ratio_pct(cn, co),
                    "candidate_vs_baseline_off_pct": _ratio_pct(co, bo),
                    "candidate_vs_baseline_on_pct": _ratio_pct(cn, bn),
                }
            baseline_pairs = []
            candidate_pairs = []
            for gpu in gpus:
                baseline_off = (
                    rows.get((gpu, baseline_label, False), {}).get("bench") or {}
                ).get(metric)
                baseline_on = (
                    rows.get((gpu, baseline_label, True), {}).get("bench") or {}
                ).get(metric)
                candidate_off = (
                    rows.get((gpu, candidate_label, False), {}).get("bench") or {}
                ).get(metric)
                candidate_on = (
                    rows.get((gpu, candidate_label, True), {}).get("bench") or {}
                ).get(metric)
                if baseline_on not in (None, 0) and baseline_off not in (None, 0):
                    baseline_pairs.append((baseline_on, baseline_off))
                if candidate_on not in (None, 0) and candidate_off not in (None, 0):
                    candidate_pairs.append((candidate_on, candidate_off))
            baseline_off_values = [
                (row.get("bench") or {}).get(metric)
                for (gpu, label, capture), row in rows.items()
                if label == baseline_label and not capture
            ]
            baseline_on_values = [
                (row.get("bench") or {}).get(metric)
                for (gpu, label, capture), row in rows.items()
                if label == baseline_label and capture
            ]
            candidate_off_values = [
                (row.get("bench") or {}).get(metric)
                for (gpu, label, capture), row in rows.items()
                if label == candidate_label and not capture
            ]
            candidate_on_values = [
                (row.get("bench") or {}).get(metric)
                for (gpu, label, capture), row in rows.items()
                if label == candidate_label and capture
            ]
            aggregate_geomean = {
                "baseline_capture_cost_pct": _geomean_ratio_pct(baseline_pairs),
                "candidate_capture_cost_pct": _geomean_ratio_pct(candidate_pairs),
                "candidate_vs_baseline_off_pct": _ratio_of_geomeans_pct(
                    candidate_off_values, baseline_off_values
                ),
                "candidate_vs_baseline_on_pct": _ratio_of_geomeans_pct(
                    candidate_on_values, baseline_on_values
                ),
                "capture_cost_difference_in_differences_pct": _capture_did_pct(
                    candidate_pairs, baseline_pairs
                ),
            }
            cell_summary["metrics"][metric] = {
                "per_gpu": per_gpu,
                "aggregate_geomean": aggregate_geomean,
                "crossed_geomean": (
                    aggregate_geomean if scheduling_mode == "cross-gpu-repeat" else None
                ),
            }

        for gpu, label in sorted({(key[0], key[1]) for key in rows}):
            on = rows.get((gpu, label, True), {})
            cell_summary["coverage"][f"gpu{gpu}:{label}"] = {
                "coverage": on.get("export_coverage_frac"),
                "drain_s": on.get("measured_drain_s"),
                "cache_hit_rate_pct": on.get("cache_hit_rate_pct"),
                "counter_delta": on.get("measured_counter_delta"),
                "last_log_counters": on.get("capture_counters_last_log"),
            }
            for capture in (False, True):
                row = rows.get((gpu, label, capture), {})
                process = row.get("process_resources") or {}
                gpu_resources = row.get("gpu_resources") or {}
                cell_summary["resources"][f"gpu{gpu}:{label}:cap={int(capture)}"] = {
                    "startup_s": row.get("server_startup_s"),
                    "rss_peak_bytes": process.get("rss_peak_bytes"),
                    "cpu_mean_one_core_units": process.get("cpu_mean_one_core_units"),
                    "process_read_bytes_delta": process.get("process_read_bytes_delta"),
                    "process_write_bytes_delta": process.get(
                        "process_write_bytes_delta"
                    ),
                    "host_net_pernic_delta": process.get("host_net_pernic_delta"),
                    "gpu_memory_max_single_mb": gpu_resources.get(
                        "memory_max_single_mb"
                    ),
                    "gpu_util_mean_pct": gpu_resources.get("gpu_util_mean_pct"),
                    "power_mean_w": gpu_resources.get("power_mean_w"),
                }
        summary["cells"][cell] = cell_summary
    return summary


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--numa-node", type=int, default=None)
    parser.add_argument("--shard-base", type=int, default=20)
    parser.add_argument("--mooncake-python-root", required=True)
    parser.add_argument("--expected-mooncake-version", default="0.3.12.post1")
    parser.add_argument(
        "--only-cells",
        default=None,
        help="comma-separated diagnosis subset; omitted is the formal full suite",
    )
    parser.add_argument(
        "--cross-gpu-repeat",
        action="store_true",
        help=(
            "repeat each version on the other GPU; opt in only to investigate "
            "a material card-dependent discrepancy"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()

    driver_root = Path(__file__).resolve().parents[2]
    matrix_script = driver_root / "test/manual/bench_hidden_capture_matrix.py"
    driver = _git_identity(driver_root)
    baseline = {
        **_git_identity(Path(opts.baseline_root)),
        "label": opts.baseline_label,
    }
    candidate = {
        **_git_identity(Path(opts.candidate_root)),
        "label": opts.candidate_label,
    }
    mooncake_python_root = Path(opts.mooncake_python_root).resolve()
    mooncake = _mooncake_identity(mooncake_python_root)
    if mooncake["version"] != opts.expected_mooncake_version:
        raise SystemExit(
            "Mooncake version mismatch: "
            f"expected={opts.expected_mooncake_version}, actual={mooncake['version']}"
        )
    if "group_ids" not in mooncake["replicate_config_fields"]:
        raise SystemExit(f"Mooncake binding lacks group_ids: {mooncake}")
    for identity, expected in (
        (baseline, opts.baseline_revision),
        (candidate, opts.candidate_revision),
    ):
        if identity["revision"] != expected:
            raise SystemExit(
                f"{identity['label']} revision mismatch: "
                f"expected={expected}, actual={identity['revision']}"
            )
        if identity["dirty"]:
            raise SystemExit(f"{identity['label']} worktree is dirty: {identity}")
    if driver["dirty"]:
        raise SystemExit(f"driver worktree must be committed and clean: {driver}")

    gpus = tuple(part.strip() for part in opts.gpus.split(",") if part.strip())
    if (
        not gpus
        or len(set(gpus)) != len(gpus)
        or not all(gpu.isdigit() for gpu in gpus)
    ):
        raise SystemExit("--gpus must name one or more distinct physical GPU indices")
    if opts.cross_gpu_repeat and len(gpus) != 2:
        raise SystemExit("--cross-gpu-repeat requires exactly two GPUs")
    labels = (opts.baseline_label, opts.candidate_label)
    if labels[0] == labels[1] or _safe_label(labels[0]) == _safe_label(labels[1]):
        raise SystemExit(
            "baseline and candidate labels must be distinct after sanitizing"
        )
    cells = SUITE_CELLS
    if opts.only_cells:
        requested = tuple(
            part.strip() for part in opts.only_cells.split(",") if part.strip()
        )
        unknown = sorted(set(requested) - set(SUITE_CELLS))
        if unknown:
            raise SystemExit(f"unknown frozen cells: {unknown}")
        cells = requested

    out_dir = Path(opts.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not opts.resume:
        raise SystemExit(f"refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = out_dir / "run_manifest.json"
    attempts_path = out_dir / "attempts.jsonl"
    summary_path = out_dir / "comparison_summary.json"

    run_manifest = {
        "schema": "hidden-capture-version-comparison-v1",
        "suite": SUITE,
        "suite_cells": list(SUITE_CELLS),
        "selected_cells": list(cells),
        "formal_full_suite": cells == SUITE_CELLS,
        "driver": driver,
        "driver_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "matrix_script": {
            "path": str(matrix_script),
            "sha256": _sha256(matrix_script),
        },
        "baseline": baseline,
        "candidate": candidate,
        "mooncake": mooncake,
        "mooncake_python_root": str(mooncake_python_root),
        "gpus": list(gpus),
        "scheduling_mode": (
            "cross-gpu-repeat" if opts.cross_gpu_repeat else "fixed-cell-gpu"
        ),
        "numa_node": opts.numa_node,
        "numa_policy": (
            "CPU node pinned; memory placement is first-touch because this "
            "container does not permit set_mempolicy"
            if opts.numa_node is not None
            else "unbound"
        ),
        "shard_base": opts.shard_base,
        "host_network_accounting": (
            "host NIC counters are pair-shared because complementary GPU legs "
            "run concurrently; process IO and capture byte counters are per leg"
        ),
        "hardware": _hardware_inventory(),
        "started_unix_s": time.time(),
    }
    _write_json(run_manifest_path, run_manifest)

    scheduling_mode = "cross-gpu-repeat" if opts.cross_gpu_repeat else "fixed-cell-gpu"
    waves = comparison_waves(
        cells,
        gpus,
        baseline,
        candidate,
        cross_gpu_repeat=opts.cross_gpu_repeat,
    )
    attempts = []
    for wave_index, wave in enumerate(waves):
        specs = []
        for arm in wave:
            cell = arm["cell"]
            step = arm["step"]
            worker = arm["worker"]
            gpu = arm["gpu"]
            revision = arm["revision"]
            capture = arm["capture"]
            attempt_id = _safe_label(
                f"{cell}.step{step}.gpu{gpu}.{revision['label']}.cap{int(capture)}"
            )
            attempt_dir = out_dir / attempt_id
            attempt_dir.mkdir(exist_ok=True)
            out = attempt_dir / "row.jsonl"
            artifacts = attempt_dir / "raw"
            command, env = _build_command(
                matrix_script=matrix_script,
                driver_root=driver_root,
                revision=revision,
                capture=capture,
                cell=cell,
                gpu=gpu,
                shard=opts.shard_base + worker,
                out=out,
                artifact_dir=artifacts,
                numa_node=opts.numa_node,
                mooncake_python_root=mooncake_python_root,
                expected_mooncake_version=opts.expected_mooncake_version,
            )
            specs.append(
                {
                    "attempt_id": attempt_id,
                    "cell": cell,
                    "cell_index": arm["cell_index"],
                    "wave": wave_index,
                    "step": step,
                    "worker": worker,
                    "gpu": gpu,
                    "version_label": revision["label"],
                    "revision": revision["revision"],
                    "capture": capture,
                    "driver_root": str(driver_root),
                    "command": command,
                    "command_shell": shlex.join(command),
                    "env": env,
                    "out": str(out),
                    "artifact_dir": str(artifacts),
                    "stdout": str(attempt_dir / "driver.stdout"),
                    "stderr": str(attempt_dir / "driver.stderr"),
                }
            )
        print(
            json.dumps(
                {
                    "event": "start_wave",
                    "wave": wave_index,
                    "cells": [spec["cell"] for spec in specs],
                    "attempts": [spec["attempt_id"] for spec in specs],
                }
            ),
            flush=True,
        )
        running = [
            _launch_attempt(spec, resume=opts.resume, dry_run=opts.dry_run)
            for spec in specs
        ]
        finished = [_finish_attempt(record) for record in running]
        for record in finished:
            attempts.append(record)
            if not opts.dry_run:
                with attempts_path.open("a") as stream:
                    stream.write(json.dumps(_attempt_public(record)) + "\n")
        print(
            json.dumps(
                {
                    "event": "finish_wave",
                    "wave": wave_index,
                    "results": [
                        {
                            "attempt": record["attempt_id"],
                            "exit": record.get("exit_code"),
                            "gate_failures": record.get("row_gate_failures"),
                        }
                        for record in finished
                    ],
                }
            ),
            flush=True,
        )

    if opts.dry_run:
        _write_json(
            out_dir / "dry_run_schedule.json", [_attempt_public(a) for a in attempts]
        )
        return 0

    summary = summarize_attempts(
        attempts,
        labels,
        cells,
        scheduling_mode=scheduling_mode,
    )
    _write_json(summary_path, summary)
    run_manifest["finished_unix_s"] = time.time()
    run_manifest["all_gates_pass"] = summary["all_gates_pass"]
    run_manifest["attempt_count"] = len(attempts)
    run_manifest["comparison_summary"] = _artifact_record(summary_path)
    _write_json(run_manifest_path, run_manifest)

    index = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            index.append(_artifact_record(path))
    _write_json(out_dir / "artifact_index.json", index)
    return 0 if summary["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
