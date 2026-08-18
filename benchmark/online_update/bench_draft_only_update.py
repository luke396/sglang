#!/usr/bin/env python3
"""Benchmark repeated draft-only tensor updates against a running SGLang server.

The producer loads both checkpoints on CPU once.  Every update gets a fresh
filename-backed shared-memory payload before generation is paused, so payload
preparation is measured separately from the service interruption.  Background
native `/generate` traffic records the same TTFT/TPOT definitions used by
``sglang.benchmark.serving`` plus exact speculative acceptance counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pynvml
import requests
from safetensors.torch import load_file

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.draft_tensor_preshard import (
    preshard_dflash_named_tensors,
)

MIB = 1024 * 1024


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "values": values,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "population_variance": statistics.pvariance(values),
        "population_stdev": statistics.pstdev(values),
        "sample_variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if mean else None
        ),
    }


def nullable_distribution(values: list[Optional[float]]) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return distribution(present) | {"missing_count": len(values) - len(present)}


def digest_mapping(values: dict[str, str]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def post(base_url: str, path: str, payload: dict, timeout: float = 300) -> dict:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"{path} returned {response.status_code}: {body}")
    return body


def timed_post(base_url: str, path: str, payload: dict) -> dict[str, Any]:
    started = time.perf_counter()
    response = post(base_url, path, payload)
    ended = time.perf_counter()
    return {
        "started": started,
        "ended": ended,
        "seconds": ended - started,
        "response": response,
    }


@dataclass
class ChecksumSnapshot:
    target: list[dict[str, str]]
    draft: list[dict[str, str]]
    engine: Any

    def summary(self) -> dict[str, Any]:
        return {
            "ranks": [
                {
                    "rank": rank,
                    "target_count": len(self.target[rank]),
                    "draft_count": len(self.draft[rank]),
                    "target_digest": digest_mapping(self.target[rank]),
                    "draft_digest": digest_mapping(self.draft[rank]),
                }
                for rank in range(len(self.target))
            ],
            "engine": self.engine,
        }


def checksum_snapshot(base_url: str) -> ChecksumSnapshot:
    body = post(base_url, "/weights_checker", {})
    target = []
    draft = []
    for rank in body["ranks"]:
        checksums = rank["checksums"]
        target.append(
            {
                name: value
                for name, value in checksums.items()
                if not name.startswith("draft.")
            }
        )
        draft.append(
            {
                name: value
                for name, value in checksums.items()
                if name.startswith("draft.")
            }
        )
    return ChecksumSnapshot(
        target=target,
        draft=draft,
        engine=body.get("per_engine_checksum"),
    )


def compare_checksums(
    reference: ChecksumSnapshot, current: ChecksumSnapshot
) -> dict[str, Any]:
    if len(reference.target) != len(current.target):
        return {
            "rank_count_matches": False,
            "reference_rank_count": len(reference.target),
            "current_rank_count": len(current.target),
        }
    ranks = []
    for rank in range(len(reference.target)):
        target_names = sorted(
            name
            for name in set(reference.target[rank]) | set(current.target[rank])
            if reference.target[rank].get(name) != current.target[rank].get(name)
        )
        draft_names = sorted(
            name
            for name in set(reference.draft[rank]) | set(current.draft[rank])
            if reference.draft[rank].get(name) != current.draft[rank].get(name)
        )
        ranks.append(
            {
                "rank": rank,
                "target_changed_count": len(target_names),
                "target_changed_names": target_names,
                "draft_changed_count": len(draft_names),
                "draft_changed_names": draft_names,
            }
        )
    return {
        "rank_count_matches": True,
        "ranks": ranks,
        "engine_matches": reference.engine == current.engine,
    }


class NvmlSampler:
    def __init__(self, gpu_indices: list[int], interval_seconds: float):
        pynvml.nvmlInit()
        self.gpu_indices = gpu_indices
        self.handles = {
            index: pynvml.nvmlDeviceGetHandleByIndex(index) for index in gpu_indices
        }
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, dict[int, int]]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> dict[int, int]:
        return {
            index: pynvml.nvmlDeviceGetMemoryInfo(handle).used
            for index, handle in self.handles.items()
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append((time.perf_counter(), self._read()))
            time.sleep(self.interval_seconds)

    def start(self) -> None:
        self.thread.start()

    def snapshot(self) -> dict[int, int]:
        return self._read()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        pynvml.nvmlShutdown()

    def window(
        self,
        start: float,
        end: float,
        baseline_override: Optional[dict[int, int]] = None,
    ) -> dict[str, Any]:
        baseline_start = start - 0.050
        baseline = [
            sample for stamp, sample in self.samples if baseline_start <= stamp < start
        ]
        during = [sample for stamp, sample in self.samples if start <= stamp <= end]
        result = {}
        for index in self.gpu_indices:
            baseline_values = [sample[index] for sample in baseline]
            during_values = [sample[index] for sample in during]
            baseline_value = (
                baseline_override[index]
                if baseline_override is not None
                else (
                    int(statistics.median(baseline_values)) if baseline_values else None
                )
            )
            peak_value = max(during_values) if during_values else None
            result[str(index)] = {
                "baseline_bytes": baseline_value,
                "peak_bytes": peak_value,
                "peak_delta_bytes": (
                    peak_value - baseline_value
                    if baseline_value is not None and peak_value is not None
                    else None
                ),
                "sample_count": len(during_values),
            }
        return result


class TrafficRunner:
    def __init__(
        self,
        *,
        base_url: str,
        concurrency: int,
        output_tokens: int,
        timeout_seconds: float,
    ):
        self.base_url = base_url
        self.concurrency = concurrency
        self.output_tokens = output_tokens
        self.timeout_seconds = timeout_seconds
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.phase = "warmup"
        self.active = 0
        self.sequence = 0
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.adjusted_itls_by_kind: dict[str, list[float]] = {
            "baseline": [],
            "updated": [],
            "restored": [],
        }
        self.transition: Optional[dict[str, Any]] = None
        self.transition_results: list[dict[str, Any]] = []
        self.threads = [
            threading.Thread(target=self._worker, args=(index,), daemon=True)
            for index in range(concurrency)
        ]

    @staticmethod
    def phase_kind(phase: str) -> Optional[str]:
        if phase == "baseline-initial":
            return "baseline"
        if phase.startswith("updated-"):
            return "updated"
        if phase.startswith("restored-"):
            return "restored"
        return None

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def begin_transition(self, label: str, started: float) -> None:
        with self.lock:
            self.transition = {
                "label": label,
                "started": started,
                "ended": None,
                "raw_gap_ms": [],
                "adjusted_itl_ms": [],
            }

    def end_transition(self, ended: float) -> None:
        with self.lock:
            if self.transition is not None:
                self.transition["ended"] = ended

    def finish_transition(self) -> dict[str, Any]:
        with self.lock:
            transition = self.transition
            self.transition = None
            if transition is None:
                raise RuntimeError("no active traffic transition")
            self.transition_results.append(transition)
            return transition

    def _record_gap(
        self,
        *,
        left: float,
        right: float,
        new_tokens: int,
        adjusted_itl_ms: float,
    ) -> None:
        with self.lock:
            transition = self.transition
            if transition is None or transition["ended"] is None:
                return
            if left <= transition["ended"] and right >= transition["started"]:
                transition["raw_gap_ms"].append((right - left) * 1000)
                transition["adjusted_itl_ms"].extend([adjusted_itl_ms] * new_tokens)

    def _one_request(self, worker_index: int, session: requests.Session) -> None:
        with self.lock:
            phase_start = self.phase
            sequence = self.sequence
            self.sequence += 1
            self.active += 1

        prompt = (
            f"Worker {worker_index}, request {sequence}. Continue this sequence:"
            + " one two three four five six seven eight" * 16
        )
        started = time.perf_counter()
        first_token = None
        last_token = started
        last_output_len = 0
        adjusted_itls_ms: list[float] = []
        final_meta: dict[str, Any] = {}
        error = None
        try:
            with session.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": self.output_tokens,
                        "ignore_eos": True,
                    },
                    "stream": True,
                },
                stream=True,
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    payload = (
                        raw_line[5:].strip()
                        if raw_line.startswith(b"data:")
                        else raw_line
                    )
                    if payload == b"[DONE]":
                        continue
                    data = json.loads(payload)
                    meta = data.get("meta_info") or {}
                    final_meta.update(meta)
                    text = data.get("text")
                    output_len = int(meta.get("completion_tokens", last_output_len))
                    if not text or output_len <= last_output_len:
                        continue
                    now = time.perf_counter()
                    new_tokens = output_len - last_output_len
                    if first_token is None:
                        first_token = now
                    else:
                        adjusted_itl_ms = (now - last_token) * 1000 / new_tokens
                        adjusted_itls_ms.extend([adjusted_itl_ms] * new_tokens)
                        self._record_gap(
                            left=last_token,
                            right=now,
                            new_tokens=new_tokens,
                            adjusted_itl_ms=adjusted_itl_ms,
                        )
                    last_token = now
                    last_output_len = output_len
        except Exception as exc:
            error = repr(exc)
        ended = time.perf_counter()
        with self.lock:
            phase_end = self.phase
            self.active -= 1

        if error is not None or first_token is None or last_output_len <= 0:
            with self.lock:
                self.errors.append(error or "request returned no tokens")
            return

        ttft_ms = (first_token - started) * 1000
        latency_ms = (ended - started) * 1000
        tpot_ms = (
            (ended - first_token) * 1000 / (last_output_len - 1)
            if last_output_len > 1
            else 0.0
        )
        record = {
            "phase_start": phase_start,
            "phase_end": phase_end,
            "pure_phase": phase_start == phase_end,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "latency_ms": latency_ms,
            "output_tokens": last_output_len,
            "max_adjusted_itl_ms": max(adjusted_itls_ms, default=0.0),
            "spec_accept_rate": final_meta.get("spec_accept_rate"),
            "spec_accept_length": final_meta.get("spec_accept_length"),
            "spec_num_correct_drafts": final_meta.get("spec_num_correct_drafts"),
            "spec_num_proposed_drafts": final_meta.get("spec_num_proposed_drafts"),
            "spec_verify_ct": final_meta.get("spec_verify_ct"),
        }
        with self.lock:
            self.records.append(record)
            kind = self.phase_kind(phase_start) if phase_start == phase_end else None
            if kind is not None:
                self.adjusted_itls_by_kind[kind].extend(adjusted_itls_ms)

    def _worker(self, worker_index: int) -> None:
        session = requests.Session()
        try:
            while not self.stop_event.is_set():
                self._one_request(worker_index, session)
        finally:
            session.close()

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def wait_active(self, timeout: float = 30) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self.lock:
                active = self.active
            if active == self.concurrency:
                return
            time.sleep(0.005)
        raise TimeoutError(
            f"traffic did not reach concurrency={self.concurrency}; active={active}"
        )

    def phase_count(self, phase: str) -> int:
        with self.lock:
            return sum(
                record["pure_phase"] and record["phase_start"] == phase
                for record in self.records
            )

    def wait_phase(
        self, phase: str, *, min_samples: int, min_seconds: float, timeout: float
    ) -> None:
        started = time.perf_counter()
        deadline = started + timeout
        while time.perf_counter() < deadline:
            if (
                time.perf_counter() - started >= min_seconds
                and self.phase_count(phase) >= min_samples
            ):
                return
            time.sleep(0.025)
        raise TimeoutError(
            f"phase {phase!r} has {self.phase_count(phase)} pure requests; "
            f"wanted {min_samples} after {min_seconds}s"
        )

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=self.timeout_seconds + 10)
        alive = [
            index for index, thread in enumerate(self.threads) if thread.is_alive()
        ]
        if alive:
            raise RuntimeError(f"traffic threads did not stop: {alive}")

    def phase_summary(self, phase: str) -> dict[str, Any]:
        with self.lock:
            records = [
                record
                for record in self.records
                if record["pure_phase"] and record["phase_start"] == phase
            ]
        return summarize_request_records(records)

    def aggregate_summary(self, kind: str) -> dict[str, Any]:
        with self.lock:
            records = [
                record
                for record in self.records
                if record["pure_phase"]
                and self.phase_kind(record["phase_start"]) == kind
            ]
            itls = list(self.adjusted_itls_by_kind[kind])
        summary = summarize_request_records(records)
        summary["adjusted_itl_ms"] = distribution(itls)
        return summary


def summarize_request_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(record["spec_num_correct_drafts"] or 0) for record in records)
    proposed = sum(int(record["spec_num_proposed_drafts"] or 0) for record in records)
    verify = sum(int(record["spec_verify_ct"] or 0) for record in records)
    completion = sum(int(record["output_tokens"]) for record in records)
    return {
        "request_count": len(records),
        "ttft_ms": distribution([record["ttft_ms"] for record in records]),
        "tpot_ms": distribution([record["tpot_ms"] for record in records]),
        "e2e_latency_ms": distribution([record["latency_ms"] for record in records]),
        "per_request_max_adjusted_itl_ms": distribution(
            [record["max_adjusted_itl_ms"] for record in records]
        ),
        "spec": {
            "num_correct_drafts": correct,
            "num_proposed_drafts": proposed,
            "verify_ct": verify,
            "completion_tokens": completion,
            "accept_rate": correct / proposed if proposed else None,
            "accept_length": completion / verify if verify else None,
            "per_request_accept_rate": distribution(
                [
                    float(record["spec_accept_rate"])
                    for record in records
                    if record["spec_accept_rate"] is not None
                ]
            ),
            "per_request_accept_length": distribution(
                [
                    float(record["spec_accept_length"])
                    for record in records
                    if record["spec_accept_length"] is not None
                ]
            ),
        },
    }


def load_checkpoint(path: Path) -> tuple[dict, list[tuple[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    tensors = load_file(str(path), device="cpu")
    items = list(tensors.items())
    elapsed = time.perf_counter() - started
    sizes = [(name, tensor.numel() * tensor.element_size()) for name, tensor in items]
    return (
        tensors,
        items,
        {
            "path": str(path),
            "cpu_load_seconds": elapsed,
            "tensor_count": len(items),
            "checkpoint_bytes": sum(size for _, size in sizes),
            "largest_tensors": sorted(sizes, key=lambda item: item[1], reverse=True)[
                :10
            ],
        },
    )


def serialize_payloads(
    items: list[tuple[str, Any]], tp_size: int, update_mode: str
) -> tuple[list[str], dict[str, Any], list[list[tuple[str, Any]]]]:
    if update_mode == "atomic-presharded":
        per_rank_items, preshard_info = preshard_dflash_named_tensors(items, tp_size)
    else:
        per_rank_items = [items for _ in range(tp_size)]
        preshard_info = {
            "source_bytes": sum(
                tensor.numel() * tensor.element_size() for _, tensor in items
            ),
            "per_rank_bytes": [
                sum(tensor.numel() * tensor.element_size() for _, tensor in items)
                for _ in range(tp_size)
            ],
            "mode": "replicated-full-checkpoint",
        }

    per_rank_seconds = []
    payloads = []
    for rank_items in per_rank_items:
        started = time.perf_counter()
        payloads.append(
            MultiprocessingSerializer.serialize(
                rank_items,
                output_str=True,
                cpu_sharing_strategy="file_system",
            )
        )
        per_rank_seconds.append(time.perf_counter() - started)
    return (
        payloads,
        {
            "per_rank_seconds": per_rank_seconds,
            "total_seconds": sum(per_rank_seconds),
            "metadata_bytes": [len(payload) for payload in payloads],
            "preshard": preshard_info,
        },
        per_rank_items,
    )


def get_server_info(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/server_info", timeout=60)
    response.raise_for_status()
    body = response.json()
    return {
        "internal_states": body.get("internal_states"),
        "version": body.get("version"),
        "model_path": body.get("model_path"),
        "weight_version": body.get("weight_version"),
        "draft_weight_update": body.get("draft_weight_update"),
    }


def scan_graph_log(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    text = path.read_text(errors="replace")
    return {
        "path": str(path),
        "capture_begin_count": len(re.findall(r"Capture .* CUDA graph begin", text)),
        "decode_cuda_graph_true_count": len(
            re.findall(r"Decode batch.*cuda graph: True", text)
        ),
        "decode_cuda_graph_false_count": len(
            re.findall(r"Decode batch.*cuda graph: False", text)
        ),
        "scheduler_exception_count": text.count("Scheduler hit an exception"),
    }


def transition_update(
    *,
    base_url: str,
    traffic: TrafficRunner,
    sampler: NvmlSampler,
    payloads: list[str],
    weight_version: str,
    label: str,
    update_mode: str,
    recovery: bool = False,
) -> dict[str, Any]:
    traffic.set_phase(f"transition-{label}")
    traffic.wait_active()
    memory_before = sampler.snapshot()
    transition_started = time.perf_counter()
    traffic.begin_transition(label, transition_started)
    common_payload = {
        "serialized_named_tensors": payloads,
        "load_format": None,
        "flush_cache": False,
        "draft_only": True,
        "torch_empty_cache": False,
        "weight_version": weight_version,
    }
    if update_mode == "atomic-presharded":
        common_payload.update(
            {
                "tensors_are_pre_sharded": True,
                "collect_phase_timings": True,
                "recovery": recovery,
            }
        )

    pause = None
    update = None
    cont = None
    if update_mode == "atomic-presharded":
        update = timed_post(
            base_url,
            "/update_weights_from_tensor_atomic",
            common_payload,
        )
        transition_ended = update["ended"]
        phase_timings = update["response"].get("phase_timings_ms") or {}
        pause_window_seconds = phase_timings.get("pause_window_ms", 0.0) / 1000
    else:
        try:
            pause = timed_post(base_url, "/pause_generation", {"mode": "in_place"})
            update = timed_post(
                base_url,
                "/update_weights_from_tensor",
                common_payload,
            )
        finally:
            if pause is not None:
                cont = timed_post(
                    base_url,
                    "/continue_generation",
                    {"torch_empty_cache": False},
                )
        if update is None or cont is None:
            raise RuntimeError(f"incomplete update transition {label}")
        transition_ended = cont["ended"]
        pause_window_seconds = transition_ended - pause["started"]
        phase_timings = update["response"].get("phase_timings_ms") or {}

    traffic.end_transition(transition_ended)
    time.sleep(0.100)
    gap = traffic.finish_transition()
    return {
        "label": label,
        "update_mode": update_mode,
        "pause": pause,
        "update": update,
        "continue": cont,
        "pause_window_seconds": pause_window_seconds,
        "phase_timings_ms": phase_timings,
        "rank_phase_timings_ms": update["response"].get("rank_phase_timings_ms", []),
        "rank_host_memory_bytes": update["response"].get("rank_host_memory_bytes", []),
        "memory": sampler.window(
            transition_started, transition_ended, baseline_override=memory_before
        ),
        "traffic_gap": {
            "raw_gap_ms": distribution(gap["raw_gap_ms"]),
            "adjusted_itl_ms": distribution(gap["adjusted_itl_ms"]),
        },
    }


def timed_post_allow_error(
    base_url: str, path: str, payload: dict, timeout: float = 300
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    ended = time.perf_counter()
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    return {
        "started": started,
        "ended": ended,
        "seconds": ended - started,
        "status_code": response.status_code,
        "response": body,
    }


def run_fail_closed_fault_injection(
    *,
    args: argparse.Namespace,
    baseline: ChecksumSnapshot,
    original_items: list[tuple[str, Any]],
    updated_items: list[tuple[str, Any]],
) -> dict[str, Any]:
    before = get_server_info(args.base_url)
    updated_payloads, updated_prep, updated_keepalive = serialize_payloads(
        updated_items, args.tp_size, args.update_mode
    )
    failed = timed_post_allow_error(
        args.base_url,
        "/update_weights_from_tensor_atomic",
        {
            "serialized_named_tensors": updated_payloads,
            "load_format": None,
            "flush_cache": False,
            "draft_only": True,
            "torch_empty_cache": False,
            "weight_version": f"{args.weight_version_prefix}-must-not-commit",
            "tensors_are_pre_sharded": True,
            "collect_phase_timings": True,
            # The checkpoint leads with confidence_head.* tensors, which the
            # server consumes without applying when the confidence head is
            # disabled (static ragged-verify mode), so a fault after 2 tensors
            # fires before any real draft mutation and the post-failure
            # "draft changed" assertion below can never hold. 8 lands the
            # fault safely inside decoder-parameter mutations.
            "fault_injection_after_tensors": 8,
        },
    )
    del updated_payloads, updated_keepalive
    if failed["status_code"] < 400:
        raise RuntimeError("fault-injected update unexpectedly succeeded")
    failed_body = failed["response"]
    if not failed_body.get("partial_update") or not failed_body.get("unhealthy"):
        raise RuntimeError(f"fault response was not partial+unhealthy: {failed_body}")

    after_failure = get_server_info(args.base_url)
    health = after_failure.get("draft_weight_update") or {}
    if not health.get("unhealthy") or not health.get("paused"):
        raise RuntimeError(f"instance did not stay fail-closed: {health}")
    if after_failure.get("weight_version") != before.get("weight_version"):
        raise RuntimeError("weight_version advanced after a failed commit")

    continue_attempt = timed_post_allow_error(
        args.base_url,
        "/continue_generation",
        {"torch_empty_cache": False},
    )
    if continue_attempt["status_code"] != 409:
        raise RuntimeError(
            f"continue was not rejected while unhealthy: {continue_attempt}"
        )

    failed_snapshot = checksum_snapshot(args.base_url)
    failed_comparison = compare_checksums(baseline, failed_snapshot)
    if any(rank["target_changed_count"] for rank in failed_comparison.get("ranks", [])):
        raise RuntimeError("target changed during fault-injected draft commit")
    if not any(
        rank["draft_changed_count"] for rank in failed_comparison.get("ranks", [])
    ):
        raise RuntimeError("fault injection did not occur after a draft mutation")

    original_payloads, original_prep, original_keepalive = serialize_payloads(
        original_items, args.tp_size, args.update_mode
    )
    recovered = timed_post(
        args.base_url,
        "/update_weights_from_tensor_atomic",
        {
            "serialized_named_tensors": original_payloads,
            "load_format": None,
            "flush_cache": False,
            "draft_only": True,
            "torch_empty_cache": False,
            "weight_version": f"{args.weight_version_prefix}-fault-recovered",
            "tensors_are_pre_sharded": True,
            "collect_phase_timings": True,
            "recovery": True,
        },
    )
    del original_payloads, original_keepalive
    after_recovery = get_server_info(args.base_url)
    recovery_health = after_recovery.get("draft_weight_update") or {}
    if recovery_health.get("unhealthy") or recovery_health.get("paused"):
        raise RuntimeError(
            f"explicit recovery did not restore health: {recovery_health}"
        )
    restored_snapshot = checksum_snapshot(args.base_url)
    restored_comparison = compare_checksums(baseline, restored_snapshot)
    for rank in restored_comparison.get("ranks", []):
        if rank["target_changed_count"] or rank["draft_changed_count"]:
            raise RuntimeError(
                f"fault recovery checksum mismatch on rank {rank['rank']}"
            )
    if not restored_comparison.get("engine_matches", False):
        raise RuntimeError("combined checksum did not restore after fault")

    return {
        "status": "passed",
        "before": before,
        "failed_update": failed,
        "after_failure": after_failure,
        "continue_attempt": continue_attempt,
        "failed_checksum": {
            "snapshot": failed_snapshot.summary(),
            "vs_baseline": failed_comparison,
        },
        "updated_payload_preparation": updated_prep,
        "recovery": recovered,
        "after_recovery": after_recovery,
        "recovered_checksum": {
            "snapshot": restored_snapshot.summary(),
            "vs_baseline": restored_comparison,
        },
        "original_payload_preparation": original_prep,
    }


def memory_leak_analysis(
    marks: list[dict[str, Any]], gpu_indices: list[int]
) -> dict[str, Any]:
    result = {}
    for index in gpu_indices:
        points = [
            (mark["elapsed_seconds"], mark["memory_bytes"][str(index)])
            for mark in marks
            if mark["state"] == "restored"
        ]
        # Exclude the first restored point: the CUDA allocator is allowed one
        # warm-up reservation for the largest streamed tensor.
        fit_points = points[1:]
        slope = None
        if len(fit_points) >= 2:
            xs = [point[0] for point in fit_points]
            ys = [point[1] for point in fit_points]
            x_mean = statistics.fmean(xs)
            y_mean = statistics.fmean(ys)
            denominator = sum((x - x_mean) ** 2 for x in xs)
            if denominator:
                slope = (
                    sum((x - x_mean) * (y - y_mean) for x, y in fit_points)
                    / denominator
                )
        values = [point[1] for point in points]
        result[str(index)] = {
            "restored_points": len(points),
            "first_restored_bytes": values[0] if values else None,
            "last_restored_bytes": values[-1] if values else None,
            "growth_bytes": values[-1] - values[0] if values else None,
            "post_warmup_slope_bytes_per_second": slope,
            "post_warmup_slope_mib_per_hour": (
                slope * 3600 / MIB if slope is not None else None
            ),
            "restored_memory_bytes": distribution([float(value) for value in values]),
        }
    return result


def host_memory_leak_analysis(
    cycles: list[dict[str, Any]], tp_size: int
) -> dict[str, Any]:
    result = {}
    for rank in range(tp_size):
        points = []
        for cycle in cycles:
            transition = cycle["restore_transition"]
            rank_payload = next(
                (
                    payload
                    for payload in transition["rank_host_memory_bytes"]
                    if payload["rank"] == rank
                ),
                None,
            )
            if rank_payload is None:
                continue
            rss = rank_payload.get("scheduler_rss_after_release_bytes")
            if rss is not None:
                points.append((transition["update"]["ended"], float(rss)))
        fit_points = points[1:]
        slope = None
        if len(fit_points) >= 2:
            xs = [point[0] for point in fit_points]
            ys = [point[1] for point in fit_points]
            x_mean = statistics.fmean(xs)
            y_mean = statistics.fmean(ys)
            denominator = sum((x - x_mean) ** 2 for x in xs)
            if denominator:
                slope = (
                    sum((x - x_mean) * (y - y_mean) for x, y in fit_points)
                    / denominator
                )
        values = [point[1] for point in points]
        result[str(rank)] = {
            "restored_points": len(points),
            "first_restored_rss_bytes": values[0] if values else None,
            "last_restored_rss_bytes": values[-1] if values else None,
            "growth_bytes": values[-1] - values[0] if values else None,
            "post_warmup_slope_bytes_per_second": slope,
            "post_warmup_slope_mib_per_hour": (
                slope * 3600 / MIB if slope is not None else None
            ),
            "restored_rss_bytes": distribution(values),
        }
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def source_git_head() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main(args: argparse.Namespace) -> int:
    started_wall = time.time()
    started_perf = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "configuration": vars(args)
        | {
            "original_checkpoint": str(args.original_checkpoint),
            "updated_checkpoint": str(args.updated_checkpoint),
            "artifact": str(args.artifact),
            "server_log": str(args.server_log) if args.server_log else None,
        },
        "started_unix_seconds": started_wall,
        "execution": {
            "argv": shlex.join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_sha": source_git_head(),
        },
        "cycles": [],
        "invariant_failures": [],
    }

    original_tensors, original_items, original_info = load_checkpoint(
        args.original_checkpoint
    )
    updated_tensors, updated_items, updated_info = load_checkpoint(
        args.updated_checkpoint
    )
    # Keep both mappings alive until every consumer has returned.
    result["checkpoints"] = {"original": original_info, "updated": updated_info}

    warmup = post(
        args.base_url,
        "/generate",
        {
            "text": "Name three colors.",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 32},
        },
    )
    result["warmup_output_tokens"] = warmup.get("meta_info", {}).get(
        "completion_tokens"
    )
    baseline = checksum_snapshot(args.base_url)
    result["baseline_checksum"] = baseline.summary()
    result["server_info_before"] = get_server_info(args.base_url)
    result["graph_log_before"] = scan_graph_log(args.server_log)

    sampler = NvmlSampler(args.gpu_indices, args.nvml_interval_seconds)
    traffic = TrafficRunner(
        base_url=args.base_url,
        concurrency=args.traffic_concurrency,
        output_tokens=args.output_tokens,
        timeout_seconds=args.request_timeout_seconds,
    )
    sampler.start()
    traffic.start()
    traffic.wait_active()
    traffic.set_phase("baseline-initial")
    traffic.wait_phase(
        "baseline-initial",
        min_samples=args.min_phase_samples,
        min_seconds=args.dwell_seconds,
        timeout=args.phase_timeout_seconds,
    )
    result["initial_request_metrics"] = traffic.phase_summary("baseline-initial")

    expected_updated: Optional[ChecksumSnapshot] = None
    memory_marks = []
    try:
        for cycle_index in range(args.repeats):
            cycle: dict[str, Any] = {"index": cycle_index}

            (
                updated_payloads,
                updated_serialization,
                updated_payload_keepalive,
            ) = serialize_payloads(updated_items, args.tp_size, args.update_mode)
            cycle["updated_payload_preparation"] = updated_serialization
            cycle["update_transition"] = transition_update(
                base_url=args.base_url,
                traffic=traffic,
                sampler=sampler,
                payloads=updated_payloads,
                weight_version=f"{args.weight_version_prefix}-updated-{cycle_index}",
                label=f"update-{cycle_index}",
                update_mode=args.update_mode,
            )
            del updated_payloads, updated_payload_keepalive

            updated_phase = f"updated-{cycle_index}"
            traffic.set_phase(updated_phase)
            traffic.wait_phase(
                updated_phase,
                min_samples=args.min_phase_samples,
                min_seconds=args.dwell_seconds,
                timeout=args.phase_timeout_seconds,
            )
            cycle["updated_request_metrics"] = traffic.phase_summary(updated_phase)
            traffic.set_phase(f"validate-updated-{cycle_index}")
            updated_snapshot = checksum_snapshot(args.base_url)
            baseline_to_updated = compare_checksums(baseline, updated_snapshot)
            cycle["updated_checksum"] = {
                "snapshot": updated_snapshot.summary(),
                "vs_baseline": baseline_to_updated,
            }
            if expected_updated is None:
                expected_updated = updated_snapshot
            else:
                updated_comparison = compare_checksums(
                    expected_updated, updated_snapshot
                )
                cycle["updated_checksum"]["vs_first_updated"] = updated_comparison
                if not updated_comparison.get("rank_count_matches", False):
                    result["invariant_failures"].append(
                        f"cycle {cycle_index}: updated rank count changed"
                    )
                for rank in updated_comparison.get("ranks", []):
                    if (
                        rank["target_changed_count"] != 0
                        or rank["draft_changed_count"] != 0
                    ):
                        result["invariant_failures"].append(
                            f"cycle {cycle_index}: updated checksum differs from "
                            f"cycle 0 on rank {rank['rank']}"
                        )
                if not updated_comparison.get("engine_matches", False):
                    result["invariant_failures"].append(
                        f"cycle {cycle_index}: updated engine checksum differs "
                        "from cycle 0"
                    )

            for rank in baseline_to_updated.get("ranks", []):
                if rank["target_changed_count"] != 0:
                    result["invariant_failures"].append(
                        f"cycle {cycle_index}: target changed on rank {rank['rank']}"
                    )
                if rank["draft_changed_count"] == 0:
                    result["invariant_failures"].append(
                        f"cycle {cycle_index}: draft did not change on rank {rank['rank']}"
                    )

            updated_memory = sampler.snapshot()
            memory_marks.append(
                {
                    "cycle": cycle_index,
                    "state": "updated",
                    "elapsed_seconds": time.perf_counter() - started_perf,
                    "memory_bytes": {
                        str(index): value for index, value in updated_memory.items()
                    },
                }
            )

            (
                original_payloads,
                original_serialization,
                original_payload_keepalive,
            ) = serialize_payloads(original_items, args.tp_size, args.update_mode)
            cycle["original_payload_preparation"] = original_serialization
            cycle["restore_transition"] = transition_update(
                base_url=args.base_url,
                traffic=traffic,
                sampler=sampler,
                payloads=original_payloads,
                weight_version=f"{args.weight_version_prefix}-restored-{cycle_index}",
                label=f"restore-{cycle_index}",
                update_mode=args.update_mode,
            )
            del original_payloads, original_payload_keepalive

            restored_phase = f"restored-{cycle_index}"
            traffic.set_phase(restored_phase)
            traffic.wait_phase(
                restored_phase,
                min_samples=args.min_phase_samples,
                min_seconds=args.dwell_seconds,
                timeout=args.phase_timeout_seconds,
            )
            cycle["restored_request_metrics"] = traffic.phase_summary(restored_phase)
            traffic.set_phase(f"validate-restored-{cycle_index}")
            restored_snapshot = checksum_snapshot(args.base_url)
            restored_comparison = compare_checksums(baseline, restored_snapshot)
            cycle["restored_checksum"] = {
                "snapshot": restored_snapshot.summary(),
                "vs_baseline": restored_comparison,
            }
            for rank in restored_comparison.get("ranks", []):
                if (
                    rank["target_changed_count"] != 0
                    or rank["draft_changed_count"] != 0
                ):
                    result["invariant_failures"].append(
                        f"cycle {cycle_index}: restore mismatch on rank {rank['rank']}"
                    )
            if not restored_comparison.get("engine_matches", False):
                result["invariant_failures"].append(
                    f"cycle {cycle_index}: combined engine checksum did not restore"
                )

            restored_memory = sampler.snapshot()
            memory_marks.append(
                {
                    "cycle": cycle_index,
                    "state": "restored",
                    "elapsed_seconds": time.perf_counter() - started_perf,
                    "memory_bytes": {
                        str(index): value for index, value in restored_memory.items()
                    },
                }
            )
            result["cycles"].append(cycle)
            result["memory_marks"] = memory_marks
            atomic_write_json(args.artifact, result)
    finally:
        traffic.set_phase("shutdown")
        traffic.stop()
        sampler.stop()

    if args.fault_injection:
        result["fault_injection"] = run_fail_closed_fault_injection(
            args=args,
            baseline=baseline,
            original_items=original_items,
            updated_items=updated_items,
        )
        atomic_write_json(args.artifact, result)
    # Preserve references until after every update/recovery IPC consumer exits.
    del original_items, original_tensors, updated_items, updated_tensors

    result["request_metrics"] = {
        "baseline": traffic.aggregate_summary("baseline"),
        "updated": traffic.aggregate_summary("updated"),
        "restored": traffic.aggregate_summary("restored"),
        "cross_transition_request_count": sum(
            not record["pure_phase"] for record in traffic.records
        ),
        "total_request_count": len(traffic.records),
        "errors": traffic.errors,
    }
    result["transition_distributions"] = {
        "update_ms": distribution(
            [
                cycle["update_transition"]["update"]["seconds"] * 1000
                for cycle in result["cycles"]
            ]
        ),
        "restore_ms": distribution(
            [
                cycle["restore_transition"]["update"]["seconds"] * 1000
                for cycle in result["cycles"]
            ]
        ),
        "update_pause_window_ms": distribution(
            [
                cycle["update_transition"]["pause_window_seconds"] * 1000
                for cycle in result["cycles"]
            ]
        ),
        "restore_pause_window_ms": distribution(
            [
                cycle["restore_transition"]["pause_window_seconds"] * 1000
                for cycle in result["cycles"]
            ]
        ),
        "update_payload_preparation_ms": distribution(
            [
                cycle["updated_payload_preparation"]["total_seconds"] * 1000
                for cycle in result["cycles"]
            ]
        ),
        "restore_payload_preparation_ms": distribution(
            [
                cycle["original_payload_preparation"]["total_seconds"] * 1000
                for cycle in result["cycles"]
            ]
        ),
    }
    result["phase_timing_distributions_ms"] = {}
    result["host_memory_distributions_bytes"] = {}
    for state, transition_name in (
        ("update", "update_transition"),
        ("restore", "restore_transition"),
    ):
        phase_keys = {
            key
            for cycle in result["cycles"]
            for key in cycle[transition_name]["phase_timings_ms"]
        }
        result["phase_timing_distributions_ms"][state] = {
            key: distribution(
                [
                    cycle[transition_name]["phase_timings_ms"][key]
                    for cycle in result["cycles"]
                    if key in cycle[transition_name]["phase_timings_ms"]
                ]
            )
            for key in sorted(phase_keys)
        }
        rank_host = {}
        for cycle in result["cycles"]:
            for rank_payload in cycle[transition_name]["rank_host_memory_bytes"]:
                rank = str(rank_payload["rank"])
                for key, value in rank_payload.items():
                    if key != "rank":
                        rank_host.setdefault(rank, {}).setdefault(key, []).append(
                            float(value)
                        )
        result["host_memory_distributions_bytes"][state] = {
            rank: {key: distribution(values) for key, values in sorted(fields.items())}
            for rank, fields in sorted(rank_host.items())
        }

    for index in args.gpu_indices:
        for state, transition_name in (
            ("update", "update_transition"),
            ("restore", "restore_transition"),
        ):
            peak_deltas = [
                cycle[transition_name]["memory"][str(index)]["peak_delta_bytes"]
                for cycle in result["cycles"]
            ]
            result["transition_distributions"][
                f"gpu_{index}_{state}_peak_delta_mib"
            ] = nullable_distribution(
                [value / MIB if value is not None else None for value in peak_deltas]
            )

    result["memory_leak_analysis"] = memory_leak_analysis(
        memory_marks, args.gpu_indices
    )
    result["host_memory_leak_analysis"] = host_memory_leak_analysis(
        result["cycles"], args.tp_size
    )
    result["server_info_after"] = get_server_info(args.base_url)
    result["graph_log_after"] = scan_graph_log(args.server_log)
    if result["graph_log_before"] and result["graph_log_after"]:
        if (
            result["graph_log_before"]["capture_begin_count"]
            != result["graph_log_after"]["capture_begin_count"]
        ):
            result["invariant_failures"].append(
                "CUDA Graph capture count changed during updates"
            )
        if result["graph_log_after"]["scheduler_exception_count"]:
            result["invariant_failures"].append(
                "scheduler exception found in server log"
            )
        if (
            result["graph_log_after"]["decode_cuda_graph_true_count"]
            <= result["graph_log_before"]["decode_cuda_graph_true_count"]
        ):
            result["invariant_failures"].append(
                "no live decode using CUDA Graph was logged during updates"
            )
    if traffic.errors:
        result["invariant_failures"].append(
            f"{len(traffic.errors)} background traffic requests failed"
        )

    result["duration_seconds"] = time.perf_counter() - started_perf
    result["finished_unix_seconds"] = time.time()
    result["status"] = "passed" if not result["invariant_failures"] else "failed"
    atomic_write_json(args.artifact, result)
    print(json.dumps(result["transition_distributions"], indent=2, sort_keys=True))
    print(json.dumps(result["request_metrics"], indent=2, sort_keys=True))
    print(json.dumps(result["memory_leak_analysis"], indent=2, sort_keys=True))
    print(f"artifact={args.artifact} status={result['status']}")
    return 0 if result["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument(
        "--update-mode",
        choices=("legacy", "atomic-presharded"),
        default="legacy",
    )
    parser.add_argument("--fault-injection", action="store_true")
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--updated-checkpoint", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--traffic-concurrency", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--min-phase-samples", type=int, default=24)
    parser.add_argument("--dwell-seconds", type=float, default=0.0)
    parser.add_argument("--phase-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--nvml-interval-seconds", type=float, default=0.002)
    parser.add_argument("--weight-version-prefix", default="draft-only-bench")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    if args.tp_size != len(args.gpu_indices):
        parser.error("--tp-size must match the number of --gpu-indices")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.fault_injection and args.update_mode != "atomic-presharded":
        parser.error("--fault-injection requires --update-mode atomic-presharded")
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
