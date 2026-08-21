"""Performance-degradation characterization matrix for hidden-state capture.

Runs capture OFF vs ON (Mooncake sink) on Qwen3-8B DSpark. OFF is stock
sglang (no capture flag); ON adds --enable-hidden-state-capture with the
mooncake sink. The delta between the two IS the cost of capture.

THE STANDARD MATRIX v2 (project convention — use these, don't improvise):

  BASE workload: random 4096-in / 512-out (realistic chat/RAG shape; the
  longer decode leg exercises the verify-capture path in proportion).
  All serving features ON: chunked prefill 8192 + radix cache + prefill &
  decode CUDA graphs.

  --cells regression   per-change gate, 12 runs, one GPU each (shardable
                       across GPUs via --only + CUDA_VISIBLE_DEVICES):
    rand_low   x2  random cold, rate=2  (light-load anchor, noise floor)
    rand_high      random cold, rate=8  (~125% of capacity: sustained
                   queueing + admission pressure; verify-ring stress)
    warm_low       shared-prefix (~50% radix hit), rate=2
    warm_high      shared-prefix, rate=8 (hit + overload interaction,
                   closest to production)
    bare_high      random cold, rate=8, chunked/radix/graphs ALL OFF
                   (isolates capture-cost dependence on the feature stack)

  --cells all          adds supplement cells (saturation, longer inputs,
                       dp2) for milestone/merge characterization.

Concurrency semantics: fixed-rate cells are open-loop Poisson — in-flight
count is emergent (rate x e2e), NOT pinned. Use cell["max_concurrency"]
(closed-loop semaphore in bench_serving) when a pinned concurrency is
needed; regression cells use open-loop rates because production load is
open-loop.

v1 rows (1024/128 BASE) are superseded: not comparable to v2 rows.

Measurement hygiene:
- fixed seed and prompt shapes; 16-request warmup per run;
- every leg uses a fresh server process, store namespace, and writer epoch;
  warm-prefix objects are never deleted under a live producer index;
- exact pre/post counters come from the read-only internal-state endpoint;
  export coverage is the ``export_ok_ct`` delta after queue/ring/lane
  quiescence, so counting is non-destructive;
- GPU memory is sampled DURING the run by a background thread (1 Hz,
  nvidia-smi, restricted to CUDA_VISIBLE_DEVICES when set); reported as
  max-over-time of per-GPU max and of the sum;
- capture-side counters (export_ok / *_miss_ct) are parsed from the server's
  periodic stats log lines when present (logged every 30 seconds), giving
  direct miss attribution rather than coverage-only inference;
- the FULL bench_serving result dict is preserved per row (all latency
  percentiles, ITL, concurrency, accept_length, cache stats, server_info).

Rows append to a JSONL for auditability.

Sharding: cells are independent; run several shards in parallel, one GPU
each, by splitting with --only and giving each shard its own --shard index
(offsets the server and mooncake-master ports):

    CUDA_VISIBLE_DEVICES=0 ... --only rand_low --shard 0 &
    CUDA_VISIBLE_DEVICES=1 ... --only rand_high --shard 1 &

Usage:
    PYTHONPATH=python SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/bench_hidden_capture_matrix.py \
        --out /tmp/capture_matrix.jsonl [--cells regression|supplement|all]
"""

import argparse
import asyncio
import copy
import glob
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid

import psutil
import requests

from sglang.benchmark.serving import run_benchmark
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    get_benchmark_args,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
BASE_MASTER_PORT = 50060
BASE_SERVER_PORT = 21500
STORE_ID = "matrix_capture"


def _default_mooncake_master():
    try:
        import mooncake

        return os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master")
    except Exception:
        return "mooncake_master"


MASTER_BIN = os.environ.get("MOONCAKE_MASTER_BIN") or _default_mooncake_master()
SEED = 42
OUTER_WARMUP_REQUESTS = 16
DRAIN_MAX_S = 300.0

# Set per-shard in main(); defaults for shard 0.
MASTER_PORT = BASE_MASTER_PORT
SERVER_URL = f"http://127.0.0.1:{BASE_SERVER_PORT}"

_STATS_RE = re.compile(r"hidden capture stats: (\{.*\})")
ARTIFACT_DIR = None


def _server_args(cell):
    args = [
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--speculative-draft-attention-backend",
        "fa3",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "1",
        "--random-seed",
        str(SEED),
    ]
    features = cell["features"]
    chunked = cell.get("chunked", features)
    radix = cell.get("radix", features)
    graphs = cell.get("graphs", features)
    if chunked:
        args += ["--chunked-prefill-size", "8192"]
    else:
        args += ["--chunked-prefill-size", "-1"]
    if not radix:
        args += ["--disable-radix-cache"]
    if graphs:
        args += ["--cuda-graph-max-bs-decode", "64"]
    else:
        args += ["--cuda-graph-backend-prefill", "disabled"]
        args += ["--disable-decode-cuda-graph"]
    if cell["capture"]:
        args += ["--enable-hidden-state-capture"]
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity():
    source_root = os.environ.get("SGLANG_CHARACTERIZATION_SOURCE_ROOT")
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        cwd=source_root or os.path.dirname(__file__),
    ).strip()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=repo_root
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=repo_root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        text=True,
        cwd=repo_root,
    ).splitlines()
    digest = hashlib.sha256(diff)
    for relative in sorted(untracked):
        path = os.path.join(repo_root, relative)
        digest.update(relative.encode())
        if os.path.isfile(path):
            with open(path, "rb") as source:
                digest.update(source.read())
    return {
        "git_revision": revision,
        "worktree_diff_sha256": digest.hexdigest(),
        "worktree_dirty": bool(diff or untracked),
        "untracked_files": sorted(untracked),
    }


def _gpu_process_residue():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return [line for line in output.splitlines() if line.strip()]
    except Exception as error:
        return [f"readback-error: {error}"]


def _gpu_process_pids(readback):
    pids = set()
    for line in readback:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[1].isdigit():
            pids.add(int(fields[1]))
    return pids


def _wait_for_gpu_process_settle(baseline, timeout_s=30.0):
    """Wait for only processes introduced by this leg to disappear.

    Other agents can legitimately own other GPUs on this machine. Comparing
    PIDs against a pre-leg inventory avoids classifying those processes as
    residue while retaining the complete raw readback at every poll.
    """
    baseline_pids = _gpu_process_pids(baseline)
    start = time.monotonic()
    history = []
    quiet = 0
    while True:
        current = _gpu_process_residue()
        introduced = sorted(_gpu_process_pids(current) - baseline_pids)
        history.append(
            {
                "elapsed_s": time.monotonic() - start,
                "introduced_pids": introduced,
                "readback": current,
            }
        )
        quiet = quiet + 1 if not introduced else 0
        if quiet >= 2:
            return current, history, time.monotonic() - start, True
        if time.monotonic() - start >= timeout_s:
            return current, history, time.monotonic() - start, False
        time.sleep(1.0)


def _attempt_log_lineage(attempt_name, repeat_idx):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", attempt_name)
    result = {}
    for suffix, label in ((".log", "stdout"), (".err", "stderr")):
        matches = glob.glob(
            os.path.join(ARTIFACT_DIR, f"{safe_name}_r{repeat_idx}_*{suffix}")
        )
        if matches:
            path = max(matches, key=os.path.getmtime)
            result[label] = {"path": path, "sha256": _sha256(path)}
    return result


def _capture_snapshot():
    response = requests.get(SERVER_URL + "/server_info", timeout=30)
    response.raise_for_status()
    states = response.json().get("internal_states") or []
    snapshots = [state.get("hidden_capture") for state in states]
    return [snapshot for snapshot in snapshots if snapshot is not None]


def _sum_capture_stats(snapshots):
    total = {}
    for snapshot in snapshots:
        for name, value in snapshot.get("stats", {}).items():
            total[name] = total.get(name, 0) + int(value)
    return total


def _counter_delta(after, before):
    return {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in sorted(set(before) | set(after))
    }


def _validated_export_coverage(
    *,
    exported,
    completed,
    expected_requests,
    measured_warmup_requests,
    outer_warmup_drained,
    measured_drained,
):
    """Bind the export counter delta to exactly the measured request set.

    The counter baseline is taken only after the explicit outer warmup has
    drained.  ``run_benchmark`` must therefore perform no additional warmup
    inside the measured invocation: such an export is absent from
    ``bench.completed`` and makes coverage method-invalid (for example the
    historical 161 / 160 result).
    """
    if measured_warmup_requests != 0:
        raise AssertionError(
            "measured run must set warmup_requests=0 so export coverage and "
            "bench.completed describe the same request set"
        )
    if not outer_warmup_drained:
        raise AssertionError("outer warmup did not drain before counter baseline")
    if not measured_drained:
        raise AssertionError("measured capture pipeline did not drain")
    if completed != expected_requests:
        raise AssertionError(
            f"measured request set incomplete: completed={completed}, "
            f"expected={expected_requests}"
        )
    if completed <= 0:
        raise AssertionError(f"coverage denominator must be positive: {completed}")
    if exported < 0 or exported > completed:
        raise AssertionError(
            f"export coverage counter is outside the measured request set: "
            f"exported={exported}, completed={completed}"
        )
    coverage = exported / completed
    if not 0.0 <= coverage <= 1.0:
        raise AssertionError(f"invalid export coverage: {coverage}")
    return coverage


def _capture_quiescent(snapshots):
    for snapshot in snapshots:
        state = snapshot["state"]
        sink = state.get("sink") or {}
        if any(
            int(state.get(name, 0))
            for name in (
                "prefill_ring_inflight",
                "verify_ring_inflight",
                "finalize_worker_active",
                "export_queue_pending",
                "export_worker_active",
            )
        ):
            return False
        if int(sink.get("ready_tasks", 0)):
            return False
        lane_states = sink.get("lane_states") or {}
        if any(
            state_name != "FREE" and count for state_name, count in lane_states.items()
        ):
            return False
    return True


def _drain_capture(timeout_s=DRAIN_MAX_S):
    started = time.monotonic()
    deadline = started + timeout_s
    history = []
    quiet = 0
    last_exported = None
    while time.monotonic() < deadline:
        snapshots = _capture_snapshot()
        stats = _sum_capture_stats(snapshots)
        exported = stats.get("export_ok_ct", 0)
        quiescent = _capture_quiescent(snapshots)
        history.append(
            {
                "elapsed_s": time.monotonic() - started,
                "export_ok_ct": exported,
                "quiescent": quiescent,
                "snapshots": snapshots,
            }
        )
        quiet = quiet + 1 if quiescent and exported == last_exported else 0
        if quiet >= 2:
            return snapshots, history, time.monotonic() - started, True
        last_exported = exported
        time.sleep(1.0)
    snapshots = _capture_snapshot()
    return snapshots, history, time.monotonic() - started, False


class ProcessResourceSampler:
    """Aggregate server process-tree CPU/RSS/IO and host NIC byte deltas."""

    def __init__(self, pid):
        self.pid = pid
        self._started_s = None
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        self.net_start = psutil.net_io_counters()._asdict()
        self.net_end = None

    def _run(self):
        previous = None
        while not self._stop.is_set():
            try:
                root = psutil.Process(self.pid)
                processes = [root, *root.children(recursive=True)]
                rss = 0
                cpu = 0.0
                read_bytes = 0
                write_bytes = 0
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                        cpu += sum(process.cpu_times()[:2])
                        io = process.io_counters()
                        read_bytes += io.read_bytes
                        write_bytes += io.write_bytes
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                now = time.monotonic()
                cpu_pct = None
                if previous is not None and now > previous[0]:
                    cpu_pct = 100.0 * (cpu - previous[1]) / (now - previous[0])
                previous = (now, cpu)
                self.samples.append(
                    {
                        "elapsed_s": now,
                        "rss_bytes": rss,
                        "cpu_percent_one_core_units": cpu_pct,
                        "read_bytes": read_bytes,
                        "write_bytes": write_bytes,
                        "processes": len(processes),
                    }
                )
            except psutil.NoSuchProcess:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._started_s = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self.net_end = psutil.net_io_counters()._asdict()

    def summary(self):
        cpu = [sample["cpu_percent_one_core_units"] for sample in self.samples]
        cpu = [value for value in cpu if value is not None]
        return {
            "rss_peak_bytes": max(
                (sample["rss_bytes"] for sample in self.samples), default=0
            ),
            "rss_first_bytes": self.samples[0]["rss_bytes"] if self.samples else 0,
            "rss_last_bytes": self.samples[-1]["rss_bytes"] if self.samples else 0,
            "cpu_mean_one_core_units": sum(cpu) / len(cpu) if cpu else None,
            "process_count_peak": max(
                (sample["processes"] for sample in self.samples), default=0
            ),
            "process_read_bytes_delta": (
                self.samples[-1]["read_bytes"] - self.samples[0]["read_bytes"]
                if len(self.samples) > 1
                else 0
            ),
            "process_write_bytes_delta": (
                self.samples[-1]["write_bytes"] - self.samples[0]["write_bytes"]
                if len(self.samples) > 1
                else 0
            ),
            "host_net_bytes_sent_delta": self.net_end["bytes_sent"]
            - self.net_start["bytes_sent"],
            "host_net_bytes_recv_delta": self.net_end["bytes_recv"]
            - self.net_start["bytes_recv"],
            "samples": [
                {**sample, "elapsed_s": sample["elapsed_s"] - self._started_s}
                for sample in self.samples
            ],
        }


class GpuMemSampler:
    """1 Hz during-run nvidia-smi sampler: max-over-time of per-GPU max and
    of the all-GPU sum (labels what each number means, unlike a single
    post-run snapshot). Restricted to CUDA_VISIBLE_DEVICES when set, so
    parallel shards on other GPUs don't pollute the numbers."""

    def __init__(self):
        self.max_single_mb = 0
        self.max_sum_mb = 0
        self.samples = []
        self._started_s = None
        self._stop = threading.Event()
        self._thread = None
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._id_args = ["--id=" + visible] if visible else []

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        *self._id_args,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                vals = [int(x) for x in out.split()]
                self.max_single_mb = max(self.max_single_mb, max(vals))
                self.max_sum_mb = max(self.max_sum_mb, sum(vals))
                self.samples.append(
                    {
                        "elapsed_s": time.monotonic() - self._started_s,
                        "per_gpu_mb": vals,
                        "sum_mb": sum(vals),
                    }
                )
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._started_s = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


class CaptureStateSampler:
    """Poll the read-only capture snapshot while measured traffic runs."""

    def __init__(self, enabled, interval_s=2.0):
        self.enabled = enabled
        self.interval_s = interval_s
        self.samples = []
        self._started_s = None
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                snapshots = _capture_snapshot()
                self.samples.append(
                    {
                        "elapsed_s": time.monotonic() - self._started_s,
                        "snapshots": snapshots,
                    }
                )
            except Exception as error:
                self.samples.append(
                    {
                        "elapsed_s": time.monotonic() - self._started_s,
                        "error": str(error),
                    }
                )
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._started_s = time.monotonic()
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _parse_capture_counters(log_path):
    """Last 'hidden capture stats: {...}' line per replica process, summed."""
    try:
        text = open(log_path, errors="replace").read()
    except OSError:
        return None
    matches = _STATS_RE.findall(text)
    if not matches:
        return None
    # The log interleaves replicas; take the final snapshot per distinct
    # counter-set is impractical without pids -- keep the LAST line (single
    # replica) and note this is a lower bound under DP.
    try:
        return json.loads(matches[-1].replace("'", '"'))
    except json.JSONDecodeError:
        return None


def run_cell(cell, repeat_idx, attempt_name):
    gpu_process_baseline = _gpu_process_residue()
    run_store_id = "{}_s{}_r{}_{}".format(
        STORE_ID,
        MASTER_PORT - BASE_MASTER_PORT,
        repeat_idx,
        uuid.uuid4().hex[:10],
    )
    env = {
        **os.environ,
        "SGLANG_RAGGED_VERIFY_MODE": "compact",
    }
    if cell["capture"]:
        env.update(
            {
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": run_store_id,
                "MOONCAKE_MASTER": f"127.0.0.1:{MASTER_PORT}",
                "MOONCAKE_PROTOCOL": "tcp",
                # Size the bench store above one run's full sample volume;
                # otherwise the store's eviction watermark rejects puts
                # mid-run and coverage measures the BENCH store, not the
                # capture pipeline. v2 samples are ~4.5k tokens x ~57KB/token
                # -> ~260MB/sample x 300 requests needs headroom.
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "96gb",
                "SGLANG_HIDDEN_CAPTURE_PREFIX_LANES": str(cell.get("prefix_lanes", 2)),
                "SGLANG_HIDDEN_CAPTURE_PREFIX_MAX_SEGMENT_ROWS": str(
                    cell.get("max_segment_rows", 256)
                ),
            }
        )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", attempt_name)
    log_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".log",
        prefix=f"{safe_name}_r{repeat_idx}_",
        dir=ARTIFACT_DIR,
        delete=False,
    )
    err_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".err",
        prefix=f"{safe_name}_r{repeat_idx}_",
        dir=ARTIFACT_DIR,
        delete=False,
    )
    process = popen_launch_server(
        TARGET_MODEL,
        SERVER_URL,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_server_args(cell),
        env=env,
        return_stdout_stderr=(log_file, err_file),
    )
    row = None
    try:
        common = dict(
            base_url=SERVER_URL,
            tokenizer=TARGET_MODEL,
            num_prompts=cell["num_prompts"],
            request_rate=cell["rate"],
            max_concurrency=cell.get("max_concurrency"),
            seed=SEED,
        )
        if cell["prefix"] == "warm":
            # Shared-prefix: system 4096 + question 512 (in-shape ~4.6k, close
            # to the random cells' 4096) with 8 prompts/group -> the radix hit
            # rate lands near the 7/8 group-interior fraction under low load.
            args = get_benchmark_args(
                dataset_name="generated-shared-prefix",
                gsp_num_groups=max(2, cell["num_prompts"] // 8),
                gsp_prompts_per_group=8,
                gsp_system_prompt_len=4096,
                gsp_question_len=512,
                gsp_output_len=IN_OUT[1],
                **common,
            )
        else:
            args = get_benchmark_args(
                dataset_name="random",
                random_input_len=cell.get("input_len", IN_OUT[0]),
                random_output_len=cell.get("output_len", IN_OUT[1]),
                **common,
            )
        args.cache_report = True
        # This harness owns warmup placement.  The explicit outer warmup is
        # before the capture-counter baseline; run_benchmark's default inner
        # warmup (one request) would otherwise contaminate measured coverage.
        args.warmup_requests = 0

        warmup = copy.deepcopy(args)
        warmup.num_prompts = OUTER_WARMUP_REQUESTS
        warmup.max_concurrency = None
        warmup.cache_report = False
        warmup.warmup_requests = 0
        if cell["prefix"] == "warm":
            # GeneratedSharedPrefixDataset ignores ``num_prompts`` and emits
            # gsp_num_groups * gsp_prompts_per_group rows.  Make the explicit
            # outer warmup cardinality real rather than merely relabelling a
            # full measured set (which produced a misleading 99.98% hit rate).
            if OUTER_WARMUP_REQUESTS % warmup.gsp_prompts_per_group:
                raise AssertionError(
                    "outer warmup request target must be divisible by "
                    "gsp_prompts_per_group"
                )
            warmup.gsp_num_groups = (
                OUTER_WARMUP_REQUESTS // warmup.gsp_prompts_per_group
            )
        warmup_res = asyncio.run(asyncio.to_thread(run_benchmark, warmup))
        warmup_completed = int(warmup_res["completed"])
        if warmup_completed != OUTER_WARMUP_REQUESTS:
            raise AssertionError(
                f"outer warmup request set mismatch: completed={warmup_completed}, "
                f"expected={OUTER_WARMUP_REQUESTS}"
            )
        if cell["prefix"] == "cold":
            requests.post(SERVER_URL + "/flush_cache", timeout=30)
        if cell["capture"]:
            before_snapshots, warmup_drain, warmup_drain_s, warmup_drained = (
                _drain_capture()
            )
            before_stats = _sum_capture_stats(before_snapshots)
        else:
            before_snapshots = []
            before_stats = {}
            warmup_drain = []
            warmup_drain_s = 0.0
            warmup_drained = True

        with (
            GpuMemSampler() as mem,
            ProcessResourceSampler(process.pid) as resources,
            CaptureStateSampler(cell["capture"]) as capture_sampler,
        ):
            res = asyncio.run(asyncio.to_thread(run_benchmark, args))
            if cell["capture"]:
                (
                    after_snapshots,
                    measured_drain,
                    measured_drain_s,
                    measured_drained,
                ) = _drain_capture()
                after_stats = _sum_capture_stats(after_snapshots)
                counter_delta = _counter_delta(after_stats, before_stats)
            else:
                after_snapshots = []
                after_stats = {}
                measured_drain = []
                measured_drain_s = 0.0
                measured_drained = True
                counter_delta = {}

        completed = int(res["completed"])
        expected_requests = int(cell["num_prompts"])
        if completed != expected_requests:
            raise AssertionError(
                f"measured request set incomplete: completed={completed}, "
                f"expected={expected_requests}"
            )
        measured_coverage = None
        if cell["capture"]:
            measured_coverage = _validated_export_coverage(
                exported=int(counter_delta.get("export_ok_ct", 0)),
                completed=completed,
                expected_requests=expected_requests,
                measured_warmup_requests=int(args.warmup_requests),
                outer_warmup_drained=warmup_drained,
                measured_drained=measured_drained,
            )

        row = {
            "cell": {k: v for k, v in cell.items()},
            "repeat": repeat_idx,
            "server_args": " ".join(_server_args(cell)),
            "gpu_mem_max_single_mb": mem.max_single_mb,
            "gpu_mem_max_sum_mb": mem.max_sum_mb,
            "gpu_mem_samples": mem.samples,
            "process_resources": resources.summary(),
            "capture_state_samples": capture_sampler.samples,
            "store_id": run_store_id if cell["capture"] else None,
            "stdout_path": log_file.name,
            "stderr_path": err_file.name,
            "source": _source_identity(),
            # The FULL bench_serving result: every latency percentile
            # (mean/median/std/p90/p95/p99 of TTFT/TPOT/ITL/E2E), all
            # throughput variants, concurrency, accept_length, cache stats.
            "bench": {
                k: v
                for k, v in res.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            "cache_hit_rate_pct": (res.get("cache_report") or {}).get(
                "cache_hit_rate_pct"
            ),
            "measured_request_set": {
                "seed": int(args.seed),
                "requested": expected_requests,
                "completed": completed,
                "inner_warmup_requests": int(args.warmup_requests),
                "counter_window": (
                    "post_outer_warmup_drain_to_post_measured_drain"
                    if cell["capture"]
                    else None
                ),
            },
            "outer_warmup_request_set": {
                "seed": int(warmup.seed),
                "requested": OUTER_WARMUP_REQUESTS,
                "completed": warmup_completed,
                "inner_warmup_requests": int(warmup.warmup_requests),
                "gsp_num_groups": (
                    int(warmup.gsp_num_groups) if cell["prefix"] == "warm" else None
                ),
                "gsp_prompts_per_group": (
                    int(warmup.gsp_prompts_per_group)
                    if cell["prefix"] == "warm"
                    else None
                ),
            },
        }
        if cell["capture"]:
            exported = counter_delta.get("export_ok_ct", 0)
            row["counter_baseline"] = before_stats
            row["counter_final"] = after_stats
            row["measured_counter_delta"] = counter_delta
            row["capture_state_baseline"] = before_snapshots
            row["capture_state_final"] = after_snapshots
            row["warmup_drain_history"] = warmup_drain
            row["warmup_drain_s"] = warmup_drain_s
            row["warmup_drained"] = warmup_drained
            row["measured_drain_history"] = measured_drain
            row["measured_drain_s"] = measured_drain_s
            row["measured_drained"] = measured_drained
            row["exported_samples_at_drain"] = exported
            row["export_coverage_frac"] = measured_coverage
            # Keep the log parse as an independent lineage/debug readback.
            row["capture_counters_last_log"] = _parse_capture_counters(err_file.name)
        return row
    finally:
        terminate_and_kill_process_tree(process, terminate_timeout=60, wait_timeout=60)
        if isinstance(row, dict):
            row["server_exit_code_after_cleanup"] = process.poll()
            row["gpu_process_baseline"] = gpu_process_baseline
            row["gpu_process_residue_immediate"] = _gpu_process_residue()
            (
                final_gpu_readback,
                gpu_settle_history,
                gpu_settle_s,
                gpu_settled,
            ) = _wait_for_gpu_process_settle(gpu_process_baseline)
            row["gpu_process_residue_after_cleanup"] = final_gpu_readback
            row["gpu_process_introduced_after_cleanup"] = sorted(
                _gpu_process_pids(final_gpu_readback)
                - _gpu_process_pids(gpu_process_baseline)
            )
            row["gpu_process_settle_history"] = gpu_settle_history
            row["gpu_process_settle_s"] = gpu_settle_s
            row["gpu_process_settled"] = gpu_settled
        for f in (log_file, err_file):
            try:
                f.close()
            except OSError:
                pass
        if isinstance(row, dict):
            row["raw_log_sha256"] = {
                "stdout": _sha256(log_file.name),
                "stderr": _sha256(err_file.name),
            }


def _pair(cells, name, base, repeats=1, **overrides):
    for capture in (False, True):
        for r in range(repeats):
            cells.append(
                (
                    "{}|cap={}".format(name, int(capture)),
                    r,
                    {**base, **overrides, "capture": capture},
                )
            )


# v2 BASE workload: 4k in / 512 out, all serving features on, open-loop.
IN_OUT = (4096, 512)

BASE = dict(
    prefix="cold",
    features=True,
    rate=2.0,
    num_prompts=100,
    max_concurrency=None,
)

# Rate ladder: server capacity at 4k/512 is ~6-7 req/s (H200, DSpark 8B);
# low = comfortable in-flight handful, high = ~125% capacity (sustained
# queueing + admission pressure without the rate=inf artifact).
RATE_LOW = 2.0
RATE_HIGH = 8.0


def regression_cells():
    """The per-change gate (see module docstring). Cells are independent —
    shard across GPUs with --only + --shard."""
    cells = []
    _pair(cells, "rand_low", BASE, repeats=2)
    _pair(cells, "rand_high", BASE, rate=RATE_HIGH, num_prompts=240)
    _pair(cells, "warm_low", BASE, prefix="warm", num_prompts=96)
    _pair(cells, "warm_high", BASE, prefix="warm", rate=RATE_HIGH, num_prompts=240)
    _pair(cells, "bare_high", BASE, features=False, rate=RATE_HIGH, num_prompts=240)
    return cells


def supplement_cells():
    """Milestone extras on top of regression."""
    cells = []
    # Saturation: unbounded rate = offered load beyond capacity; the OFF/ON
    # delta here measures max-throughput cost, unlike fixed-rate cells.
    _pair(cells, "saturation", BASE, rate=float("inf"), num_prompts=200)
    # Pinned-concurrency variant (closed loop): the counterpart to the
    # open-loop rate cells when a fixed in-flight count is wanted.
    _pair(
        cells,
        "conc32",
        BASE,
        rate=float("inf"),
        max_concurrency=32,
        num_prompts=200,
    )
    _pair(cells, "input_long_16k", BASE, rate=1.0, num_prompts=50, input_len=16384)
    _pair(
        cells,
        "short_low",
        BASE,
        rate=RATE_LOW,
        num_prompts=160,
        input_len=512,
        output_len=128,
    )
    _pair(
        cells,
        "graph_off_high",
        BASE,
        rate=RATE_HIGH,
        num_prompts=240,
        graphs=False,
    )
    _pair(
        cells,
        "chunked_off_high",
        BASE,
        rate=RATE_HIGH,
        num_prompts=240,
        chunked=False,
    )
    # Bounded continuous traffic. At rate=4 the 480-request warm leg runs
    # for roughly two minutes, long enough to classify producer/consumer
    # backlog and memory slopes without turning this into an endurance test.
    _pair(
        cells,
        "soak_warm",
        BASE,
        prefix="warm",
        rate=4.0,
        num_prompts=480,
        stop_rule="480 requests (~120s); stop if final 60s slopes are stable",
    )
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cells",
        choices=["regression", "supplement", "all"],
        default="regression",
    )
    parser.add_argument(
        "--capture-order",
        choices=["off-on", "on-off"],
        default="off-on",
        help="order paired capture-off/on legs for crossed same-GPU repeats",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="directory retaining raw server stdout/stderr for every attempt",
    )
    parser.add_argument("--only", default=None, help="substring filter on cell name")
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="parallel-shard index: offsets server + mooncake-master ports "
        "so shards on different GPUs don't collide",
    )
    parser.add_argument("--prefix-lanes", type=int, default=None)
    parser.add_argument("--max-segment-rows", type=int, default=None)
    opts = parser.parse_args()

    global MASTER_PORT, SERVER_URL, ARTIFACT_DIR
    MASTER_PORT = BASE_MASTER_PORT + opts.shard
    SERVER_URL = f"http://127.0.0.1:{BASE_SERVER_PORT + opts.shard}"
    ARTIFACT_DIR = opts.artifact_dir or opts.out + ".artifacts"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    master_stdout = open(os.path.join(ARTIFACT_DIR, "mooncake_master.log"), "w")
    master_stderr = open(os.path.join(ARTIFACT_DIR, "mooncake_master.err"), "w")
    master = subprocess.Popen(
        [
            MASTER_BIN if os.path.exists(MASTER_BIN) else "mooncake_master",
            "--port",
            str(MASTER_PORT),
            # The master also binds a metrics HTTP server on a FIXED default
            # port (9003); without this, parallel shards' masters collide
            # there and every loser dies -> store setup -1 on that shard.
            "--metrics_port",
            str(9100 + opts.shard),
        ],
        stdout=master_stdout,
        stderr=master_stderr,
    )
    time.sleep(2)
    try:
        todo = []
        if opts.cells in ("regression", "all"):
            todo += regression_cells()
        if opts.cells in ("supplement", "all"):
            todo += supplement_cells()
        for _, _, cell in todo:
            if opts.prefix_lanes is not None:
                cell["prefix_lanes"] = opts.prefix_lanes
            if opts.max_segment_rows is not None:
                cell["max_segment_rows"] = opts.max_segment_rows
        if opts.only:
            todo = [t for t in todo if opts.only in t[0]]
        if opts.capture_order == "on-off":
            todo.sort(
                key=lambda item: (
                    item[0].split("|", 1)[0],
                    item[1],
                    0 if item[2]["capture"] else 1,
                )
            )

        for i, (name, repeat_idx, cell) in enumerate(todo):
            print(f"=== [{i + 1}/{len(todo)}] {name} repeat={repeat_idx} ===")
            try:
                row = run_cell(cell, repeat_idx, name)
                row["name"] = name
                row["status"] = "ok"
            except Exception as exc:
                row = {
                    "name": name,
                    "repeat": repeat_idx,
                    "cell": {
                        k: (str(v) if v == float("inf") else v) for k, v in cell.items()
                    },
                    "status": f"error: {exc}",
                    "error_type": type(exc).__name__,
                    "attempt_logs": _attempt_log_lineage(name, repeat_idx),
                    "source": _source_identity(),
                    "gpu_process_residue_after_cleanup": _gpu_process_residue(),
                }
            row_out = {
                k: (str(v) if isinstance(v, float) and v == float("inf") else v)
                for k, v in row.items()
            }
            for sub in ("cell", "bench"):
                if isinstance(row_out.get(sub), dict):
                    row_out[sub] = {
                        k: (str(v) if isinstance(v, float) and v == float("inf") else v)
                        for k, v in row_out[sub].items()
                    }
            with open(opts.out, "a") as f:
                f.write(json.dumps(row_out) + "\n")
            bench = row_out.get("bench") or {}
            print(
                json.dumps(
                    {
                        "name": row_out.get("name"),
                        "status": row_out.get("status"),
                        "output_throughput": bench.get("output_throughput"),
                        "p50_tpot_ms": bench.get("median_tpot_ms"),
                        "p99_tpot_ms": bench.get("p99_tpot_ms"),
                        "concurrency": bench.get("concurrency"),
                        "cache_hit_rate_pct": row_out.get("cache_hit_rate_pct"),
                        "export_coverage_frac": row_out.get("export_coverage_frac"),
                        "gpu_mem_max_single_mb": row_out.get("gpu_mem_max_single_mb"),
                    }
                )
            )
    finally:
        master.terminate()
        try:
            master.wait(timeout=30)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait(timeout=30)
        master_stdout.close()
        master_stderr.close()
        run_summary = {
            "master_exit_code": master.returncode,
            "master_logs": {
                "stdout": {
                    "path": master_stdout.name,
                    "sha256": _sha256(master_stdout.name),
                },
                "stderr": {
                    "path": master_stderr.name,
                    "sha256": _sha256(master_stderr.name),
                },
            },
            "gpu_process_residue": _gpu_process_residue(),
            "source": _source_identity(),
            "output": opts.out,
        }
        with open(opts.out + ".run.json", "w") as summary:
            json.dump(run_summary, summary, sort_keys=True)


if __name__ == "__main__":
    main()
