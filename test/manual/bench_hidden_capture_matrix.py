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
- the Mooncake capture store is CLEARED after warmup and before the measured
  run, so export coverage counts only measured-run samples. The recorded
  ``pre_measure_keys_removed`` is a STORE-KEY count (meta + tensor keys, and
  may include tensor-key residue from the previous cell whose meta counting
  left them behind), NOT a number of warmup samples;
- export coverage is observed at a FIXED 20-second post-run drain horizon
  (this client build has no non-destructive count, so no stability polling);
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
import json
import os
import re
import subprocess
import tempfile
import threading
import time

import requests

from sglang.benchmark.serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    get_benchmark_args,
    popen_launch_server,
)

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
BASE_MASTER_PORT = 50060
BASE_SERVER_PORT = 21500
STORE_ID = "matrix_capture"
MASTER_BIN = "/usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master"
SEED = 42
DRAIN_HORIZON_S = 20.0

# Set per-shard in main(); defaults for shard 0.
MASTER_PORT = BASE_MASTER_PORT
SERVER_URL = f"http://127.0.0.1:{BASE_SERVER_PORT}"

_STATS_RE = re.compile(r"hidden capture stats: (\{.*\})")


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
    ]
    if cell["features"]:
        args += ["--chunked-prefill-size", "8192"]
        args += ["--cuda-graph-max-bs-decode", "64"]
    else:
        # Bare stack: no chunking, no radix cache, no CUDA graphs.
        args += ["--chunked-prefill-size", "-1"]
        args += ["--disable-radix-cache"]
        args += ["--cuda-graph-backend-prefill", "disabled"]
        args += ["--disable-decode-cuda-graph"]
    if cell["capture"]:
        args += ["--enable-hidden-state-capture"]
    return args


class GpuMemSampler:
    """1 Hz during-run nvidia-smi sampler: max-over-time of per-GPU max and
    of the all-GPU sum (labels what each number means, unlike a single
    post-run snapshot). Restricted to CUDA_VISIBLE_DEVICES when set, so
    parallel shards on other GPUs don't pollute the numbers."""

    def __init__(self):
        self.max_single_mb = 0
        self.max_sum_mb = 0
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
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def _mooncake_store():
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    rc = store.setup(
        "localhost",
        "P2PHANDSHAKE",
        1 * 1024**3,
        64 * 1024**2,
        "tcp",
        "",
        f"127.0.0.1:{MASTER_PORT}",
    )
    return store if rc == 0 else None


def _clear_capture_store(store):
    if store is None or not hasattr(store, "remove_by_regex"):
        return -1
    return int(store.remove_by_regex(f"{STORE_ID}/.*"))


def _count_meta(store):
    if store is None or not hasattr(store, "remove_by_regex"):
        return -1
    # Counting via removal resets state for the next cell; matrix data is
    # disposable. remove_by_regex returns the number of keys removed.
    return int(store.remove_by_regex(f"{STORE_ID}/.*/g0/meta"))


def _drained_export_count(store):
    """Meta-key count at a FIXED 20-second post-run drain horizon.

    This client build has no non-destructive count/scan, so there is no
    stability polling — just a fixed wait, then one destructive count
    (remove_by_regex returns the number of keys removed, which for the meta
    pattern equals exported samples; it leaves tensor keys behind for the
    next cell's pre-measure clear to sweep)."""
    time.sleep(DRAIN_HORIZON_S)
    return _count_meta(store)


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


def run_cell(cell, repeat_idx):
    env = {
        **os.environ,
        "SGLANG_RAGGED_VERIFY_MODE": "compact",
    }
    if cell["capture"]:
        env.update(
            {
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": STORE_ID,
                "MOONCAKE_MASTER": f"127.0.0.1:{MASTER_PORT}",
                "MOONCAKE_PROTOCOL": "tcp",
                # Size the bench store above one run's full sample volume;
                # otherwise the store's eviction watermark rejects puts
                # mid-run and coverage measures the BENCH store, not the
                # capture pipeline. v2 samples are ~4.5k tokens x ~57KB/token
                # -> ~260MB/sample x 300 requests needs headroom.
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "96gb",
            }
        )
    log_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", prefix="matrix_server_", delete=False
    )
    err_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".err", prefix="matrix_server_", delete=False
    )
    process = popen_launch_server(
        TARGET_MODEL,
        SERVER_URL,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_server_args(cell),
        env=env,
        return_stdout_stderr=(log_file, err_file),
    )
    store = _mooncake_store() if cell["capture"] else None
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

        warmup = copy.deepcopy(args)
        warmup.num_prompts = 16
        warmup.max_concurrency = None
        warmup.cache_report = False
        asyncio.run(asyncio.to_thread(run_benchmark, warmup))
        if cell["prefix"] == "cold":
            requests.post(SERVER_URL + "/flush_cache", timeout=30)
        # Remove warmup exports so coverage counts ONLY the measured run.
        warmup_exports = _clear_capture_store(store)

        with GpuMemSampler() as mem:
            res = asyncio.run(asyncio.to_thread(run_benchmark, args))

        row = {
            "cell": {k: v for k, v in cell.items()},
            "repeat": repeat_idx,
            "server_args": " ".join(_server_args(cell)),
            "gpu_mem_max_single_mb": mem.max_single_mb,
            "gpu_mem_max_sum_mb": mem.max_sum_mb,
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
        }
        if cell["capture"]:
            exported = _drained_export_count(store)
            # Store-KEY count from the pre-measure clear (meta + tensor keys,
            # possibly including previous-cell tensor residue) — NOT a number
            # of warmup samples.
            row["pre_measure_keys_removed"] = warmup_exports
            row["exported_samples_at_drain"] = exported
            row["drain_horizon_s"] = DRAIN_HORIZON_S
            row["export_coverage_frac"] = (
                exported / max(1, res["completed"]) if exported >= 0 else None
            )
            row["capture_counters_last_log"] = _parse_capture_counters(err_file.name)
        return row
    finally:
        if store is not None:
            store.close()
        kill_process_tree(process.pid)
        for f in (log_file, err_file):
            try:
                f.close()
                os.unlink(f.name)
            except OSError:
                pass


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
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cells",
        choices=["regression", "supplement", "all"],
        default="regression",
    )
    parser.add_argument("--only", default=None, help="substring filter on cell name")
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="parallel-shard index: offsets server + mooncake-master ports "
        "so shards on different GPUs don't collide",
    )
    opts = parser.parse_args()

    global MASTER_PORT, SERVER_URL
    MASTER_PORT = BASE_MASTER_PORT + opts.shard
    SERVER_URL = f"http://127.0.0.1:{BASE_SERVER_PORT + opts.shard}"

    master = subprocess.Popen(
        [
            MASTER_BIN if os.path.exists(MASTER_BIN) else "mooncake_master",
            "--port",
            str(MASTER_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    try:
        todo = []
        if opts.cells in ("regression", "all"):
            todo += regression_cells()
        if opts.cells in ("supplement", "all"):
            todo += supplement_cells()
        if opts.only:
            todo = [t for t in todo if opts.only in t[0]]

        for i, (name, repeat_idx, cell) in enumerate(todo):
            print(f"=== [{i + 1}/{len(todo)}] {name} repeat={repeat_idx} ===")
            try:
                row = run_cell(cell, repeat_idx)
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


if __name__ == "__main__":
    main()
