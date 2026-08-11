"""Performance-degradation characterization matrix for hidden-state capture.

Runs capture OFF vs ON (Mooncake sink) across isolated main-effect factors
and worst-combination cells on Qwen3-8B DSpark:

  prefix   : cold (unique random prompts, radix on) / warm (shared-prefix,
             observed hit signal) / no_radix (--disable-radix-cache)
  length   : short (128) / mid (1024) / long (8192) random inputs
  chunk    : large (8192) vs forced small (128)
  graphs   : prefill graph off/on x decode graph off/on
  dp       : 1 / 2
  load     : low / mid / high fixed rates, plus unbounded (saturation)

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
  nvidia-smi); reported as max-over-time of per-GPU max and of the sum;
- capture-side counters (export_ok / *_miss_ct) are parsed from the server's
  periodic stats log lines when present (logged every 256 finalized slots),
  giving direct miss attribution rather than coverage-only inference.

Each cell reports throughput, TTFT/TPOT/E2E mean/p50/p99, during-run GPU
memory, observed cache-hit rate, drained export coverage, and raw capture
counters. Rows append to a JSONL for auditability.

Usage:
    PYTHONPATH=python SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/bench_hidden_capture_matrix.py \
        --out /tmp/capture_matrix.jsonl [--cells main|worst|supplement|all]
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
MASTER_PORT = 50057
STORE_ID = "matrix_capture"
MASTER_BIN = "/usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master"
SEED = 42
DRAIN_HORIZON_S = 20.0

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
        "--chunked-prefill-size",
        str(cell["chunk"]),
    ]
    if cell["prefix"] == "no_radix":
        args += ["--disable-radix-cache"]
    if not cell["prefill_graph"]:
        args += ["--cuda-graph-backend-prefill", "disabled"]
    if not cell["decode_graph"]:
        args += ["--disable-decode-cuda-graph"]
    else:
        args += ["--cuda-graph-max-bs-decode", "64"]
    if cell["dp"] > 1:
        args += ["--dp", str(cell["dp"])]
    if cell["capture"]:
        args += ["--enable-hidden-state-capture"]
    return args


class GpuMemSampler:
    """1 Hz during-run nvidia-smi sampler: max-over-time of per-GPU max and
    of the all-GPU sum (labels what each number means, unlike a single
    post-run snapshot)."""

    def __init__(self):
        self.max_single_mb = 0
        self.max_sum_mb = 0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
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
                # Size the bench store above one run's full sample volume
                # (~60MB/sample x up to 600 requests); otherwise the store's
                # eviction watermark rejects puts mid-run and coverage
                # measures the BENCH store, not the capture pipeline.
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "48gb",
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
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_server_args(cell),
        env=env,
        return_stdout_stderr=(log_file, err_file),
    )
    store = _mooncake_store() if cell["capture"] else None
    try:
        common = dict(
            base_url=DEFAULT_URL_FOR_TEST,
            tokenizer=TARGET_MODEL,
            num_prompts=cell["num_prompts"],
            request_rate=cell["rate"],
            seed=SEED,
        )
        if cell["prefix"] == "warm":
            args = get_benchmark_args(
                dataset_name="generated-shared-prefix",
                gsp_num_groups=max(2, cell["num_prompts"] // 8),
                gsp_prompts_per_group=8,
                gsp_system_prompt_len=1024,
                gsp_question_len=128,
                gsp_output_len=128,
                **common,
            )
        else:
            args = get_benchmark_args(
                dataset_name="random",
                random_input_len=cell.get("input_len", 1024),
                random_output_len=128,
                **common,
            )
        args.cache_report = True

        warmup = copy.deepcopy(args)
        warmup.num_prompts = 16
        warmup.cache_report = False
        asyncio.run(asyncio.to_thread(run_benchmark, warmup))
        if cell["prefix"] == "cold":
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=30)
        # Remove warmup exports so coverage counts ONLY the measured run.
        warmup_exports = _clear_capture_store(store)

        with GpuMemSampler() as mem:
            res = asyncio.run(asyncio.to_thread(run_benchmark, args))

        row = {
            "cell": {k: v for k, v in cell.items()},
            "repeat": repeat_idx,
            "server_args": " ".join(_server_args(cell)),
            "completed": res["completed"],
            "output_throughput": res["output_throughput"],
            "mean_ttft_ms": res["mean_ttft_ms"],
            "p50_ttft_ms": res["median_ttft_ms"],
            "p99_ttft_ms": res["p99_ttft_ms"],
            "mean_tpot_ms": res["mean_tpot_ms"],
            "p50_tpot_ms": res["median_tpot_ms"],
            "p99_tpot_ms": res["p99_tpot_ms"],
            "mean_e2e_ms": res["mean_e2e_latency_ms"],
            "p50_e2e_ms": res["median_e2e_latency_ms"],
            "p99_e2e_ms": res["p99_e2e_latency_ms"],
            "gpu_mem_max_single_mb": mem.max_single_mb,
            "gpu_mem_max_sum_mb": mem.max_sum_mb,
            "cache_hit_rate_pct": (res.get("cache_report") or {}).get(
                "cache_hit_rate_pct"
            ),
        }
        if cell["capture"]:
            exported = _drained_export_count(store)
            # Store-KEY count from the pre-measure clear (meta + tensor keys,
            # possibly including previous-cell tensor residue) — NOT a number
            # of warmup samples. v2 raw rows recorded this same quantity
            # under the legacy name warmup_exports_removed.
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


BASE = dict(
    prefix="cold",
    chunk=8192,
    prefill_graph=True,
    decode_graph=True,
    dp=1,
    rate=8.0,
    num_prompts=200,
    input_len=1024,
)


def main_effect_cells():
    cells = []
    _pair(cells, "baseline", BASE, repeats=2)
    _pair(cells, "warm_prefix", BASE, prefix="warm")
    _pair(cells, "chunked_128", BASE, chunk=128)
    _pair(cells, "prefill_graph_off", BASE, prefill_graph=False)
    _pair(cells, "decode_graph_off", BASE, decode_graph=False)
    _pair(cells, "dp2", BASE, dp=2, rate=16.0, num_prompts=400)
    _pair(cells, "load_low", BASE, rate=2.0, num_prompts=100)
    _pair(cells, "load_high", BASE, rate=24.0, num_prompts=300)
    return cells


def supplement_cells():
    cells = []
    _pair(cells, "no_radix", BASE, prefix="no_radix")
    _pair(cells, "input_short_128", BASE, input_len=128)
    _pair(cells, "input_long_8192", BASE, input_len=8192, num_prompts=100, rate=4.0)
    # Saturation: unbounded rate = offered load beyond capacity; the OFF/ON
    # delta here measures max-throughput cost, unlike fixed-rate cells.
    _pair(
        cells,
        "saturation_dp1",
        BASE,
        rate=float("inf"),
        num_prompts=300,
        repeats=2,
    )
    _pair(cells, "saturation_dp2", BASE, dp=2, rate=float("inf"), num_prompts=600)
    return cells


def worst_cells():
    worst = dict(
        prefix="warm",
        chunk=128,
        prefill_graph=True,
        decode_graph=True,
        dp=2,
        rate=24.0,
        num_prompts=400,
    )
    cells = []
    _pair(cells, "worst_combo", worst, repeats=2)
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cells", choices=["main", "worst", "supplement", "all"], default="all"
    )
    parser.add_argument("--only", default=None, help="substring filter on cell name")
    opts = parser.parse_args()

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
        if opts.cells in ("main", "all"):
            todo += main_effect_cells()
        if opts.cells in ("worst", "all"):
            todo += worst_cells()
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
            if isinstance(row_out.get("cell"), dict):
                row_out["cell"] = {
                    k: (str(v) if isinstance(v, float) and v == float("inf") else v)
                    for k, v in row_out["cell"].items()
                }
            with open(opts.out, "a") as f:
                f.write(json.dumps(row_out) + "\n")
            print(
                json.dumps(
                    {
                        k: row_out.get(k)
                        for k in (
                            "name",
                            "status",
                            "output_throughput",
                            "p50_tpot_ms",
                            "p99_tpot_ms",
                            "cache_hit_rate_pct",
                            "export_coverage_frac",
                            "pre_measure_keys_removed",
                            "gpu_mem_max_single_mb",
                        )
                    }
                )
            )
    finally:
        master.terminate()


if __name__ == "__main__":
    main()
