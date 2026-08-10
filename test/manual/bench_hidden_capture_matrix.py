"""Performance-degradation characterization matrix for hidden-state capture.

Runs capture OFF vs ON (Mooncake sink) across isolated main-effect factors
and a few worst-combination cells on Qwen3-8B DSpark:

  prefix   : cold (unique random prompts) vs warm (generated-shared-prefix,
             hit signal read from the server's cache report, not assumed)
  chunk    : large (8192 = effectively unchunked at these lengths) vs forced
             small (128)
  graphs   : prefill graph off/on x decode graph off/on
  dp       : 1 / 2
  load     : low / mid / high request rates

Fixed seed and prompt shapes; 16-request warmup per run; key cells repeated.
Each cell reports throughput, TTFT/TPOT/E2E mean/p50/p99, peak GPU memory,
cache-hit rate, and Mooncake export coverage. Results append to a JSONL for
auditability.

Usage:
    PYTHONPATH=python SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/bench_hidden_capture_matrix.py \
        --out /tmp/capture_matrix.jsonl [--cells main] [--repeats 2]
"""

import argparse
import asyncio
import copy
import json
import os
import subprocess
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


def _gpu_peak_mb(pids):
    """Max used_memory across the server's GPUs via nvidia-smi query."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return max(int(x) for x in out.split())
    except Exception:
        return -1


def _mooncake_export_count():
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
    if rc != 0:
        return -1
    try:
        if hasattr(store, "remove_by_regex"):
            # Counting by removal is fine: matrix data is disposable, and it
            # resets the store between cells.
            return int(store.remove_by_regex(f"{STORE_ID}/.*/g0/meta"))
        return -1
    finally:
        store.close()


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
            }
        )
    process = popen_launch_server(
        TARGET_MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_server_args(cell),
        env=env,
    )
    try:
        if cell["prefix"] == "warm":
            args = get_benchmark_args(
                base_url=DEFAULT_URL_FOR_TEST,
                dataset_name="generated-shared-prefix",
                tokenizer=TARGET_MODEL,
                num_prompts=cell["num_prompts"],
                request_rate=cell["rate"],
                seed=SEED,
                gsp_num_groups=max(2, cell["num_prompts"] // 8),
                gsp_prompts_per_group=8,
                gsp_system_prompt_len=1024,
                gsp_question_len=128,
                gsp_output_len=128,
            )
        else:
            args = get_benchmark_args(
                base_url=DEFAULT_URL_FOR_TEST,
                dataset_name="random",
                tokenizer=TARGET_MODEL,
                num_prompts=cell["num_prompts"],
                random_input_len=cell.get("input_len", 1024),
                random_output_len=128,
                request_rate=cell["rate"],
                seed=SEED,
            )
        args.cache_report = True

        warmup = copy.deepcopy(args)
        warmup.num_prompts = 16
        warmup.cache_report = False
        asyncio.run(asyncio.to_thread(run_benchmark, warmup))
        if cell["prefix"] == "cold":
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache", timeout=30)

        res = asyncio.run(asyncio.to_thread(run_benchmark, args))
        peak_mb = _gpu_peak_mb(None)

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
            "peak_gpu_mb": peak_mb,
            "cache_hit_rate_pct": (res.get("cache_report") or {}).get(
                "cache_hit_rate_pct"
            ),
        }
        if cell["capture"]:
            time.sleep(3)  # let the export queue drain
            exported = _mooncake_export_count()
            row["exported_samples"] = exported
            row["export_rate"] = (
                exported / max(1, res["completed"]) if exported >= 0 else None
            )
        return row
    finally:
        kill_process_tree(process.pid)


def main_effect_cells():
    """OFF/ON pairs isolating one factor each around a common baseline."""
    base = dict(
        prefix="cold",
        chunk=8192,
        prefill_graph=True,
        decode_graph=True,
        dp=1,
        rate=8.0,
        num_prompts=200,
        input_len=1024,
    )
    cells = []

    def pair(name, repeats=1, **overrides):
        for capture in (False, True):
            for r in range(repeats):
                cells.append(
                    ("{}|cap={}".format(name, int(capture)), r,
                     {**base, **overrides, "capture": capture})
                )

    pair("baseline", repeats=2)
    pair("warm_prefix", prefix="warm")
    pair("chunked_128", chunk=128)
    pair("prefill_graph_off", prefill_graph=False)
    pair("decode_graph_off", decode_graph=False)
    pair("dp2", dp=2, rate=16.0, num_prompts=400)
    pair("load_low", rate=2.0, num_prompts=100)
    pair("load_high", rate=24.0, num_prompts=300)
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
    for capture in (False, True):
        for r in range(2):
            cells.append(
                ("worst_combo|cap={}".format(int(capture)), r,
                 {**worst, "capture": capture})
            )
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells", choices=["main", "worst", "all"], default="all")
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
        if opts.only:
            todo = [t for t in todo if opts.only in t[0]]

        for i, (name, repeat_idx, cell) in enumerate(todo):
            print(f"=== [{i + 1}/{len(todo)}] {name} repeat={repeat_idx} ===")
            try:
                row = run_cell(cell, repeat_idx)
                row["name"] = name
                row["status"] = "ok"
            except Exception as exc:  # keep the matrix going; record the failure
                row = {
                    "name": name,
                    "repeat": repeat_idx,
                    "cell": cell,
                    "status": f"error: {exc}",
                }
            with open(opts.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in (
                "name", "status", "output_throughput", "p50_tpot_ms",
                "p99_tpot_ms", "p99_ttft_ms", "cache_hit_rate_pct",
                "export_rate", "peak_gpu_mb")}))
    finally:
        master.terminate()


if __name__ == "__main__":
    main()
