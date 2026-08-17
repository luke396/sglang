"""Manual A0-vs-A1 overhead benchmark for hidden-state capture.

Runs the same fixed workload twice against a DSpark server — capture off (A0)
then capture on (A1) — and reports serving-side deltas (throughput, TTFT/TPOT
p50/p99) plus capture-side counters (exported samples).

This is smoke-level evidence on one machine/model; the production gate in the
capture plan (real traffic mix, target hardware, Nsight/DCGM attribution)
still applies.

Usage:
    PYTHONPATH=python python3 test/manual/bench_hidden_capture_overhead.py \
        [--num-prompts 200] [--request-rate 8] \
        [--input-len 1024] [--output-len 128]
"""

import argparse
import json
import os
import shutil
import tempfile

from sglang.test.test_utils import is_in_ci, run_bench_serving

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"


def _server_args(capture: bool, dp: int = 1):
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
    if dp > 1:
        args += ["--dp", str(dp)]
    if capture:
        args.append("--enable-hidden-state-capture")
    return args


def _run(name: str, capture: bool, opts) -> dict:
    capture_dir = None
    mooncake = capture and opts.sink == "mooncake"
    if capture:
        os.environ["SGLANG_HIDDEN_CAPTURE_SINK"] = opts.sink
        if mooncake:
            os.environ["MOONCAKE_MASTER"] = f"127.0.0.1:{opts.mooncake_port}"
            os.environ["MOONCAKE_PROTOCOL"] = "tcp"
            os.environ["SGLANG_HIDDEN_CAPTURE_STORE_ID"] = "bench_capture"
        else:
            capture_dir = tempfile.mkdtemp(prefix="hidden_capture_bench_")
            os.environ["SGLANG_HIDDEN_CAPTURE_DIR"] = capture_dir
    os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "compact"

    res = run_bench_serving(
        model=TARGET_MODEL,
        num_prompts=opts.num_prompts,
        request_rate=opts.request_rate,
        other_server_args=_server_args(capture, dp=opts.dp),
        random_input_len=opts.input_len,
        random_output_len=opts.output_len,
        need_warmup=True,
        seed=42,
    )
    out = {
        "name": name,
        "output_throughput": res["output_throughput"],
        "input_throughput": res["input_throughput"],
        "mean_ttft_ms": res["mean_ttft_ms"],
        "p99_ttft_ms": res["p99_ttft_ms"],
        "mean_tpot_ms": res["mean_tpot_ms"],
        "p99_tpot_ms": res["p99_tpot_ms"],
        "completed": res["completed"],
    }
    if capture_dir:
        ckpts = [f for f in os.listdir(capture_dir) if f.endswith(".ckpt")]
        out["exported_samples"] = len(ckpts)
        out["export_rate"] = len(ckpts) / max(1, res["completed"])
        # ~50 MB per 1k-token sample; a few rounds fill /tmp. Count, then drop.
        shutil.rmtree(capture_dir, ignore_errors=True)
    elif mooncake:
        out["exported_samples"] = _count_mooncake_samples(opts.mooncake_port)
        out["export_rate"] = out["exported_samples"] / max(1, res["completed"])
    return out


def _count_mooncake_samples(port: int) -> int:
    """Count exported meta keys (best effort: the master evicts by lease, so
    this undercounts under sustained load; still a sanity signal)."""
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    rc = store.setup(
        "localhost",
        "P2PHANDSHAKE",
        1 * 1024**3,
        64 * 1024**2,
        "tcp",
        "",
        f"127.0.0.1:{port}",
    )
    if rc != 0:
        return -1
    try:
        # No prefix-scan API on this client build; count via removal-by-regex
        # dry alternative is unavailable, so probe is_exist on nothing and
        # return -1 to signal "not countable" unless remove_by_regex exists.
        if hasattr(store, "remove_by_regex"):
            # remove_by_regex returns the number of keys removed; counting by
            # removing meta keys AFTER the bench is acceptable (bench data is
            # disposable) and gives an exact produced-count lower bound.
            prefix = int(store.remove_by_regex("bench_capture/_samples/.*/meta"))
            legacy = int(store.remove_by_regex("bench_capture/.*/g0/meta"))
            return prefix + legacy
        return -1
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--sink", choices=["file", "mooncake"], default="file")
    parser.add_argument("--mooncake-port", type=int, default=50053)
    parser.add_argument("--dp", type=int, default=1)
    opts = parser.parse_args()
    assert not is_in_ci(), "manual benchmark; do not run in CI"

    master = None
    if opts.sink == "mooncake":
        master_bin = "/usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master"
        import subprocess

        master = subprocess.Popen(
            [
                master_bin if os.path.exists(master_bin) else "mooncake_master",
                "--port",
                str(opts.mooncake_port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time

        time.sleep(2)
    try:
        _bench(opts)
    finally:
        if master is not None:
            master.terminate()


def _bench(opts):

    runs = {"A0": [], "A1": []}
    # Alternate A0/A1 within each round so slow environment drift (thermals,
    # page cache) biases both arms equally.
    for _ in range(opts.rounds):
        runs["A0"].append(_run("A0 capture-off", capture=False, opts=opts))
        runs["A1"].append(_run("A1 capture-on", capture=True, opts=opts))

    print(json.dumps(runs, indent=2))
    print(f"\n=== A1 vs A0 over {opts.rounds} round(s) (mean [min..max]) ===")
    for key, higher_is_better in [
        ("output_throughput", True),
        ("mean_ttft_ms", False),
        ("p99_ttft_ms", False),
        ("mean_tpot_ms", False),
        ("p99_tpot_ms", False),
    ]:
        a0_vals = [r[key] for r in runs["A0"]]
        a1_vals = [r[key] for r in runs["A1"]]
        base = sum(a0_vals) / len(a0_vals)
        on = sum(a1_vals) / len(a1_vals)
        pct = (on - base) / base * 100 if base else float("nan")
        direction = "regression" if (pct < 0) == higher_is_better else "ok"
        print(
            f"{key}: {base:.2f} [{min(a0_vals):.2f}..{max(a0_vals):.2f}] -> "
            f"{on:.2f} [{min(a1_vals):.2f}..{max(a1_vals):.2f}]  "
            f"({pct:+.2f}%, {direction})"
        )
    total_exported = sum(r.get("exported_samples", 0) for r in runs["A1"])
    total_completed = sum(r["completed"] for r in runs["A1"])
    print(f"exported_samples: {total_exported} / {total_completed}")


if __name__ == "__main__":
    main()
