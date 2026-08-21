"""Capture-attribution runner: paired capture-on/off serving runs, two modes.

The measurement instrument behind issue #10 lever acceptance (issue #11).
One runner, one server-launch path, two modes:

- ``--mode bare``: no profiler.  Produces clean per-request TTFT/ITL streams;
  ``--report on.json off.json`` computes the paired, length-bucketed deltas.
  Time numbers are only valid from this mode.
- ``--mode trace``: identical run bracketed by an nsys session (launch ->
  health -> warmup OUTSIDE the collection window -> start -> burst -> stop),
  exporting a sqlite for test/manual/nsys_report.py.  nsys intercepts every
  host CUDA API call and the arms differ by tens of thousands of calls, so
  latency numbers from this mode are systematically inflated: counts and
  attribution only.

Workload contract (pinned; a before/after comparison is only valid when every
item matches):

- Dataset: bench_serving ``custom`` JSONL (``{"conversations": [{"content":
  prompt}, {"content": completion}]}``).  The completion's token count is the
  request's output length (replayed exactly: ignore_eos=True).  Real-traffic
  files are sampled from the private SpecLoop log and NEVER enter this repo;
  run.json records sha256, row count and the input-length distribution so a
  comparison can assert same-file.
- Arrival schedule: Poisson at ``--request-rate`` under ``--seed`` (both the
  row shuffle and the inter-arrival draws derive from the seed, so paired
  arms replay identical schedules; the paired report asserts it).
- Sampling: temperature 0.9 (matches the 2026-08-20 attribution run and real
  traffic; sampled accept lengths make kernel counts arm-variant, which the
  nsys reader's parity tolerances account for).
- Warmup requests run before the measured window, then the prefix cache is
  flushed so both arms enter measurement with an equally cold radix tree.
- Capture arm: mooncake sink, WINDOW_S == PERIOD_S (every request captured).

Instrument-validity guards (a leg that fails these is method-invalid, not a
result): the mooncake master binary defaults to the SERVER interpreter's own
mooncake package (a version-mismatched master rejects mount_segment, store
setup fails with -900 and capture runs silently degraded while the bench
completes normally); the capture arm fails fast right after warmup if the
pipeline staged nothing or any degraded-mode counter is nonzero, and fails
after the burst unless every measured request exported.  Source identity is
resolved once at startup, so a staging-dir launch without
SGLANG_CHARACTERIZATION_SOURCE_ROOT dies in seconds, not at the manifest
write after a completed leg.

Convert the 2026-08-20 burst format with ``--convert-burst old.jsonl
new.jsonl`` (synthesizes a completion with the row's max_new_tokens tokens).

Example (one leg):
    python3 bench_capture_attribution.py --mode trace --arms on,off \\
        --dataset /path/to/burst-16-32k.custom.jsonl \\
        --gpu 2 --port 21620 --master-port 50220 \\
        --model-path /data/models/Qwen3-8B \\
        --draft-model-path /data/models/dspark_qwen3_8b_block7 \\
        --out ~/capture-baselines/2026-08-20-e9b4917/trace_r0

Archiving convention: every full set lands in
``~/capture-baselines/<date>-<commit>/`` (see test/manual/README-capture.md).
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PYTHON = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "python")


def _load_matrix_module():
    """Import bench_hidden_capture_matrix (same dir) for its shared helpers.

    Single source of truth for _source_identity and the capture drain
    protocol; the matrix owns them, this runner points its module-level
    SERVER_URL at the leg under test.
    """
    spec = importlib.util.spec_from_file_location(
        "bench_hidden_capture_matrix",
        os.path.join(_HERE, "bench_hidden_capture_matrix.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TEMPERATURE = 0.9
WARMUP_REQUESTS = 2
WARMUP_OUTPUT_TOKENS = 16
SERVER_READY_TIMEOUT_S = 600.0
NSYS_TRACE = "cuda,nvtx,osrt,cudnn,cublas"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _server_command(args, port):
    return [
        args.server_python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--speculative-draft-attention-backend",
        "fa3",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        args.draft_model_path,
        "--dp-size",
        "1",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.75",
        "--page-size",
        "1",
        "--random-seed",
        str(args.seed),
    ]


def _leg_env(args, arm, store_id):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONPATH": _REPO_PYTHON,
        "SGLANG_RAGGED_VERIFY_MODE": "compact",
        # Environment pinned by the 2026-08-20 attribution session; kept so
        # new baselines stay comparable run-to-run on this machine.
        "FLASHINFER_USE_CUDA_NORM": "1",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    }
    if arm == "on":
        env.update(
            {
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": store_id,
                # Full-rate capture: every arrival is inside the window.
                "SGLANG_HIDDEN_CAPTURE_WINDOW_S": "3600",
                "SGLANG_HIDDEN_CAPTURE_PERIOD_S": "3600",
                # Long-prompt workloads must export, not sample_too_large-miss
                # (default 16384 silently drops every 16-32k request, leaving
                # the export stage unmeasured; the 2026-08-20 session hit this
                # with 13/14 misses).
                "SGLANG_HIDDEN_CAPTURE_MAX_EXPORT_TOKENS": "32768",
                "MOONCAKE_MASTER": f"127.0.0.1:{args.master_port}",
                "MOONCAKE_PROTOCOL": "tcp",
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": args.mooncake_segment_size,
            }
        )
    return env


def _wait_ready(url, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{url}/health_generate", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server at {url} not ready after {timeout_s}s")


def _load_custom_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _warmup_and_flush(url, dataset_rows):
    for index, row in enumerate(dataset_rows[:WARMUP_REQUESTS]):
        prompt = row["conversations"][0].get(
            "content", row["conversations"][0].get("value", "")
        )
        response = requests.post(
            f"{url}/generate",
            json={
                "text": prompt,
                "rid": f"warmup-{index}",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": WARMUP_OUTPUT_TOKENS,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
    # Equal-cold radix state for the measured window in both arms.
    requests.post(f"{url}/flush_cache", timeout=60).raise_for_status()


def _bench_args(args, url, leg_dir):
    from sglang.test.test_utils import get_benchmark_args

    bench = get_benchmark_args(
        base_url=url,
        dataset_name="custom",
        dataset_path=args.dataset,
        tokenizer=args.model_path,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    bench.temperature = TEMPERATURE
    # Warmup placement is owned by this runner (outside any trace window).
    bench.warmup_requests = 0
    bench.flush_cache = False
    bench.output_file = os.path.join(leg_dir, "bench_serving.jsonl")
    return bench


class _Nsys:
    def __init__(self, session, output_base):
        self.session = session
        self.output_base = output_base

    def launch_prefix(self):
        return [
            "nsys",
            "launch",
            f"--session-new={self.session}",
            f"--trace={NSYS_TRACE}",
        ]

    def start(self):
        subprocess.run(
            [
                "nsys",
                "start",
                f"--session={self.session}",
                f"--output={self.output_base}",
                "--force-overwrite=true",
                "--sample=none",
                "--cpuctxsw=none",
            ],
            check=True,
        )

    def stop(self):
        subprocess.run(["nsys", "stop", f"--session={self.session}"], check=True)

    def shutdown(self):
        subprocess.run(
            ["nsys", "shutdown", f"--session={self.session}", "--kill=sigterm"],
            check=False,
        )

    def export_sqlite(self):
        rep = f"{self.output_base}.nsys-rep"
        sqlite_path = f"{self.output_base}.sqlite"
        subprocess.run(
            [
                "nsys",
                "export",
                "--type=sqlite",
                f"--output={sqlite_path}",
                "--force-overwrite=true",
                rep,
            ],
            check=True,
        )
        return rep, sqlite_path


def _terminate(process, timeout_s=60):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _default_master_bin(server_python):
    """mooncake_master shipped with the SERVER interpreter's mooncake package.

    The master must version-match the server-side client: a mismatched master
    rejects mount_segment ("invalid rpc arg"), store setup fails with -900,
    and capture runs silently degraded while the bench completes normally.
    Resolving from the server venv makes the default version-matched by
    construction; --master-bin still overrides.
    """
    resolved = subprocess.run(
        [
            server_python,
            "-c",
            "import mooncake, os; print(os.path.join("
            "os.path.dirname(mooncake.__file__), 'mooncake_master'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"mooncake_master not found in the server interpreter's mooncake "
            f"package: {resolved}"
        )
    return resolved


def _assert_capture_live(warmed_stats):
    """Fail fast when the warmed-up capture pipeline is degraded or empty.

    Store-setup failures leave the bench runnable while capture stages
    nothing; catching that here costs seconds instead of a lost leg.
    """
    degraded = {
        name: value
        for name, value in warmed_stats.items()
        if name.startswith("capture_degraded") and value
    }
    if degraded:
        raise AssertionError(f"capture degraded after warmup: {degraded}")
    if not warmed_stats.get("rows_staged_ct"):
        raise AssertionError(
            "capture staged no rows for the warmup requests; store setup "
            "likely failed (check mooncake.log and server.log in the leg dir)"
        )


def _capture_validity_error(stats_delta, drained, completed):
    """None when the measured window's capture coverage is method-valid.

    The counter delta must show a drained pipeline, no degraded-mode
    entries, and one export per measured request — anything else means the
    instrument was not measuring what the report claims (the 2026-08-20
    session's 13/14 sample_too_large misses shipped unnoticed this way).
    """
    if not drained:
        return "capture pipeline did not drain after the burst"
    degraded = {
        name: value
        for name, value in stats_delta.items()
        if name.startswith("capture_degraded") and value
    }
    if degraded:
        return f"degraded-mode counters nonzero in the measured window: {degraded}"
    exported = stats_delta.get("export_ok_ct", 0)
    if exported != completed:
        misses = {
            name: value
            for name, value in stats_delta.items()
            if name.endswith("_miss_ct") and value
        }
        return (
            f"export coverage {exported}/{completed} in the measured window "
            f"(misses: {misses or 'none recorded'})"
        )
    return None


def run_leg(args, matrix, arm, leg_dir, source_identity):
    from sglang.benchmark.serving import run_benchmark

    os.makedirs(leg_dir, exist_ok=True)
    port = args.port
    url = f"http://127.0.0.1:{port}"
    matrix.SERVER_URL = url
    store_id = f"attr_{arm}_{uuid.uuid4().hex[:10]}"
    env = _leg_env(args, arm, store_id)
    dataset_rows = _load_custom_rows(args.dataset)

    master = None
    nsys = None
    server_log = open(os.path.join(leg_dir, "server.log"), "w")
    if arm == "on":
        master_bin = args.master_bin or _default_master_bin(args.server_python)
        master = subprocess.Popen(
            [
                master_bin,
                f"--rpc_port={args.master_port}",
                f"--metrics_port={args.master_port + 1}",
            ],
            stdout=open(os.path.join(leg_dir, "mooncake.log"), "w"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        time.sleep(3)

    command = _server_command(args, port)
    if arm == "on":
        command.append("--enable-hidden-state-capture")
    if args.mode == "trace":
        nsys = _Nsys(
            session=f"attr-{arm}-{os.getpid()}",
            output_base=os.path.join(leg_dir, f"serving-{arm}"),
        )
        command = nsys.launch_prefix() + command

    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=env)
    result = {"mode": args.mode, "arm": arm, "gpu": args.gpu, "store_id": store_id}
    try:
        _wait_ready(url, SERVER_READY_TIMEOUT_S)
        _warmup_and_flush(url, dataset_rows)

        if arm == "on":
            before_snapshots, _, _, drained = matrix._drain_capture()
            if not drained:
                raise AssertionError("capture pipeline did not drain after warmup")
            before_stats = matrix._sum_capture_stats(before_snapshots)
            _assert_capture_live(before_stats)
        else:
            before_stats = {}

        if nsys is not None:
            nsys.start()
        burst_start = time.time()
        bench = run_benchmark(_bench_args(args, url, leg_dir))
        burst_end = time.time()
        if nsys is not None:
            nsys.stop()

        if arm == "on":
            after_snapshots, _, drain_s, drained = matrix._drain_capture()
            after_stats = matrix._sum_capture_stats(after_snapshots)
            stats_delta = matrix._counter_delta(after_stats, before_stats)
            result["capture_drained"] = drained
            result["capture_drain_s"] = drain_s
            result["capture_stats_delta"] = stats_delta
            result["captured_rows"] = stats_delta.get("rows_staged_ct", 0)
            validity_error = _capture_validity_error(
                stats_delta, drained, int(bench.get("completed", 0))
            )
            result["capture_validity_error"] = validity_error
        result.update(
            {
                "burst_start": burst_start,
                "burst_end": burst_end,
                "completed": bench.get("completed"),
                "bench": {
                    key: value
                    for key, value in bench.items()
                    if key not in ("generated_texts", "errors")
                },
            }
        )
    finally:
        if nsys is not None:
            nsys.shutdown()
            time.sleep(5)
        _terminate(server)
        if master is not None:
            _terminate(master, timeout_s=10)
        server_log.close()

    if nsys is not None:
        rep, sqlite_path = nsys.export_sqlite()
        result["nsys_rep"] = {"path": rep, "sha256": _sha256(rep)}
        result["nsys_sqlite"] = {"path": sqlite_path, "sha256": _sha256(sqlite_path)}

    result["contract"] = {
        "dataset": {"path": args.dataset, "sha256": _sha256(args.dataset),
                    "rows": len(dataset_rows)},
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate,
        "seed": args.seed,
        "temperature": TEMPERATURE,
        "warmup_requests": WARMUP_REQUESTS,
        "server_command": _server_command(args, port),
        "capture_env": sorted(
            key for key in _leg_env(args, arm, store_id)
            if key.startswith(("SGLANG_HIDDEN_CAPTURE", "MOONCAKE"))
        ),
    }
    result["source_identity"] = source_identity
    run_json = os.path.join(leg_dir, f"run-{arm}.json")
    with open(run_json, "w") as handle:
        json.dump(result, handle, indent=1)
    validity_error = result.get("capture_validity_error")
    if validity_error:
        raise AssertionError(
            f"leg is method-invalid ({validity_error}); evidence kept at {run_json}"
        )
    print(f"[leg done] mode={args.mode} arm={arm} -> {run_json}")
    return run_json


BUCKETS = (("<2k", 0, 2048), ("2-8k", 2048, 8192), ("8-16k", 8192, 16384),
           ("16-32k", 16384, 32768), (">=32k", 32768, 1 << 40))


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def paired_report(on_path, off_path):
    with open(on_path) as handle:
        on = json.load(handle)
    with open(off_path) as handle:
        off = json.load(handle)
    for leg, name in ((on, "on"), (off, "off")):
        if leg.get("mode") != "bare":
            print(f"WARNING: arm {name} is mode={leg.get('mode')}; "
                  "latency numbers from traced runs are observer-inflated")
    on_bench, off_bench = on["bench"], off["bench"]
    if on["contract"]["dataset"]["sha256"] != off["contract"]["dataset"]["sha256"]:
        raise AssertionError("arms ran different dataset files")
    if on_bench["input_lens"] != off_bench["input_lens"]:
        raise AssertionError(
            "arms saw different request orders (seed/schedule mismatch); "
            "paired deltas are invalid"
        )

    lens = on_bench["input_lens"]
    ttft_on = [value * 1e3 for value in on_bench["ttfts"]]
    ttft_off = [value * 1e3 for value in off_bench["ttfts"]]
    print(f"paired requests: {len(lens)}  "
          f"(on completed={on['completed']}, off completed={off['completed']})")
    print(f"{'bucket':8s} {'n':>4s} {'off_p50':>9s} {'on_p50':>9s} "
          f"{'d_mean':>8s} {'d_p50':>8s} {'d_p90':>8s} {'d_p99':>8s}")
    for label, low, high in BUCKETS:
        indexes = [i for i, length in enumerate(lens) if low <= length < high]
        if not indexes:
            continue
        deltas = [ttft_on[i] - ttft_off[i] for i in indexes]
        print(
            f"{label:8s} {len(indexes):4d} "
            f"{_quantile([ttft_off[i] for i in indexes], 0.5):8.1f}ms "
            f"{_quantile([ttft_on[i] for i in indexes], 0.5):8.1f}ms "
            f"{sum(deltas) / len(deltas):+7.1f}ms "
            f"{_quantile(deltas, 0.5):+7.1f}ms "
            f"{_quantile(deltas, 0.9):+7.1f}ms "
            f"{_quantile(deltas, 0.99):+7.1f}ms"
        )
    itl_on = [gap * 1e3 for request in on_bench["itls"] for gap in request]
    itl_off = [gap * 1e3 for request in off_bench["itls"] for gap in request]
    for q in (0.5, 0.99):
        quantile_on = _quantile(itl_on, q)
        quantile_off = _quantile(itl_off, q)
        print(f"ITL p{int(q * 100)}: off={quantile_off:.2f}ms on={quantile_on:.2f}ms "
              f"({(quantile_on - quantile_off) / quantile_off * 100:+.1f}%)")


def convert_burst(source_path, target_path, tokenizer_path):
    """2026-08-20 burst format -> bench_serving custom format.

    The synthetic completion carries exactly the row's max_new_tokens tokens,
    so custom.py's output_len (completion token count) replays the original
    per-row output length under ignore_eos.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    unit = " ok"
    with open(source_path) as source, open(target_path, "w") as target:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target_tokens = row["max_new_tokens"]
            count = target_tokens
            completion = unit * count
            while len(tokenizer.encode(completion)) > target_tokens and count > 1:
                count -= 1
                completion = unit * count
            while len(tokenizer.encode(completion)) < target_tokens:
                count += 1
                completion = unit * count
            target.write(json.dumps({
                "conversations": [
                    {"content": row["text"]},
                    {"content": completion},
                ]
            }) + "\n")
    print(f"converted {source_path} -> {target_path} "
          f"(sha256 {_sha256(target_path)})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("bare", "trace"))
    parser.add_argument("--arms", default="on,off",
                        help="comma-separated capture arms to run serially")
    parser.add_argument("--dataset", help="custom-format JSONL (workload contract)")
    parser.add_argument("--num-prompts", type=int, default=40)
    parser.add_argument("--request-rate", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=21600)
    parser.add_argument("--master-port", type=int, default=50200)
    parser.add_argument(
        "--mooncake-segment-size",
        default="64gb",
        help="sized above one run's full export volume, or the store's "
        "eviction watermark rejects puts mid-run and coverage measures the "
        "bench store instead of the capture pipeline (40 x ~20k-token "
        "samples x ~49KB/row is ~38GB)",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model-path",
                        default="deepseek-ai/dspark_qwen3_8b_block7")
    parser.add_argument(
        "--server-python",
        default=sys.executable,
        help="interpreter for the server process (a venv whose mooncake "
        "version satisfies the capture sink; PYTHONPATH still points at "
        "this repo)",
    )
    parser.add_argument("--master-bin",
                        help="mooncake_master binary (default: alongside the "
                        "SERVER interpreter's mooncake package, so master and "
                        "server client are version-matched)")
    parser.add_argument("--out", help="leg artifact directory")
    parser.add_argument("--report", nargs=2, metavar=("ON_JSON", "OFF_JSON"),
                        help="paired bare-mode TTFT/ITL delta report")
    parser.add_argument("--convert-burst", nargs=2, metavar=("IN", "OUT"),
                        help="convert 2026-08-20 burst jsonl to custom format")
    args = parser.parse_args()

    if args.report:
        paired_report(*args.report)
        return
    if args.convert_burst:
        convert_burst(*args.convert_burst, tokenizer_path=args.model_path)
        return
    if not (args.mode and args.dataset and args.out):
        parser.error("--mode, --dataset and --out are required to run legs")

    sys.path.insert(0, _REPO_PYTHON)
    matrix = _load_matrix_module()
    # Resolve once, before any leg: from a staging dir this needs
    # SGLANG_CHARACTERIZATION_SOURCE_ROOT, and failing here costs seconds
    # instead of a completed leg dying at the manifest write.
    try:
        source_identity = matrix._source_identity()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"cannot resolve source identity ({error}); when running from a "
            "staging copy, set SGLANG_CHARACTERIZATION_SOURCE_ROOT to the "
            "real worktree"
        ) from error
    run_jsons = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm not in ("on", "off"):
            parser.error(f"unknown arm {arm!r}")
        run_jsons[arm] = run_leg(args, matrix, arm, args.out, source_identity)
    if args.mode == "bare" and set(run_jsons) == {"on", "off"}:
        paired_report(run_jsons["on"], run_jsons["off"])


if __name__ == "__main__":
    main()
