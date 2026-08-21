"""E2E: DSpark server with SGLANG_HIDDEN_CAPTURE_SINK=mooncake against a local
mooncake_master; consume exported samples with SpecForge-style zero-copy reads.

Manual (not CI-registered): requires the mooncake binaries and a free master
port. Run:
    PYTHONPATH=python SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/test_hidden_capture_mooncake_e2e.py

The same harness also owns bounded serving fault characterization so the
repository keeps one real-Mooncake lifecycle entry point::
    python3 test/manual/test_hidden_capture_mooncake_e2e.py \
        --fault-scenario all --fault-out /tmp/faults.json \
        --fault-artifact-dir /tmp/fault-logs
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
from hidden_capture_v6_reader import read_v6_sample

from sglang.srt.state_capturer.hidden_host import (
    HiddenCaptureStats,
    HiddenHostSidecar,
)
from sglang.srt.state_capturer.hidden_mooncake import MooncakeHiddenSink
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
MASTER_PORT = 50052
STORE_ID = "e2e_capture"
# The pip wrapper chmods a root-owned path and fails for non-root; call the
# packaged binary directly.


def _default_mooncake_master():
    try:
        import mooncake

        return os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master")
    except Exception:
        return "mooncake_master"


MASTER_BIN = os.environ.get("MOONCAKE_MASTER_BIN") or _default_mooncake_master()


def _protocol_sidecar(stats):
    return HiddenHostSidecar(
        num_slots=64,
        aux_width=8,
        last_width=4,
        dtype=torch.bfloat16,
        stats=stats,
    )


def _write_protocol_rows(sidecar, slots, tokens, seed):
    generator = torch.Generator().manual_seed(seed)
    aux = torch.randn(len(slots), 8, generator=generator).to(torch.bfloat16)
    last = torch.randn(len(slots), 4, generator=generator).to(torch.bfloat16)
    slot_tensor = torch.tensor(slots, dtype=torch.long)
    generations = sidecar.write_rows(
        slots=slot_tensor,
        aux_rows=aux,
        last_rows=last,
        tokens=torch.tensor(tokens, dtype=torch.long),
    )
    return aux, last, dict(zip(slots, generations.tolist()))


def _protocol_sink(store_id, master_address, stats, writer_epoch, store=None):
    injected = {}
    if store is not None:
        from mooncake.store import ReplicateConfig

        payload_config = ReplicateConfig()
        payload_config.replica_num = 1
        manifest_config = ReplicateConfig()
        manifest_config.replica_num = 1
        if hasattr(manifest_config, "with_hard_pin"):
            manifest_config.with_hard_pin = True
        injected = {
            "store": store,
            "replicate_config": payload_config,
            "manifest_config": manifest_config,
            "replicate_config_cls": ReplicateConfig,
        }
    return MooncakeHiddenSink(
        store_id=store_id,
        max_export_tokens=64,
        master_address=master_address,
        stats=stats,
        max_segment_rows=4,
        prefix_lanes=2,
        aux_width=8,
        last_width=4,
        dtype=torch.bfloat16,
        writer_epoch=writer_epoch,
        **injected,
    )


def _publish_protocol_sample(sink, sidecar, sample_id, tokens, slots, own):
    return sink.put_prefix_sample(
        sample_id=sample_id,
        rid=f"rid-{sample_id}",
        tokens=torch.tensor(tokens, dtype=torch.long),
        slots=torch.tensor(slots, dtype=torch.long),
        prompt_len=max(1, len(tokens) - 2),
        own_slot_gens=own,
        sidecar=sidecar,
    )


def _run_real_prefix_protocol_checks(store, master_address):
    """Real-master prefix/restart/missing-object acceptance checks."""
    store_id = "e2e_protocol"
    stats = HiddenCaptureStats()
    sink = _protocol_sink(store_id, master_address, stats, "epoch-a")
    sink.write_fingerprint({"model_path": "protocol", "revision": "v6"})
    sidecar = _protocol_sidecar(stats)
    try:
        tokens_a = [1, 2, 3, 4, 5, 6]
        slots_a = list(range(6))
        aux_a, last_a, own_a = _write_protocol_rows(
            sidecar, slots_a, tokens_a, seed=101
        )
        assert _publish_protocol_sample(sink, sidecar, "a", tokens_a, slots_a, own_a)

        tokens_ab = tokens_a + [7, 8, 9]
        slots_ab = slots_a + [6, 7, 8]
        aux_b, last_b, own_b = _write_protocol_rows(
            sidecar, [6, 7, 8], [7, 8, 9], seed=102
        )
        assert _publish_protocol_sample(sink, sidecar, "ab", tokens_ab, slots_ab, own_b)
        rebuilt_ab = read_v6_sample(store, store_id=store_id, sample_id="ab")
        assert rebuilt_ab["input_ids"].tolist() == tokens_ab
        assert torch.equal(rebuilt_ab["aux"], torch.cat((aux_a, aux_b)))
        assert torch.equal(rebuilt_ab["last_hidden"], torch.cat((last_a, last_b)))

        tokens_ac = tokens_a + [70, 71]
        slots_ac = slots_a + [9, 10]
        aux_c, last_c, own_c = _write_protocol_rows(
            sidecar, [9, 10], [70, 71], seed=103
        )
        assert _publish_protocol_sample(sink, sidecar, "ac", tokens_ac, slots_ac, own_c)
        rebuilt_ac = read_v6_sample(store, store_id=store_id, sample_id="ac")
        assert torch.equal(rebuilt_ac["aux"], torch.cat((aux_a, aux_c)))
        assert torch.equal(rebuilt_ac["last_hidden"], torch.cat((last_a, last_c)))

        # Diverge inside A's final two-row segment. The first four-row segment
        # is reused; row 5 is read back from the immutable boundary segment;
        # only the divergent row is sourced from the current sidecar.
        tokens_fork = tokens_a[:5] + [70]
        aux_fork, last_fork, own_fork = _write_protocol_rows(
            sidecar, [11], [70], seed=106
        )
        assert _publish_protocol_sample(
            sink,
            sidecar,
            "fork",
            tokens_fork,
            slots_a[:5] + [11],
            own_fork,
        )
        rebuilt_fork = read_v6_sample(store, store_id=store_id, sample_id="fork")
        assert torch.equal(rebuilt_fork["aux"], torch.cat((aux_a[:5], aux_fork)))
        assert torch.equal(
            rebuilt_fork["last_hidden"], torch.cat((last_a[:5], last_fork))
        )

        # Dropping a local read and removing A's sample-scoped objects never
        # removes shared segments. Descendants remain independently readable.
        del rebuilt_ab, rebuilt_ac, rebuilt_fork
        for key in (
            f"{store_id}/_samples/a/meta",
            f"{store_id}/_samples/a/input_ids",
        ):
            store.remove(key, force=True)
            assert int(store.is_exist(key)) == 0
        assert (
            read_v6_sample(store, store_id=store_id, sample_id="ab")[
                "input_ids"
            ].tolist()
            == tokens_ab
        )
        assert (
            read_v6_sample(store, store_id=store_id, sample_id="ac")[
                "input_ids"
            ].tolist()
            == tokens_ac
        )

        # Missing any component is a whole-sample miss, never a partial read.
        ab_meta = json.loads(bytes(store.get(f"{store_id}/_samples/ab/meta")))
        unique = ab_meta["segments"][-1]["segment_id"]
        missing_key = f"{store_id}/_segments/epoch-a/{unique}/aux"
        store.remove(missing_key, force=True)
        assert int(store.is_exist(missing_key)) == 0
        try:
            read_v6_sample(store, store_id=store_id, sample_id="ab", timeout_s=1)
        except AssertionError:
            pass
        else:
            raise AssertionError("missing segment component returned partial sample")

        snap = stats.snapshot()
        assert snap["prefix_reused_rows_ct"] == 16
        assert snap["prefix_boundary_republished_rows_ct"] == 1
        # Includes A's initial six rows plus AB/AC/fork suffixes (3 + 2 + 1).
        assert snap["suffix_published_rows_ct"] == 12
    finally:
        sink.close()

    # A new writer epoch starts with an empty local prefix index, while the
    # old committed sample remains readable and the manifest tail advances.
    restart_id = "e2e_restart"
    first_stats = HiddenCaptureStats()
    # Keep the remote store client alive across writer objects. This isolates
    # the restart contract (new epoch/index, no remove) from local replica=1
    # client-unmount lifetime: old samples are guaranteed only while their
    # Mooncake objects still exist.
    first = _protocol_sink(
        restart_id, master_address, first_stats, "epoch-one", store=store
    )
    first.write_fingerprint({"model_path": "restart", "revision": "v6"})
    first_sidecar = _protocol_sidecar(first_stats)
    tokens_old = [11, 12, 13, 14]
    aux_old, last_old, own_old = _write_protocol_rows(
        first_sidecar, [0, 1, 2, 3], tokens_old, seed=104
    )
    assert _publish_protocol_sample(
        first, first_sidecar, "old", tokens_old, [0, 1, 2, 3], own_old
    )
    first.close()

    second_stats = HiddenCaptureStats()
    second = _protocol_sink(
        restart_id, master_address, second_stats, "epoch-two", store=store
    )
    second.write_fingerprint({"model_path": "restart", "revision": "v6"})
    second_sidecar = _protocol_sidecar(second_stats)
    try:
        tokens_new = [11, 12, 13, 14, 15]
        aux_new, last_new, own_new = _write_protocol_rows(
            second_sidecar, [4, 5, 6, 7, 8], tokens_new, seed=105
        )
        assert _publish_protocol_sample(
            second,
            second_sidecar,
            "new",
            tokens_new,
            [4, 5, 6, 7, 8],
            own_new,
        )
        old = read_v6_sample(store, store_id=restart_id, sample_id="old")
        assert torch.equal(old["aux"], aux_old)
        assert torch.equal(old["last_hidden"], last_old)
        new = read_v6_sample(store, store_id=restart_id, sample_id="new")
        assert torch.equal(new["aux"], aux_new)
        assert torch.equal(new["last_hidden"], last_new)
        assert second_stats.prefix_reused_rows_ct == 0
        assert bytes(store.get(f"{restart_id}/_seq/0/0")) == b"old"
        assert bytes(store.get(f"{restart_id}/_seq/0/1")) == b"new"
    finally:
        second.close()

    return {
        "reused_rows": stats.prefix_reused_rows_ct,
        "boundary_rows": stats.prefix_boundary_republished_rows_ct,
        "suffix_rows": stats.suffix_published_rows_ct,
        "restart_reused_rows": second_stats.prefix_reused_rows_ct,
    }


def _fault_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fault_gpu_process_readback():
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


def _fault_gpu_pids(readback):
    result = set()
    for line in readback:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[1].isdigit():
            result.add(int(fields[1]))
    return result


def _fault_wait_gpu_settle(baseline, timeout_s=30.0):
    baseline_pids = _fault_gpu_pids(baseline)
    history = []
    quiet = 0
    started = time.monotonic()
    while True:
        current = _fault_gpu_process_readback()
        introduced = sorted(_fault_gpu_pids(current) - baseline_pids)
        history.append(
            {
                "elapsed_s": time.monotonic() - started,
                "introduced_pids": introduced,
                "readback": current,
            }
        )
        quiet = quiet + 1 if not introduced else 0
        if quiet >= 2 or time.monotonic() - started >= timeout_s:
            return {
                "baseline": baseline,
                "final": current,
                "introduced_final": introduced,
                "settled": quiet >= 2,
                "settle_s": time.monotonic() - started,
                "history": history,
            }
        time.sleep(1)


def _fault_log_lineage(artifact_dir, scenario):
    result = {}
    for name in sorted(os.listdir(artifact_dir)):
        if not name.startswith(scenario + "_"):
            continue
        path = os.path.join(artifact_dir, name)
        if os.path.isfile(path):
            result[name] = {"path": path, "sha256": _fault_sha256(path)}
    return result


def _fault_server_args():
    return [
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
        "0.55",
        "--page-size",
        "1",
        "--random-seed",
        "42",
        "--chunked-prefill-size",
        "512",
        "--cuda-graph-max-bs-decode",
        "16",
        "--enable-hidden-state-capture",
    ]


def _fault_capture_snapshots():
    response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info", timeout=15)
    response.raise_for_status()
    states = response.json().get("internal_states") or []
    return [
        state["hidden_capture"]
        for state in states
        if state.get("hidden_capture") is not None
    ]


def _fault_summed_stats(snapshots):
    result = {}
    for snapshot in snapshots:
        for name, value in snapshot["stats"].items():
            result[name] = result.get(name, 0) + int(value)
    return result


def _fault_quiescent(snapshots):
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
        # V7 async proxy nests the real sink's state under "delegate".
        sink_state = sink.get("delegate") or sink
        if int(sink_state.get("ready_tasks", 0)):
            return False
        if any(
            state_name != "FREE" and count
            for state_name, count in (sink_state.get("lane_states") or {}).items()
        ):
            return False
    return True


def _fault_drain(timeout_s=90.0):
    started = time.monotonic()
    history = []
    quiet = 0
    previous_exports = None
    while time.monotonic() - started < timeout_s:
        snapshots = _fault_capture_snapshots()
        exports = _fault_summed_stats(snapshots).get("export_ok_ct", 0)
        idle = _fault_quiescent(snapshots)
        history.append(
            {
                "elapsed_s": time.monotonic() - started,
                "export_ok_ct": exports,
                "quiescent": idle,
                "snapshots": snapshots,
            }
        )
        quiet = quiet + 1 if idle and exports == previous_exports else 0
        if quiet >= 2:
            return snapshots, history, time.monotonic() - started, True
        previous_exports = exports
        time.sleep(1.0)
    snapshots = _fault_capture_snapshots()
    return snapshots, history, time.monotonic() - started, False


def _fault_generate(index, timeout_s=60):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "text": f"Fault-injection request {index}: explain bounded queues briefly.",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "index": index,
        "status_code": response.status_code,
        "output_tokens": payload["meta_info"]["completion_tokens"],
    }


def _fault_launch_server(master_address, store_id, stdout, stderr):
    return popen_launch_server(
        TARGET_MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_fault_server_args(),
        env={
            **os.environ,
            "SGLANG_RAGGED_VERIFY_MODE": "compact",
            "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
            "SGLANG_HIDDEN_CAPTURE_STORE_ID": store_id,
            "SGLANG_HIDDEN_CAPTURE_EXPORT_QUEUE_SIZE": "4",
            "MOONCAKE_MASTER": master_address,
            "MOONCAKE_PROTOCOL": "tcp",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "8gb",
        },
        return_stdout_stderr=(stdout, stderr),
    )


def _run_fault_unavailable(artifact_dir):
    gpu_baseline = _fault_gpu_process_readback()
    stdout_path = os.path.join(artifact_dir, "unavailable_server.log")
    stderr_path = os.path.join(artifact_dir, "unavailable_server.err")
    server = None
    generated = None
    snapshots = None
    with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
        try:
            server = _fault_launch_server(
                "127.0.0.1:50999", "fault_unavailable", stdout, stderr
            )
            generated = _fault_generate(0)
            snapshots = _fault_capture_snapshots()
            if snapshots:
                raise AssertionError("capture remained active without a master")
        finally:
            if server is not None:
                terminate_and_kill_process_tree(
                    server, terminate_timeout=60, wait_timeout=60
                )
    return {
        "serving_result": generated,
        "capture_snapshots": snapshots,
        "server_exit_code": None if server is None else server.poll(),
        "gpu_cleanup": _fault_wait_gpu_settle(gpu_baseline),
        "logs": {
            "stdout": {"path": stdout_path, "sha256": _fault_sha256(stdout_path)},
            "stderr": {"path": stderr_path, "sha256": _fault_sha256(stderr_path)},
        },
    }


def _run_fault_stalled(artifact_dir):
    gpu_baseline = _fault_gpu_process_readback()
    paths = {
        name: os.path.join(artifact_dir, f"stalled_{name}.log")
        for name in ("master_stdout", "master_stderr", "server_stdout", "server_stderr")
    }
    master_stopped = False
    master = None
    server = None
    serving_results = []
    before = paused = after = None
    warm_history = recovery_history = []
    warm_drain_s = recovery_drain_s = None
    with (
        open(paths["master_stdout"], "w") as master_stdout,
        open(paths["master_stderr"], "w") as master_stderr,
        open(paths["server_stdout"], "w") as server_stdout,
        open(paths["server_stderr"], "w") as server_stderr,
    ):
        try:
            master = subprocess.Popen(
                [MASTER_BIN, "--port", "50059", "--metrics_port", "9209"],
                stdout=master_stdout,
                stderr=master_stderr,
            )
            time.sleep(2)
            server = _fault_launch_server(
                "127.0.0.1:50059", "fault_stalled", server_stdout, server_stderr
            )
            for index in range(4):
                _fault_generate(-index - 1)
            before, warm_history, warm_drain_s, warm_drained = _fault_drain()
            if not warm_drained:
                raise AssertionError("warmup did not drain before fault")

            os.kill(master.pid, signal.SIGSTOP)
            master_stopped = True
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_fault_generate, index) for index in range(32)]
                serving_results = [future.result() for future in as_completed(futures)]
            paused = _fault_capture_snapshots()
            paused_stats = _fault_summed_stats(paused)
            if len(serving_results) != 32:
                raise AssertionError("serving requests were lost during sink stall")
            if paused_stats.get("export_queue_full_miss_ct", 0) <= 0:
                raise AssertionError("bounded export queue never failed closed")
            if not any(
                snapshot["state"]["export_worker_active"] for snapshot in paused
            ):
                raise AssertionError("fault did not hold a sink call in flight")

            os.kill(master.pid, signal.SIGCONT)
            master_stopped = False
            after, recovery_history, recovery_drain_s, recovered = _fault_drain()
            if not recovered:
                raise AssertionError("capture did not drain after master recovery")
        finally:
            if master_stopped and master is not None and master.poll() is None:
                os.kill(master.pid, signal.SIGCONT)
            if server is not None:
                terminate_and_kill_process_tree(
                    server, terminate_timeout=60, wait_timeout=60
                )
            if master is not None and master.poll() is None:
                master.terminate()
                try:
                    master.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait(timeout=30)

    return {
        "serving_results": sorted(serving_results, key=lambda item: item["index"]),
        "warm_state": before,
        "paused_state": paused,
        "recovered_state": after,
        "warm_drain_history": warm_history,
        "warm_drain_s": warm_drain_s,
        "recovery_history": recovery_history,
        "recovery_drain_s": recovery_drain_s,
        "server_exit_code": None if server is None else server.poll(),
        "master_exit_code": None if master is None else master.poll(),
        "gpu_cleanup": _fault_wait_gpu_settle(gpu_baseline),
        "logs": {
            name: {"path": path, "sha256": _fault_sha256(path)}
            for name, path in paths.items()
        },
    }


def _run_fault_characterization(scenario, out_path, artifact_dir):
    if not out_path or not artifact_dir:
        raise ValueError("fault mode requires --fault-out and --fault-artifact-dir")
    os.makedirs(artifact_dir, exist_ok=True)
    result = {
        "scenario": scenario,
        "master_binary": MASTER_BIN,
        "master_binary_sha256": _fault_sha256(MASTER_BIN),
        "attempts": {},
        "status": "ok",
    }
    failures = []
    scenarios = []
    if scenario in ("all", "unavailable"):
        scenarios.append(("unavailable", _run_fault_unavailable))
    if scenario in ("all", "stalled"):
        scenarios.append(("stalled", _run_fault_stalled))
    for name, runner in scenarios:
        scenario_gpu_baseline = _fault_gpu_process_readback()
        try:
            attempt = runner(artifact_dir)
            attempt["status"] = "ok"
        except BaseException as error:
            attempt = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "logs": _fault_log_lineage(artifact_dir, name),
                "gpu_readback_after_failure": _fault_gpu_process_readback(),
            }
            failures.append(f"{name}: {type(error).__name__}: {error}")
        attempt["outer_gpu_cleanup"] = _fault_wait_gpu_settle(scenario_gpu_baseline)
        result["attempts"][name] = attempt
    if failures:
        result["status"] = "error"
        result["failures"] = failures
    with open(out_path, "w") as artifact:
        json.dump(result, artifact, sort_keys=True)
    if failures:
        raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="run real-Mooncake prefix/fork/restart checks without a GPU server",
    )
    parser.add_argument(
        "--fault-scenario",
        choices=["all", "unavailable", "stalled"],
        help="run bounded serving fault characterization instead of E2E",
    )
    parser.add_argument("--fault-out")
    parser.add_argument("--fault-artifact-dir")
    args = parser.parse_args()
    if args.fault_scenario:
        _run_fault_characterization(
            args.fault_scenario,
            args.fault_out,
            args.fault_artifact_dir,
        )
        return
    # The protocol-only sinks run in this process rather than inheriting the
    # server subprocess environment. Keep them on the same local TCP
    # transport; this characterization does not claim RDMA/NUMA behavior.
    os.environ["MOONCAKE_PROTOCOL"] = "tcp"
    os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = "8gb"
    master = subprocess.Popen(
        [
            MASTER_BIN if os.path.exists(MASTER_BIN) else "mooncake_master",
            "--port",
            str(MASTER_PORT),
            "--metrics_port",
            "9007",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    server = None
    store = None
    try:
        if args.protocol_only:
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
            assert rc == 0, f"consumer setup failed rc={rc}"
            protocol = _run_real_prefix_protocol_checks(
                store, f"127.0.0.1:{MASTER_PORT}"
            )
            print(f"Protocol-only E2E OK: {protocol}")
            return

        server = popen_launch_server(
            TARGET_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "fa3",
                "--speculative-draft-attention-backend",
                "fa3",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--enable-hidden-state-capture",
            ],
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": STORE_ID,
                "MOONCAKE_MASTER": f"127.0.0.1:{MASTER_PORT}",
                "MOONCAKE_PROTOCOL": "tcp",
            },
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        prompt = "The capital of France is Paris. The capital of Germany is"
        input_ids = tokenizer(prompt).input_ids
        resp = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        resp.raise_for_status()
        rid = resp.json()["meta_info"]["id"]
        print(f"request done, rid={rid}")

        # Consumer-side client (same store, SpecForge-style zero-copy reads).
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
        assert rc == 0, f"consumer setup failed rc={rc}"

        sample = read_v6_sample(store, store_id=STORE_ID, sample_id=rid)
        meta = sample["meta"]
        # Coverage = prompt + verify-committed decode rows (final sampled
        # token has no hidden row).
        assert meta["num_rows"] >= len(input_ids), meta
        assert meta["rid"] == rid

        out = sample["input_ids"]
        assert out[: len(input_ids)].tolist() == input_ids, "prompt input_ids mismatch"

        aux = sample["aux"]
        nb = aux.numel() * aux.element_size()
        assert not torch.isnan(aux.float()).any()
        assert aux.abs().sum() > 0, "aux tensor is all zeros"

        fp = json.loads(bytes(store.get(f"{STORE_ID}/_fingerprint")))
        assert fp["coverage"] == "prefill_and_verify_commit"
        protocol = _run_real_prefix_protocol_checks(store, f"127.0.0.1:{MASTER_PORT}")
        print(
            f"E2E OK: V6 sample view + input_ids + aux ({nb / 1e6:.1f} MB) "
            "reconstructed via registered batch_get_into; fingerprint present; "
            f"protocol={protocol}"
        )
    finally:
        if server is not None:
            terminate_and_kill_process_tree(
                server, terminate_timeout=60, wait_timeout=60
            )
        if store is not None:
            store.close()
        master.terminate()
        try:
            master.wait(timeout=30)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait(timeout=30)


if __name__ == "__main__":
    main()
