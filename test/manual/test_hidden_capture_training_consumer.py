"""Exact training-consumer check for hidden-state capture (Mooncake-only).

Acceptance path per project decision: the trainer consumes exclusively from
Mooncake. A real DSpark server exports via the mooncake sink; this script
plays the trainer — ``MooncakeFeatureStore``-style registered ``get_into``
reads — and validates the artifact against independent references:

- discovery is manifest-tail only (the SpecLoop ingest contract): the
  consumer knows nothing but the store prefix, tails ``_seq/{w}/{n}`` for
  writer streams and sequence numbers, and finds every exported sample_id
  with no rid or response channel involved;
- sample views plus immutable-segment metadata are self-describing and
  sufficient to allocate every receive buffer without any response channel;
- ``input_ids`` readback == prompt + generated tokens (final sampled token
  excluded, which has no hidden row);
- decode-region ``last_hidden`` rows satisfy the exact norm contract: row t
  through the LM head argmaxes to greedy token t+1 (validates fence, commit
  selection, and row alignment simultaneously);
- store-level ``_fingerprint`` present with the capture contract fields.

The file sink remains an M1 debug path and is intentionally NOT exercised
here.

Manual (not CI-registered): needs a GPU, the DSpark 8B pair, and the
mooncake binaries.

Run:
    PYTHONPATH=python SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/test_hidden_capture_training_consumer.py
"""

import json
import os
import subprocess
import time

import requests
import torch
from hidden_capture_v6_reader import read_v6_sample

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
MASTER_PORT = 50054
STORE_ID = "consumer_check"


def _default_mooncake_master():
    try:
        import mooncake

        return os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master")
    except Exception:
        return "mooncake_master"


MASTER_BIN = os.environ.get("MOONCAKE_MASTER_BIN") or _default_mooncake_master()

PROMPT = "The three primary colors are red, blue, and"
MAX_NEW_TOKENS = 12


def _server_args():
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
        "--cuda-graph-max-bs-decode",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "1",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enable-hidden-state-capture",
    ]


def _tail_manifest(store, until_id: str, deadline_s: float = 30.0) -> list[str]:
    """The SpecLoop ingester's discovery loop: knowing only the store prefix,
    tail ``_seq/{writer}/{n}`` — writers found by probing ``_seq/{w}/0`` with
    w increasing until a miss, entries by n increasing with gap tolerance
    (a missing n with n+1 present is an evicted/burnt hole: count, skip).
    Polls until ``until_id`` shows up (exports are asynchronous, and the
    server's own warmup request exports too, so the stream is live)."""
    deadline = time.monotonic() + deadline_s
    while True:
        sample_ids: list[str] = []
        writer = 0
        while int(store.is_exist(f"{STORE_ID}/_seq/{writer}/0")) == 1:
            n, holes = 0, 0
            while True:
                key = f"{STORE_ID}/_seq/{writer}/{n}"
                if int(store.is_exist(key)) == 1:
                    sample_ids.append(bytes(store.get(key)).decode())
                    n += 1
                    holes = 0
                elif holes < 8:  # gap detection: probe past the hole
                    n += 1
                    holes += 1
                else:
                    break
            writer += 1
        if until_id in sample_ids or time.monotonic() > deadline:
            return sample_ids
        time.sleep(0.2)


def _load_lm_head() -> torch.Tensor:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    idx_path = hf_hub_download(TARGET_MODEL, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard = hf_hub_download(TARGET_MODEL, weight_map["lm_head.weight"])
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor("lm_head.weight").float()


def main():
    master = subprocess.Popen(
        [
            MASTER_BIN if os.path.exists(MASTER_BIN) else "mooncake_master",
            "--port",
            str(MASTER_PORT),
            "--metrics_port",
            "9009",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tokenizer(PROMPT).input_ids

    server = None
    store = None
    try:
        server = popen_launch_server(
            TARGET_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_server_args(),
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": STORE_ID,
                "MOONCAKE_MASTER": f"127.0.0.1:{MASTER_PORT}",
                "MOONCAKE_PROTOCOL": "tcp",
            },
        )
        resp = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            },
        )
        resp.raise_for_status()
        result = resp.json()
        rid = result["meta_info"]["id"]
        completion_ids = result["output_ids"]

        # ---- trainer side ----
        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        rc = store.setup(
            "localhost",
            "P2PHANDSHAKE",
            2 * 1024**3,
            128 * 1024**2,
            "tcp",
            "",
            f"127.0.0.1:{MASTER_PORT}",
        )
        assert rc == 0

        # ---- discovery: manifest tail, zero rid knowledge ----
        # The stream also carries the server's warmup-request export; the
        # measured request must be discoverable among them. rid is a uuid
        # (key-safe), so sample_id passes through unhashed.
        discovered = _tail_manifest(store, until_id=rid)
        assert rid in discovered, f"manifest tail never discovered {rid}: {discovered}"
        sample_id = rid

        meta_key = f"{STORE_ID}/_samples/{sample_id}/meta"
        assert (
            int(store.is_exist(meta_key)) == 1
        ), "manifest entry visible before its meta: ordering contract broken"
        sample = read_v6_sample(store, store_id=STORE_ID, sample_id=sample_id)
        meta = sample["meta"]
        assert meta["rid"] == rid

        # Coverage: prompt + all generated tokens except the final one.
        expected_rows = len(input_ids) + MAX_NEW_TOKENS - 1
        assert meta["num_rows"] == expected_rows, (meta["num_rows"], expected_rows)

        ids = sample["input_ids"]
        assert (
            ids.tolist() == (input_ids + completion_ids)[:expected_rows]
        ), "consumed input_ids != prompt + generated tokens"

        aux = sample["aux"]
        last = sample["last_hidden"]
        assert not torch.isnan(aux.float()).any()
        assert aux.shape[0] == expected_rows and last.shape[0] == expected_rows

        # Exact norm contract on consumed decode rows: row t -> greedy t+1.
        lm_head = _load_lm_head()
        prompt_len = len(input_ids)
        for t in range(len(completion_ids) - 1):
            row = last[prompt_len + t].float()
            pred = int((row @ lm_head.T).argmax())
            assert pred == completion_ids[t + 1], (
                f"norm contract violated at consumed decode row {t}: "
                f"{pred} != {completion_ids[t + 1]}"
            )

        fp = json.loads(bytes(store.get(f"{STORE_ID}/_fingerprint")))
        assert fp["coverage"] == "prefill_and_verify_commit"
        assert fp["norm_contract"] == "post_final_norm_pre_lm_head"
        print(
            f"Mooncake V6 consumer check ALL OK: {expected_rows} rows rebuilt via "
            f"registered batch_get_into; input_ids exact; norm contract exact on "
            f"{len(completion_ids) - 1} decode rows"
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
