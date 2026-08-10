"""Exact training-consumer check for hidden-state capture (Mooncake-only).

Acceptance path per project decision: the trainer consumes exclusively from
Mooncake. A real DSpark server exports via the mooncake sink; this script
plays the trainer — ``MooncakeFeatureStore``-style registered ``get_into``
reads — and validates the artifact against independent references:

- meta is self-describing (shape/dtype/rid/num_tokens) and sufficient to
  allocate every receive buffer without any response channel;
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

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"
MASTER_PORT = 50054
STORE_ID = "consumer_check"
MASTER_BIN = "/usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master"

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


def _consume_tensor(store, key: str, spec: dict) -> torch.Tensor:
    """The trainer's actual read primitive: registered zero-copy get_into."""
    out = torch.empty(
        [int(x) for x in spec["shape"]],
        dtype=getattr(torch, spec["dtype"].split(".")[-1]),
    )
    nb = out.numel() * out.element_size()
    store.register_buffer(out.data_ptr(), nb)
    n = store.get_into(key, out.data_ptr(), nb)
    store.unregister_buffer(out.data_ptr())
    assert n == nb, (key, n, nb)
    return out


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
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    input_ids = tokenizer(PROMPT).input_ids

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
    try:
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

        meta_key = f"{STORE_ID}/{rid}/g0/meta"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and int(store.is_exist(meta_key)) != 1:
            time.sleep(0.2)
        assert int(store.is_exist(meta_key)) == 1, "meta key never appeared"
        meta = json.loads(bytes(store.get(meta_key)))
        assert meta["rid"] == rid
        assert meta["loss_mask"] == "all_ones_placeholder"

        # Coverage: prompt + all generated tokens except the final one.
        expected_rows = len(input_ids) + MAX_NEW_TOKENS - 1
        assert meta["num_tokens"] == expected_rows, (meta["num_tokens"], expected_rows)

        ids = _consume_tensor(
            store, f"{STORE_ID}/{rid}/g0/input_ids", meta["tensors"]["input_ids"]
        )
        assert (
            ids.tolist() == (input_ids + completion_ids)[:expected_rows]
        ), "consumed input_ids != prompt + generated tokens"

        aux = _consume_tensor(store, f"{STORE_ID}/{rid}/g0/aux", meta["tensors"]["aux"])
        last = _consume_tensor(
            store, f"{STORE_ID}/{rid}/g0/last_hidden", meta["tensors"]["last_hidden"]
        )
        assert not torch.isnan(aux.float()).any()
        assert aux.shape[1] == expected_rows and last.shape[1] == expected_rows

        # Exact norm contract on consumed decode rows: row t -> greedy t+1.
        lm_head = _load_lm_head()
        prompt_len = len(input_ids)
        for t in range(len(completion_ids) - 1):
            row = last[0, prompt_len + t].float()
            pred = int((row @ lm_head.T).argmax())
            assert pred == completion_ids[t + 1], (
                f"norm contract violated at consumed decode row {t}: "
                f"{pred} != {completion_ids[t + 1]}"
            )

        fp = json.loads(bytes(store.get(f"{STORE_ID}/_fingerprint")))
        assert fp["coverage"] == "prefill_and_verify_commit"
        assert fp["norm_contract"] == "post_final_norm_pre_lm_head"
        store.close()
        print(
            f"Mooncake consumer check ALL OK: {expected_rows} rows consumed via "
            f"registered get_into; input_ids exact; norm contract exact on "
            f"{len(completion_ids) - 1} decode rows"
        )
    finally:
        kill_process_tree(server.pid)
        master.terminate()


if __name__ == "__main__":
    main()
