"""E2E: DSpark server with SGLANG_HIDDEN_CAPTURE_SINK=mooncake against a local
mooncake_master; consume exported samples with SpecForge-style zero-copy reads.

Manual (not CI-registered): requires the mooncake binaries and a free master
port. Run:
    PYTHONPATH=python SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/test_hidden_capture_mooncake_e2e.py
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
MASTER_PORT = 50052
STORE_ID = "e2e_capture"
# The pip wrapper chmods a root-owned path and fails for non-root; call the
# packaged binary directly.
MASTER_BIN = "/usr/local/lib/python3.12/dist-packages/mooncake/mooncake_master"


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
    try:
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

        meta_key = f"{STORE_ID}/{rid}/g0/meta"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and int(store.is_exist(meta_key)) != 1:
            time.sleep(0.5)
        assert int(store.is_exist(meta_key)) == 1, "meta key never appeared"

        meta = json.loads(bytes(store.get(meta_key)))
        assert meta["num_tokens"] == len(input_ids), meta
        assert meta["rid"] == rid

        spec = meta["tensors"]["input_ids"]
        out = torch.empty(
            [int(x) for x in spec["shape"]],
            dtype=getattr(torch, spec["dtype"].split(".")[-1]),
        )
        nb = out.numel() * out.element_size()
        store.register_buffer(out.data_ptr(), nb)
        n = store.get_into(f"{STORE_ID}/{rid}/g0/input_ids", out.data_ptr(), nb)
        store.unregister_buffer(out.data_ptr())
        assert n == nb
        assert out.tolist() == input_ids, "input_ids mismatch"

        aux_spec = meta["tensors"]["aux"]
        aux = torch.empty(
            [int(x) for x in aux_spec["shape"]],
            dtype=getattr(torch, aux_spec["dtype"].split(".")[-1]),
        )
        nb = aux.numel() * aux.element_size()
        store.register_buffer(aux.data_ptr(), nb)
        n = store.get_into(f"{STORE_ID}/{rid}/g0/aux", aux.data_ptr(), nb)
        store.unregister_buffer(aux.data_ptr())
        assert n == nb
        assert not torch.isnan(aux.float()).any()
        assert aux.abs().sum() > 0, "aux tensor is all zeros"

        fp = json.loads(bytes(store.get(f"{STORE_ID}/_fingerprint")))
        assert fp["coverage"] == "prefill_only"
        print(
            f"E2E OK: meta + input_ids + aux ({nb / 1e6:.1f} MB) consumed "
            "via zero-copy get_into; fingerprint present"
        )
    finally:
        kill_process_tree(server.pid)
        master.terminate()


if __name__ == "__main__":
    main()
