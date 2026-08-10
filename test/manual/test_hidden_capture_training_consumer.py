"""Exact training-consumer artifact check for hidden-state capture exports.

Consumes real server exports through BOTH trainer-side paths and asserts
byte-exact fidelity:

1. File sink -> SpecForge ``OfflineEagle3Dataset.process_data`` (the actual
   offline training loader), asserting the loader's outputs are bitwise views
   of the exported tensors and the shapes/dtypes match the training contract.
2. Mooncake sink -> the ``MooncakeFeatureStore``-style registered ``get_into``
   read (the actual online consumption primitive), asserting byte-exact
   roundtrip of every key against the file-sink export of the SAME prompt
   from the same server process (cross-sink consistency).

Manual (not CI-registered): needs a GPU, the DSpark 8B pair, the local
SpecForge checkout, and the mooncake binaries.

Run:
    PYTHONPATH=python:/workspaces/SpecForge SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        python3 test/manual/test_hidden_capture_training_consumer.py
"""

import json
import os
import subprocess
import tempfile
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


def _launch(sink: str, capture_dir: str):
    env = {
        **os.environ,
        "SGLANG_RAGGED_VERIFY_MODE": "compact",
        "SGLANG_HIDDEN_CAPTURE_SINK": sink,
        "SGLANG_HIDDEN_CAPTURE_DIR": capture_dir,
        "SGLANG_HIDDEN_CAPTURE_STORE_ID": STORE_ID,
        "MOONCAKE_MASTER": f"127.0.0.1:{MASTER_PORT}",
        "MOONCAKE_PROTOCOL": "tcp",
    }
    return popen_launch_server(
        TARGET_MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_server_args(),
        env=env,
    )


def _generate(input_ids):
    resp = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        },
    )
    resp.raise_for_status()
    return resp.json()


def _wait_file(capture_dir, rid, timeout_s=30):
    path = os.path.join(capture_dir, f"{rid}.ckpt")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return torch.load(path, weights_only=True)
        time.sleep(0.2)
    raise AssertionError("file export missing")


def check_offline_loader(record, input_ids):
    """Path 1: the actual SpecForge offline training loader."""
    from specforge.data.preprocessing import OfflineEagle3Dataset

    out = OfflineEagle3Dataset.process_data(record, max_len=32768)
    T = record["input_ids"].shape[0]
    # Loader contract: hidden_state = draft input (our packed aux),
    # target = training target (our post-norm last), batch dim restored.
    assert out["hidden_state"].shape == (1, T, record["aux_hidden_state"].shape[-1])
    assert out["target"].shape == (1, T, record["hidden_state"].shape[-1])
    assert out["input_ids"].shape == (1, T)
    # Byte-exact: the loader must not have transformed the payload.
    assert torch.equal(out["hidden_state"][0], record["aux_hidden_state"][0])
    assert torch.equal(out["target"][0], record["hidden_state"][0])
    assert out["input_ids"][0].tolist()[: len(input_ids)] == input_ids
    # Loader zeroes the final loss position; everything else is our mask.
    assert out["loss_mask"][0, -1].item() == 0
    assert out["loss_mask"][0, :-1].all()
    print(f"offline loader (OfflineEagle3Dataset): byte-exact OK, T={T}")


def check_mooncake_consumer(rid, file_record):
    """Path 2: MooncakeFeatureStore-style registered get_into, cross-checked
    byte-exact against the file-sink export of the same request."""
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
    assert int(store.is_exist(meta_key)) == 1, "mooncake meta missing"
    meta = json.loads(bytes(store.get(meta_key)))

    key_to_record = {
        "aux": "aux_hidden_state",
        "last_hidden": "hidden_state",
        "input_ids": "input_ids",
    }
    for name, record_name in key_to_record.items():
        spec = meta["tensors"][name]
        out = torch.empty(
            [int(x) for x in spec["shape"]],
            dtype=getattr(torch, spec["dtype"].split(".")[-1]),
        )
        nb = out.numel() * out.element_size()
        store.register_buffer(out.data_ptr(), nb)
        n = store.get_into(f"{STORE_ID}/{rid}/g0/{name}", out.data_ptr(), nb)
        store.unregister_buffer(out.data_ptr())
        assert n == nb, (name, n, nb)
        ref = file_record[record_name]
        assert out.shape == ref.shape, (name, out.shape, ref.shape)
        assert out.dtype == ref.dtype
        assert torch.equal(out, ref), f"{name}: mooncake bytes != file bytes"
        print(f"mooncake consumer: {name} byte-exact vs file sink ({nb / 1e6:.1f} MB)")
    store.close()


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

    # Run 1: file sink -> offline loader check.
    capture_dir = tempfile.mkdtemp(prefix="consumer_check_file_")
    server = _launch("file", capture_dir)
    try:
        result = _generate(input_ids)
        rid_file = result["meta_info"]["id"]
        file_record = _wait_file(capture_dir, rid_file)
        expected_rows = len(input_ids) + MAX_NEW_TOKENS - 1
        assert file_record["input_ids"].shape[0] == expected_rows, (
            file_record["input_ids"].shape,
            expected_rows,
        )
        check_offline_loader(file_record, input_ids)
    finally:
        kill_process_tree(server.pid)

    # Run 2: mooncake sink, same prompt, greedy -> identical forwarded rows
    # (cold cache both times), cross-checked byte-exact against run 1.
    server = _launch("mooncake", capture_dir)
    try:
        result = _generate(input_ids)
        rid_mc = result["meta_info"]["id"]
        check_mooncake_consumer(rid_mc, file_record)
    finally:
        kill_process_tree(server.pid)
        master.terminate()
    print("training-consumer artifact check: ALL OK")


if __name__ == "__main__":
    main()
