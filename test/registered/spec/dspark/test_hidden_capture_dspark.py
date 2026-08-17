"""E2E tests for online hidden-state capture under DSpark.

Validates the exported per-sample ``.ckpt`` files against independent
references:

- prompt-region ``aux_hidden_state`` / ``hidden_state`` rows must match a
  HuggingFace CPU forward (residual stream after target layer k; post-norm);
- decode-region rows (verify-committed capture) must cover every generated
  token except the final one, carry the generated token ids, and satisfy the
  norm contract (row t through the LM head predicts token t+1 exactly under
  greedy decoding);
- chunked prefill and warm-prefix reuse must yield complete, consistent
  samples (or a clean whole-sample miss — never partial data);
- an undersized staging ring must produce misses without breaking serving.
"""

import json
import os
import tempfile
import time
import unittest
from contextlib import contextmanager

import requests
import torch

from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-b", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_8b_block7"


def _default_mooncake_master():
    try:
        import mooncake

        return os.path.join(os.path.dirname(mooncake.__file__), "mooncake_master")
    except Exception:
        return "mooncake_master"


MOONCAKE_MASTER_BIN = (
    os.environ.get("MOONCAKE_MASTER_BIN") or _default_mooncake_master()
)

if is_sm100_supported():
    ATTENTION_BACKEND = "trtllm_mha"
    DRAFT_ATTENTION_BACKEND = "fa4"
else:
    ATTENTION_BACKEND = "fa3"
    DRAFT_ATTENTION_BACKEND = "fa3"

CHUNKED_PREFILL_SIZE = 128
_EXPORT_POLL_TIMEOUT_S = 30.0


def _dspark_server_args(prefill_graph: bool = True):
    args = [
        "--trust-remote-code",
        "--attention-backend",
        ATTENTION_BACKEND,
        "--speculative-draft-attention-backend",
        DRAFT_ATTENTION_BACKEND,
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
        "--chunked-prefill-size",
        str(CHUNKED_PREFILL_SIZE),
        "--enable-hidden-state-capture",
    ]
    if not prefill_graph:
        args += ["--cuda-graph-backend-prefill", "disabled"]
    return args


def _generate(base_url, input_ids, max_new_tokens=8):
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        },
    )
    response.raise_for_status()
    return response.json()


def _wait_for_ckpt(capture_dir, rid, timeout_s=_EXPORT_POLL_TIMEOUT_S):
    path = os.path.join(capture_dir, f"{rid}.ckpt")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return torch.load(path, weights_only=True)
        time.sleep(0.2)
    return None


@contextmanager
def _registered_tensors(store, *tensors):
    registered = []
    try:
        for tensor in tensors:
            nbytes = tensor.numel() * tensor.element_size()
            store.register_buffer(tensor.data_ptr(), nbytes)
            registered.append(tensor)
        yield
    finally:
        for tensor in reversed(registered):
            store.unregister_buffer(tensor.data_ptr())


def _torch_dtype(name):
    return getattr(torch, str(name).split(".")[-1])


def _consume_mooncake_sample(store, store_id, sample_id, timeout_s=30):
    """Read either the default V6 sample-view format or the legacy fallback.

    The V6 branch deliberately reconstructs one contiguous sample from
    immutable segment refs, mirroring the registered ``batch_get_into`` data
    path owned by SpecLoop without importing that separate repository here.
    """
    prefix_meta_key = f"{store_id}/_samples/{sample_id}/meta"
    legacy_meta_key = f"{store_id}/{sample_id}/g0/meta"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if int(store.is_exist(prefix_meta_key)) == 1:
            break
        if int(store.is_exist(legacy_meta_key)) == 1:
            break
        time.sleep(0.2)

    if int(store.is_exist(prefix_meta_key)) == 1:
        sample = json.loads(bytes(store.get(prefix_meta_key)))
        assert sample["schema"] == "hidden-sample-view-v1"
        assert sample["sample_id"] == sample_id
        num_rows = int(sample["num_rows"])
        refs = sample["segments"]
        assert refs and sum(int(ref["rows"]) for ref in refs) == num_rows

        segment_metas = []
        expected_offset = 0
        for ref in refs:
            assert int(ref["sample_offset"]) == expected_offset
            rows = int(ref["rows"])
            assert rows > 0
            expected_offset += rows
            base = (
                f"{store_id}/_segments/{sample['writer_epoch']}/" f"{ref['segment_id']}"
            )
            segment_meta = json.loads(bytes(store.get(f"{base}/meta")))
            assert segment_meta["schema"] == "hidden-segment-v1"
            assert int(segment_meta["num_rows"]) == rows
            segment_metas.append((ref, base, segment_meta))
        assert expected_offset == num_rows

        first_meta = segment_metas[0][2]
        aux_spec = first_meta["tensors"]["aux"]
        last_spec = first_meta["tensors"]["last_hidden"]
        input_ids = torch.empty(
            num_rows, dtype=_torch_dtype(sample["input_ids"]["dtype"])
        )
        aux = torch.empty(
            (num_rows, int(aux_spec["shape"][1])), dtype=_torch_dtype(aux_spec["dtype"])
        )
        last_hidden = torch.empty(
            (num_rows, int(last_spec["shape"][1])),
            dtype=_torch_dtype(last_spec["dtype"]),
        )

        keys = [sample["input_ids"]["key"]]
        pointers = [input_ids.data_ptr()]
        sizes = [input_ids.numel() * input_ids.element_size()]
        for ref, base, segment_meta in segment_metas:
            offset = int(ref["sample_offset"])
            rows = int(ref["rows"])
            assert segment_meta["tensors"]["aux"]["dtype"] == aux_spec["dtype"]
            assert segment_meta["tensors"]["last_hidden"]["dtype"] == last_spec["dtype"]
            aux_view = aux.narrow(0, offset, rows)
            last_view = last_hidden.narrow(0, offset, rows)
            keys.extend([f"{base}/aux", f"{base}/last_hidden"])
            pointers.extend([aux_view.data_ptr(), last_view.data_ptr()])
            sizes.extend(
                [
                    aux_view.numel() * aux_view.element_size(),
                    last_view.numel() * last_view.element_size(),
                ]
            )
        with _registered_tensors(store, input_ids, aux, last_hidden):
            actual = store.batch_get_into(keys, pointers, sizes)
        assert len(actual) == len(sizes)
        assert all(int(got) == want for got, want in zip(actual, sizes))
        return {
            "meta": {
                **sample,
                "num_tokens": num_rows,
            },
            "input_ids": input_ids,
            "aux": aux.unsqueeze(0),
            "last_hidden": last_hidden.unsqueeze(0),
        }

    assert int(store.is_exist(legacy_meta_key)) == 1, f"meta missing: {sample_id}"
    meta = json.loads(bytes(store.get(legacy_meta_key)))
    out = {"meta": meta}
    tensors = []
    keys = []
    pointers = []
    sizes = []
    for name in ("aux", "last_hidden", "input_ids"):
        spec = meta["tensors"][name]
        tensor = torch.empty(
            [int(value) for value in spec["shape"]], dtype=_torch_dtype(spec["dtype"])
        )
        tensors.append(tensor)
        keys.append(f"{store_id}/{sample_id}/g0/{name}")
        pointers.append(tensor.data_ptr())
        sizes.append(tensor.numel() * tensor.element_size())
        out[name] = tensor
    with _registered_tensors(store, *tensors):
        if callable(getattr(store, "batch_get_into", None)):
            actual = store.batch_get_into(keys, pointers, sizes)
        else:
            actual = [
                store.get_into(key, pointer, size)
                for key, pointer, size in zip(keys, pointers, sizes)
            ]
    assert all(int(got) == want for got, want in zip(actual, sizes))
    return out


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float()
    b = b.float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


class TestHiddenCaptureDSpark(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture_dir = tempfile.mkdtemp(prefix="hidden_capture_")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dspark_server_args(),
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_HIDDEN_CAPTURE_DIR": cls.capture_dir,
            },
        )
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def setUp(self):
        requests.post(self.base_url + "/flush_cache")

    def _fingerprint(self):
        with open(os.path.join(self.capture_dir, "_fingerprint.json")) as f:
            return json.load(f)

    def test_a_fingerprint_written(self):
        fp = self._fingerprint()
        self.assertEqual(fp["norm_contract"], "post_final_norm_pre_lm_head")
        self.assertEqual(fp["dtype"], "torch.bfloat16")
        self.assertGreater(len(fp["aux_layer_ids"]), 0)

    def test_b_export_schema_and_hf_parity(self):
        prompt = "The capital of France is Paris. The capital of Germany is"
        input_ids = self.tokenizer(prompt).input_ids
        result = _generate(self.base_url, input_ids)
        rid = result["meta_info"]["id"]
        record = _wait_for_ckpt(self.capture_dir, rid)
        self.assertIsNotNone(record, "export did not produce a .ckpt in time")

        T = len(input_ids)
        fp = self._fingerprint()
        num_aux = len(fp["aux_layer_ids"])
        hidden = fp["hidden_size"]
        # Coverage = prompt + verify-committed decode rows; the prompt region
        # is rows [0, prompt_len) and is what the HF reference below checks.
        total_rows = record["input_ids"].shape[0]
        self.assertGreaterEqual(total_rows, T)
        self.assertEqual(int(record["prompt_len"]), T)
        self.assertEqual(record["input_ids"][:T].tolist(), input_ids)
        self.assertEqual(record["loss_mask"].tolist(), [1] * total_rows)
        self.assertEqual(
            record["aux_hidden_state"].shape, (1, total_rows, num_aux * hidden)
        )
        self.assertEqual(record["hidden_state"].shape, (1, total_rows, hidden))
        self.assertEqual(record["aux_hidden_state"].dtype, torch.bfloat16)

        # HF reference forward on CPU (independent of every sglang kernel).
        from transformers import AutoModelForCausalLM

        torch.set_num_threads(os.cpu_count() or 8)
        hf_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL, torch_dtype=torch.bfloat16
        )
        with torch.inference_mode():
            hf_out = hf_model(torch.tensor([input_ids]), output_hidden_states=True)

        # aux layer id k = residual stream after target layer k
        # = HF hidden_states[k + 1]. Prompt region only (decode rows are
        # covered by tests f/g).
        aux = record["aux_hidden_state"][0, :T].view(T, num_aux, hidden)
        for j, layer_id in enumerate(fp["aux_layer_ids"]):
            ref = hf_out.hidden_states[layer_id + 1][0]
            cos = _row_cosine(aux[:, j, :], ref)
            self.assertGreater(
                float(cos.min()),
                0.98,
                f"aux layer {layer_id}: min row cosine {float(cos.min()):.4f}",
            )

        # target-last = post-final-norm = HF hidden_states[-1].
        last = record["hidden_state"][0, :T]
        cos = _row_cosine(last, hf_out.hidden_states[-1][0])
        self.assertGreater(float(cos.min()), 0.98)

        # Semantic check on the norm contract: the last prompt row through the
        # LM head must predict the server's first greedy output token.
        logits = last[-1].float() @ hf_model.lm_head.weight.float().T
        first_output_token = result["output_ids"][0] if "output_ids" in result else None
        if first_output_token is None:
            first_output_token = self.tokenizer(
                result["text"], add_special_tokens=False
            ).input_ids[0]
        self.assertEqual(int(logits.argmax()), first_output_token)

        del hf_model

    def test_c_chunked_prefill_full_coverage(self):
        # ~40x the chunk size cadence: several chunks per request.
        base = self.tokenizer("The quick brown fox jumps over the lazy dog. ").input_ids
        input_ids = (base * 40)[: CHUNKED_PREFILL_SIZE * 3 + 17]
        result = _generate(self.base_url, input_ids, max_new_tokens=4)
        rid = result["meta_info"]["id"]
        record = _wait_for_ckpt(self.capture_dir, rid)
        self.assertIsNotNone(record, "chunked-prefill export missing")
        n = len(input_ids)
        self.assertEqual(record["input_ids"][:n].tolist(), input_ids)
        self.assertGreaterEqual(record["aux_hidden_state"].shape[1], n)

    def test_d_warm_prefix_consistency(self):
        prefix = self.tokenizer(
            "In a quiet village nestled between rolling hills, an old clockmaker "
            "spent his days repairing timepieces that the townsfolk brought him. "
        ).input_ids
        ids_a = (
            prefix + self.tokenizer("His favorite was a brass pocket watch.").input_ids
        )
        result_a = _generate(self.base_url, ids_a, max_new_tokens=4)
        record_a = _wait_for_ckpt(self.capture_dir, result_a["meta_info"]["id"])
        self.assertIsNotNone(record_a)

        # B shares the prefix (radix hit -> B never forwards those rows).
        ids_b = prefix + self.tokenizer("One winter morning a stranger came.").input_ids
        result_b = _generate(self.base_url, ids_b, max_new_tokens=4)
        self.assertEqual(result_b["meta_info"]["finish_reason"]["type"], "length")
        record_b = _wait_for_ckpt(
            self.capture_dir, result_b["meta_info"]["id"], timeout_s=15
        )

        if record_b is None:
            return  # clean whole-sample miss is allowed; wrong data is not
        self.assertEqual(record_b["input_ids"][: len(ids_b)].tolist(), ids_b)
        n = len(prefix)
        self.assertTrue(
            torch.equal(
                record_b["aux_hidden_state"][0, :n], record_a["aux_hidden_state"][0, :n]
            ),
            "warm-prefix rows must be bitwise identical to the writer's rows",
        )
        self.assertTrue(
            torch.equal(
                record_b["hidden_state"][0, :n], record_a["hidden_state"][0, :n]
            )
        )

    def test_e_decode_survives_and_output_sane(self):
        input_ids = self.tokenizer("Count from one to ten: one, two,").input_ids
        result = _generate(self.base_url, input_ids, max_new_tokens=32)
        self.assertIn("three", result["text"])

    def test_f_verify_rows_cover_decode_tokens(self):
        """Verify-committed capture: the export covers prompt + all decode
        tokens except the final sampled one, and the decode-region rows carry
        the actually-generated token ids (the training contract: rows 1:1
        with forwarded tokens)."""
        input_ids = self.tokenizer("The first five prime numbers are").input_ids
        max_new_tokens = 24
        result = _generate(self.base_url, input_ids, max_new_tokens=max_new_tokens)
        self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
        rid = result["meta_info"]["id"]
        record = _wait_for_ckpt(self.capture_dir, rid)
        self.assertIsNotNone(record, "verify-coverage export missing")

        T = record["input_ids"].shape[0]
        prompt_len = len(input_ids)
        self.assertEqual(int(record["prompt_len"]), prompt_len)
        # Full coverage: prompt + (completion - 1) rows.
        self.assertEqual(T, prompt_len + max_new_tokens - 1)
        self.assertEqual(record["aux_hidden_state"].shape[1], T)
        self.assertEqual(record["hidden_state"].shape[1], T)
        # Decode-region input_ids are the generated tokens (minus the last).
        completion_ids = result["output_ids"]
        self.assertEqual(
            record["input_ids"][prompt_len:].tolist(),
            completion_ids[: max_new_tokens - 1],
        )
        # Hidden rows in the decode region are real data, not scatter fill.
        decode_aux = record["aux_hidden_state"][0, prompt_len:].float()
        self.assertGreater(float(decode_aux.abs().sum()), 0.0)
        row_norms = decode_aux.norm(dim=-1)
        self.assertTrue(bool((row_norms > 0).all()), "zero-filled decode row leaked")

    def test_g_verify_last_hidden_semantics(self):
        """Decode-region hidden_state rows must be post-final-norm: row t
        through the LM head predicts token t+1 of the actual generation."""
        input_ids = self.tokenizer("2, 4, 6, 8,").input_ids
        result = _generate(self.base_url, input_ids, max_new_tokens=16)
        rid = result["meta_info"]["id"]
        record = _wait_for_ckpt(self.capture_dir, rid)
        self.assertIsNotNone(record)
        prompt_len = len(input_ids)
        completion_ids = result["output_ids"]

        # lm_head only; avoids loading the full model twice in one suite.
        import json as _json

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        idx_path = hf_hub_download(TARGET_MODEL, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = _json.load(f)["weight_map"]
        head_name = "lm_head.weight"
        shard = hf_hub_download(TARGET_MODEL, weight_map[head_name])
        with safe_open(shard, framework="pt") as f:
            lm_head = f.get_tensor(head_name).float()

        # Row prompt_len + t predicts completion token t+1.
        correct = 0
        checked = 0
        for t in range(0, len(completion_ids) - 1):
            row = record["hidden_state"][0, prompt_len + t].float()
            pred = int((row @ lm_head.T).argmax())
            checked += 1
            correct += int(pred == completion_ids[t + 1])
        # Greedy decode + exact captured hidden => exact argmax match.
        self.assertEqual(correct, checked)


class TestHiddenCaptureGraphOnMooncake(CustomTestCase):
    """Acceptance-path tests with the prefill CUDA graph ON (default
    breakable) and the Mooncake sink: prefix cache, chunked prefill, and
    graph-overwrite safety, all consumed through registered ``get_into``.

    Under BCG the graph captures only the transformer body — the packed aux
    is a fresh eager tensor, and the post-norm last hidden (a view of the
    shared static body-output buffer) is cloned on the forward stream at
    capture time (same-stream ordering = the overwrite fence). Diagnostic
    evidence: same-prompt exports are bitwise identical run-to-run; a broken
    fence would corrupt rows nondeterministically."""

    MASTER_PORT = 50056
    STORE_ID = "graphon_acceptance"
    MASTER_BIN = MOONCAKE_MASTER_BIN

    @classmethod
    def setUpClass(cls):
        import subprocess

        cls.master = subprocess.Popen(
            [
                (
                    cls.MASTER_BIN
                    if os.path.exists(cls.MASTER_BIN)
                    else "mooncake_master"
                ),
                "--port",
                str(cls.MASTER_PORT),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Prefill graph ON (breakable default) — no disable flag.
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dspark_server_args(prefill_graph=True),
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": cls.STORE_ID,
                "MOONCAKE_MASTER": f"127.0.0.1:{cls.MASTER_PORT}",
                "MOONCAKE_PROTOCOL": "tcp",
            },
        )
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "master") and cls.master:
            cls.master.terminate()

    def setUp(self):
        requests.post(self.base_url + "/flush_cache")

    def _consumer(self):
        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        rc = store.setup(
            "localhost",
            "P2PHANDSHAKE",
            2 * 1024**3,
            128 * 1024**2,
            "tcp",
            "",
            f"127.0.0.1:{self.MASTER_PORT}",
        )
        self.assertEqual(rc, 0)
        return store

    def _consume_sample(self, store, rid, timeout_s=30):
        return _consume_mooncake_sample(store, self.STORE_ID, rid, timeout_s)

    @staticmethod
    def _load_lm_head():
        import json as _json

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        idx_path = hf_hub_download(TARGET_MODEL, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = _json.load(f)["weight_map"]
        shard = hf_hub_download(TARGET_MODEL, weight_map["lm_head.weight"])
        with safe_open(shard, framework="pt") as f:
            return f.get_tensor("lm_head.weight").float()

    def _assert_norm_contract(self, lm_head, row, expected_token, ctx):
        """Row through the LM head must reproduce the server's greedy token,
        up to bf16-export + kernel-numerics noise: exact argmax OR the
        expected token's recomputed logit within a small gap of the max
        (diagnosed near-tie: top-2 gap 0.05 on a ~23 logit scale)."""
        logits = row.float() @ lm_head.T
        pred = int(logits.argmax())
        if pred == expected_token:
            return
        gap = float(logits.max() - logits[expected_token])
        self.assertLess(
            gap,
            0.25,
            f"{ctx}: argmax {pred} != {expected_token} with gap {gap:.3f} "
            "(beyond numerical noise -> corrupted row)",
        )

    def test_a_graph_on_overwrite_safety(self):
        """Back-to-back different prompts: the second forward overwrites the
        BCG static buffer while the first export is in flight. Both samples
        must carry their own rows (norm contract on every decode row and the
        last prompt row)."""
        lm_head = self._load_lm_head()
        results = []
        for prompt in (
            "The chemical symbol for gold is",
            "The largest planet in the solar system is",
        ):
            ids = self.tokenizer(prompt).input_ids
            result = _generate(self.base_url, ids, max_new_tokens=8)
            results.append((ids, result))

        store = self._consumer()
        try:
            for ids, result in results:
                rid = result["meta_info"]["id"]
                sample = self._consume_sample(store, rid)
                prompt_len = len(ids)
                completion_ids = result["output_ids"]
                expected_rows = prompt_len + len(completion_ids) - 1
                self.assertEqual(sample["meta"]["num_tokens"], expected_rows)
                self.assertEqual(sample["input_ids"][:prompt_len].tolist(), ids)
                self._assert_norm_contract(
                    lm_head,
                    sample["last_hidden"][0, prompt_len - 1],
                    completion_ids[0],
                    f"rid {rid} last prompt row",
                )
                for t in range(len(completion_ids) - 1):
                    self._assert_norm_contract(
                        lm_head,
                        sample["last_hidden"][0, prompt_len + t],
                        completion_ids[t + 1],
                        f"rid {rid} decode row {t}",
                    )
        finally:
            store.close()

    def test_b_cold_vs_warm_prefix_parity(self):
        """Cold request A writes the prefix rows; warm request B (radix hit,
        never forwards those rows) must export a COMPLETE sample whose prefix
        rows are byte-identical to A's — cached-prefix rows present, not
        silently omitted or recomputed."""
        prefix = self.tokenizer(
            "In a quiet village nestled between rolling hills, an old "
            "clockmaker spent his days repairing timepieces that the "
            "townsfolk brought him. "
        ).input_ids
        ids_a = prefix + self.tokenizer("His favorite was a brass watch.").input_ids
        result_a = _generate(self.base_url, ids_a, max_new_tokens=4)
        ids_b = prefix + self.tokenizer("One winter morning a stranger came.").input_ids
        result_b = _generate(self.base_url, ids_b, max_new_tokens=4)

        store = self._consumer()
        try:
            sample_a = self._consume_sample(store, result_a["meta_info"]["id"])
            sample_b = self._consume_sample(store, result_b["meta_info"]["id"])
            n = len(prefix)
            # B is complete: full coverage including the cached prefix.
            self.assertEqual(
                sample_b["meta"]["num_tokens"],
                len(ids_b) + len(result_b["output_ids"]) - 1,
            )
            self.assertEqual(sample_b["input_ids"][: len(ids_b)].tolist(), ids_b)
            # Cached-prefix rows byte-identical to the cold writer's rows.
            self.assertTrue(
                torch.equal(sample_b["aux"][0, :n], sample_a["aux"][0, :n]),
                "warm-prefix aux rows differ from the cold writer's rows",
            )
            self.assertTrue(
                torch.equal(
                    sample_b["last_hidden"][0, :n], sample_a["last_hidden"][0, :n]
                ),
                "warm-prefix last rows differ from the cold writer's rows",
            )
        finally:
            store.close()

    def test_c_chunked_prefill_single_sample(self):
        """Forced multi-chunk prefill (chunk size 128, ~3.1 chunks) must
        assemble exactly one complete sample: full row coverage, exact
        input_ids, one meta key (no duplicates/partials)."""
        base = self.tokenizer("The quick brown fox jumps over the lazy dog. ").input_ids
        input_ids = (base * 40)[: CHUNKED_PREFILL_SIZE * 3 + 17]
        result = _generate(self.base_url, input_ids, max_new_tokens=4)
        rid = result["meta_info"]["id"]

        store = self._consumer()
        try:
            sample = self._consume_sample(store, rid, timeout_s=45)
            expected_rows = len(input_ids) + len(result["output_ids"]) - 1
            self.assertEqual(sample["meta"]["num_tokens"], expected_rows)
            self.assertEqual(sample["input_ids"][: len(input_ids)].tolist(), input_ids)
            self.assertEqual(sample["aux"].shape[1], expected_rows)
            # Exactly one sample: the g0 generation is written once
            # (first-write-wins); a duplicate assembly would need a second
            # meta write, which the sink refuses.
        finally:
            store.close()


class TestHiddenCaptureNonCompactVerify(CustomTestCase):
    """v4 invariant #3 (compact vs non-compact half): the committed-row
    selector must produce identical training data regardless of the verify
    layout. Runs the exact-norm-contract and coverage checks under
    SGLANG_RAGGED_VERIFY_MODE=static (dense strided verify path)."""

    @classmethod
    def setUpClass(cls):
        cls.capture_dir = tempfile.mkdtemp(prefix="hidden_capture_static_")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dspark_server_args(),
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_HIDDEN_CAPTURE_DIR": cls.capture_dir,
            },
        )
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_noncompact_coverage_and_norm_contract(self):
        input_ids = self.tokenizer("The first five prime numbers are").input_ids
        max_new_tokens = 24
        result = _generate(self.base_url, input_ids, max_new_tokens=max_new_tokens)
        self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
        rid = result["meta_info"]["id"]
        record = _wait_for_ckpt(self.capture_dir, rid)
        self.assertIsNotNone(record, "non-compact export missing")

        prompt_len = len(input_ids)
        T = record["input_ids"].shape[0]
        self.assertEqual(T, prompt_len + max_new_tokens - 1)
        completion_ids = result["output_ids"]
        self.assertEqual(
            record["input_ids"][prompt_len:].tolist(),
            completion_ids[: max_new_tokens - 1],
        )

        # Exact norm contract on decode rows (same assertion as the compact
        # suite): row t through the LM head == greedy token t+1.
        import json as _json

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        idx_path = hf_hub_download(TARGET_MODEL, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = _json.load(f)["weight_map"]
        shard = hf_hub_download(TARGET_MODEL, weight_map["lm_head.weight"])
        with safe_open(shard, framework="pt") as f:
            lm_head = f.get_tensor("lm_head.weight").float()
        for t in range(len(completion_ids) - 1):
            row = record["hidden_state"][0, prompt_len + t].float()
            self.assertEqual(
                int((row @ lm_head.T).argmax()),
                completion_ids[t + 1],
                f"norm contract violated at decode row {t} (static mode)",
            )


class TestHiddenCaptureMissSemantics(CustomTestCase):
    """Undersized staging ring: every prefill misses, serving is unaffected."""

    @classmethod
    def setUpClass(cls):
        cls.capture_dir = tempfile.mkdtemp(prefix="hidden_capture_miss_")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dspark_server_args(),
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_HIDDEN_CAPTURE_DIR": cls.capture_dir,
                # 1 slot x 8 tokens cannot hold a 128-token chunk.
                "SGLANG_HIDDEN_CAPTURE_STAGING_SLOTS": "1",
                "SGLANG_HIDDEN_CAPTURE_STAGING_SLOT_TOKENS": "8",
            },
        )
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_stage_full_is_clean_miss(self):
        base = self.tokenizer("A stitch in time saves nine. ").input_ids
        input_ids = (base * 20)[:150]
        result = _generate(self.base_url, input_ids, max_new_tokens=8)
        self.assertEqual(result["meta_info"]["finish_reason"]["type"], "length")
        rid = result["meta_info"]["id"]
        time.sleep(5.0)  # give the export thread time to (wrongly) write
        # The launcher's tiny warmup request fits the 1x8 ring and may export
        # legitimately; only OUR oversized request must be a whole-sample miss.
        self.assertFalse(
            os.path.exists(os.path.join(self.capture_dir, f"{rid}.ckpt")),
            "stage-full must miss whole samples",
        )
        partial = [f for f in os.listdir(self.capture_dir) if f.endswith(".tmp")]
        self.assertEqual(partial, [])


class TestHiddenCaptureDPReplicas(CustomTestCase):
    """Replica-level DP (--dp 2, no DP attention) writer ownership and
    consumer completeness against the acceptance sink (Mooncake): every
    request exports exactly once (no missing writes, no duplicate or
    cross-replica writes) and each sample reads back with its own content.
    Requires 2 GPUs and the mooncake binaries."""

    MASTER_PORT = 50055
    STORE_ID = "dp_ownership"
    MASTER_BIN = MOONCAKE_MASTER_BIN

    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("needs 2 GPUs")
        import subprocess

        cls.master = subprocess.Popen(
            [
                (
                    cls.MASTER_BIN
                    if os.path.exists(cls.MASTER_BIN)
                    else "mooncake_master"
                ),
                "--port",
                str(cls.MASTER_PORT),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dspark_server_args() + ["--dp", "2"],
            env={
                **os.environ,
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_HIDDEN_CAPTURE_SINK": "mooncake",
                "SGLANG_HIDDEN_CAPTURE_STORE_ID": cls.STORE_ID,
                "MOONCAKE_MASTER": f"127.0.0.1:{cls.MASTER_PORT}",
                "MOONCAKE_PROTOCOL": "tcp",
            },
        )
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "master") and cls.master:
            cls.master.terminate()

    def _consumer(self):
        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        rc = store.setup(
            "localhost",
            "P2PHANDSHAKE",
            2 * 1024**3,
            128 * 1024**2,
            "tcp",
            "",
            f"127.0.0.1:{self.MASTER_PORT}",
        )
        self.assertEqual(rc, 0)
        return store

    def _consume_tensor(self, store, key, spec):
        out = torch.empty(
            [int(x) for x in spec["shape"]],
            dtype=getattr(torch, spec["dtype"].split(".")[-1]),
        )
        nb = out.numel() * out.element_size()
        store.register_buffer(out.data_ptr(), nb)
        n = store.get_into(key, out.data_ptr(), nb)
        store.unregister_buffer(out.data_ptr())
        self.assertEqual(n, nb, key)
        return out

    def test_dp_writer_ownership_and_completeness(self):
        # Pin each request to an explicit replica (routed_dp_rank alternating
        # 0/1) so BOTH DP ranks' writers are provably exercised — no reliance
        # on router behavior.
        prompts = [f"Write the number {i} as an English word:" for i in range(12)]
        rids = []
        expected_ids = []
        pinned_ranks = []
        for i, prompt in enumerate(prompts):
            ids = self.tokenizer(prompt).input_ids
            rank = i % 2
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "routed_dp_rank": rank,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
            )
            response.raise_for_status()
            result = response.json()
            rids.append(result["meta_info"]["id"])
            pinned_ranks.append(rank)
            # Coverage: prompt + 4 generated - 1 (final token has no row).
            expected_ids.append((ids + result["output_ids"])[: len(ids) + 3])
        # Both ranks served requests.
        self.assertEqual(set(pinned_ranks), {0, 1})
        # No duplicate rids: a duplicate would collapse two samples onto one
        # key set (the duplicate-write failure mode). First-write-wins on a
        # real duplicate is covered by the sink unit test
        # (test_hidden_capture_mooncake.py::test_duplicate_sample_id_first_write_wins).
        self.assertEqual(len(set(rids)), len(rids))

        store = self._consumer()
        try:
            # Completeness across BOTH writers: every pinned request's meta
            # appears — rank-0 and rank-1 samples alike. A dead writer on
            # either replica fails this within one rank's half of the set.
            deadline = time.monotonic() + 60
            missing = set(rids)
            while missing and time.monotonic() < deadline:
                missing = {
                    rid
                    for rid in missing
                    if int(store.is_exist(f"{self.STORE_ID}/_samples/{rid}/meta")) != 1
                }
                if missing:
                    time.sleep(0.5)
            missing_ranks = {pinned_ranks[rids.index(rid)] for rid in missing}
            self.assertEqual(
                missing,
                set(),
                f"exports dropped (from dp rank(s) {missing_ranks}): {missing}",
            )

            # Ownership: each sample's consumed input_ids belong to ITS
            # request (a cross-replica mixup or kv-slot bleed would surface
            # here), with decode rows covered on both ranks.
            for rid, expected in zip(rids, expected_ids):
                sample = _consume_mooncake_sample(
                    store, self.STORE_ID, rid, timeout_s=5
                )
                meta = sample["meta"]
                self.assertEqual(meta["rid"], rid)
                self.assertEqual(meta["num_tokens"], len(expected))
                ids_back = sample["input_ids"]
                self.assertEqual(ids_back.tolist(), expected, f"rid {rid}")
            # Fingerprint written once, coherent across two replica writers.
            fp = json.loads(bytes(store.get(f"{self.STORE_ID}/_fingerprint")))
            self.assertEqual(fp["coverage"], "prefill_and_verify_commit")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main(verbosity=3)
