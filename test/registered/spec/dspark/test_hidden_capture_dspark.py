"""E2E tests for online hidden-state capture under DSpark (M1: prefill rows).

Validates the exported per-sample ``.ckpt`` files against independent
references:

- ``input_ids`` must match the request's token ids exactly;
- ``aux_hidden_state`` rows must match a HuggingFace CPU forward at the
  draft-configured aux layers (residual stream after target layer k);
- ``hidden_state`` (post-final-norm) must match HF's final hidden states, and
  its last row through the LM head must reproduce the server's first greedy
  token;
- chunked prefill and warm-prefix reuse must yield complete, consistent
  samples (or a clean whole-sample miss — never partial data);
- an undersized staging ring must produce misses without breaking serving.
"""

import json
import os
import tempfile
import time
import unittest

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

if is_sm100_supported():
    ATTENTION_BACKEND = "trtllm_mha"
    DRAFT_ATTENTION_BACKEND = "fa4"
else:
    ATTENTION_BACKEND = "fa3"
    DRAFT_ATTENTION_BACKEND = "fa3"

CHUNKED_PREFILL_SIZE = 128
_EXPORT_POLL_TIMEOUT_S = 30.0


def _dspark_server_args():
    return [
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
        "--disable-piecewise-cuda-graph",
        "--enable-hidden-state-capture",
    ]


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
        self.assertEqual(record["input_ids"].tolist(), input_ids)
        self.assertEqual(record["loss_mask"].tolist(), [1] * T)
        self.assertEqual(record["aux_hidden_state"].shape, (1, T, num_aux * hidden))
        self.assertEqual(record["hidden_state"].shape, (1, T, hidden))
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
        # = HF hidden_states[k + 1].
        aux = record["aux_hidden_state"][0].view(T, num_aux, hidden)
        for j, layer_id in enumerate(fp["aux_layer_ids"]):
            ref = hf_out.hidden_states[layer_id + 1][0]
            cos = _row_cosine(aux[:, j, :], ref)
            self.assertGreater(
                float(cos.min()),
                0.98,
                f"aux layer {layer_id}: min row cosine {float(cos.min()):.4f}",
            )

        # target-last = post-final-norm = HF hidden_states[-1].
        last = record["hidden_state"][0]
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
        self.assertEqual(record["input_ids"].tolist(), input_ids)
        self.assertEqual(record["aux_hidden_state"].shape[1], len(input_ids))

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
        self.assertEqual(record_b["input_ids"].tolist(), ids_b)
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
    """Replica-level DP (--dp 2, no DP attention): each replica's scheduler
    runs its own capture pipeline into a shared sink dir. Requires 2 GPUs."""

    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("needs 2 GPUs")
        cls.capture_dir = tempfile.mkdtemp(prefix="hidden_capture_dp_")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dspark_server_args() + ["--dp", "2"],
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

    def test_replicas_export_without_collisions(self):
        # Distinct prompts so the round-robin router spreads them over both
        # replicas; rids are globally unique, so no filename collisions.
        prompts = [f"Write the number {i} as an English word:" for i in range(12)]
        rids = []
        for prompt in prompts:
            result = _generate(
                self.base_url, self.tokenizer(prompt).input_ids, max_new_tokens=4
            )
            rids.append(result["meta_info"]["id"])
        self.assertEqual(len(set(rids)), len(rids))

        exported = 0
        for rid, prompt in zip(rids, prompts):
            record = _wait_for_ckpt(self.capture_dir, rid)
            self.assertIsNotNone(record, f"missing export for {rid}")
            self.assertEqual(
                record["input_ids"].tolist(), self.tokenizer(prompt).input_ids
            )
            exported += 1
        self.assertEqual(exported, len(prompts))
        # Single coherent fingerprint despite two replica writers.
        with open(os.path.join(self.capture_dir, "_fingerprint.json")) as f:
            self.assertEqual(
                json.load(f)["norm_contract"], "post_final_norm_pre_lm_head"
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
