"""Small test-only reader for SGLang's V6 Mooncake producer contract.

The production consumer lives in SpecLoop.  Manual SGLang producer checks use
this module to validate real registered ``batch_get_into`` reconstruction
without adding a runtime dependency on that separate repository.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager

import torch


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name.split(".")[-1])


@contextmanager
def _registered(store, *tensors: torch.Tensor):
    registered = []
    try:
        for tensor in tensors:
            store.register_buffer(
                tensor.data_ptr(), tensor.numel() * tensor.element_size()
            )
            registered.append(tensor)
        yield
    finally:
        for tensor in reversed(registered):
            store.unregister_buffer(tensor.data_ptr())


def read_v6_sample(
    store, *, store_id: str, sample_id: str, timeout_s: float = 30.0
) -> dict:
    meta_key = f"{store_id}/_samples/{sample_id}/meta"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and int(store.is_exist(meta_key)) != 1:
        time.sleep(0.2)
    if int(store.is_exist(meta_key)) != 1:
        raise AssertionError(f"sample meta never appeared: {sample_id}")

    sample = json.loads(bytes(store.get(meta_key)))
    if sample.get("schema") != "hidden-sample-view-v1":
        raise AssertionError(f"unexpected sample schema: {sample.get('schema')}")
    if sample.get("sample_id") != sample_id:
        raise AssertionError("sample id mismatch")
    num_rows = int(sample["num_rows"])

    segment_entries = []
    expected_offset = 0
    for ref in sample["segments"]:
        offset = int(ref["sample_offset"])
        rows = int(ref["rows"])
        if offset != expected_offset or rows <= 0:
            raise AssertionError("segment refs contain a gap, overlap, or empty edge")
        expected_offset += rows
        base = f"{store_id}/_segments/{sample['writer_epoch']}/" f"{ref['segment_id']}"
        segment_meta = json.loads(bytes(store.get(f"{base}/meta")))
        if (
            segment_meta.get("schema") != "hidden-segment-v1"
            or int(segment_meta["num_rows"]) != rows
        ):
            raise AssertionError("segment meta does not match its sample ref")
        segment_entries.append((offset, rows, base, segment_meta))
    if expected_offset != num_rows or not segment_entries:
        raise AssertionError("segment refs do not strictly cover the sample")

    first_meta = segment_entries[0][3]
    aux_spec = first_meta["tensors"]["aux"]
    last_spec = first_meta["tensors"]["last_hidden"]
    input_ids = torch.empty(num_rows, dtype=_dtype(sample["input_ids"]["dtype"]))
    aux = torch.empty(
        (num_rows, int(aux_spec["shape"][1])), dtype=_dtype(aux_spec["dtype"])
    )
    last_hidden = torch.empty(
        (num_rows, int(last_spec["shape"][1])), dtype=_dtype(last_spec["dtype"])
    )

    keys = [sample["input_ids"]["key"]]
    pointers = [input_ids.data_ptr()]
    sizes = [input_ids.numel() * input_ids.element_size()]
    for offset, rows, base, segment_meta in segment_entries:
        if (
            segment_meta["tensors"]["aux"]["dtype"] != aux_spec["dtype"]
            or segment_meta["tensors"]["last_hidden"]["dtype"] != last_spec["dtype"]
        ):
            raise AssertionError("segment tensor dtype drift")
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

    with _registered(store, input_ids, aux, last_hidden):
        actual = store.batch_get_into(keys, pointers, sizes)
    if len(actual) != len(sizes) or any(
        int(received) != expected for received, expected in zip(actual, sizes)
    ):
        raise AssertionError("short V6 segment read")
    return {
        "meta": sample,
        "input_ids": input_ids,
        "aux": aux,
        "last_hidden": last_hidden,
    }
