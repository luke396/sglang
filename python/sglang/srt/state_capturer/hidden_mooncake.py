"""Mooncake sink for hidden-state capture (M2: bypass capture -> transit store).

Exports each sample as self-describing keys in a Mooncake distributed store,
consumable without any per-request response channel (bypass capture serves
real user traffic; the trainer discovers samples out of band):

    {store_id}/{sample_id}/g0/aux          raw bf16 bytes [T, K*H]
    {store_id}/{sample_id}/g0/last_hidden  raw bf16 bytes [T, H]
    {store_id}/{sample_id}/g0/input_ids    raw int64 bytes [T]
    {store_id}/{sample_id}/g0/meta         JSON: shapes/dtype/rid/loss_mask note
    {store_id}/_seq/{dp_rank}/{n}          bare sample_id (manifest, see below)
    {store_id}/_fingerprint                JSON: model/aux-layer/norm contract

Discovery: Mooncake is a flat KV store (no list/scan), so an out-of-band
consumer needs a tailable manifest. After a sample's meta lands, the sink
appends one manifest entry at a per-writer monotonically increasing sequence
number (per-writer streams because the store is first-write-wins: a shared
counter across DP replicas would silently swallow one replica's entries).
The consumer tails ``_seq/{w}/{n}`` per writer; a missing ``n`` with ``n+1``
present means the entry was evicted or its put failed — count and skip (gap
detection). Manifest entries are hard-pinned when the binding supports it:
an evicted 50MB payload is one missed sample, an evicted few-byte manifest
entry silently orphans it, and pinned entries are tiny and bounded. On
restart the sink probes for the stream's tail (exponential + binary search
over ``is_exist``) and resumes after it; first-write-wins makes overwriting
an existing sequence number impossible by construction.

Key layout matches SpecForge's ``MooncakeFeatureStore._tkey`` with a constant
generation (bypass capture never re-puts a sample id), so the trainer side can
consume with its existing zero-copy ``get_into`` path once it grows a bypass
adapter. Tensor payloads are raw buffer bytes — shape/dtype travel in ``meta``,
mirroring the spec-capture patch's convention where they ride the response.

Lifecycle: no hard pin (deliberate deviation from the v4 spec-capture patch
contract, which assumed an in-loop trainer releasing consumed samples).
Under bypass capture the trainer may lag or be offline; pinned samples would
accumulate until the store rejects every new put. Unpinned objects are
soft-pinned and evicted by the master's lease/watermark policy — an evicted
sample is a capture-miss-equivalent, consistent with the pipeline's
miss-over-backpressure semantics. Revisit pinning when the trainer is in the
loop.

Transport: one staging buffer of ``max_export_tokens`` rows is registered with
the transfer engine at startup and reused for every put (RDMA ``put_from``
requires a registered source; registration is a milliseconds-scale syscall
that must stay out of the hot path). Samples longer than the buffer are
whole-sample misses, counted via the export path's stats.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Constant generation segment: bypass capture writes each sample id once
# (first-write-wins), so the SpecForge re-put/supersede machinery is unused.
_GEN_SEGMENT = "g0"

# Restart tail search: after the binary-search candidate, scan this many
# consecutive numbers for surviving entries before declaring the tail found.
# Holes come from failed manifest puts (burnt numbers) or degraded-mode
# eviction; a cluster wider than this window is handled by the pre-put
# existence check instead.
_TAIL_SCAN_WINDOW = 64

_RECORD_TO_KEY_NAME = {
    "aux_hidden_state": "aux",
    "hidden_state": "last_hidden",
    "input_ids": "input_ids",
}


class MooncakeHiddenSink:
    """Registered-buffer zero-copy exporter with the file sink's interface.

    ``put`` / ``write_fingerprint`` are called from the single export thread;
    no internal locking is needed.
    """

    def __init__(
        self,
        *,
        store_id: str,
        row_bytes: int,
        max_export_tokens: int,
        dp_rank: int = 0,
        stats: Any = None,
        master_address: Optional[str] = None,
        store: Any = None,
        replicate_config: Any = None,
        manifest_config: Any = None,
    ) -> None:
        """``store``/``replicate_config``/``manifest_config`` are unit-test
        seams; production connects via ``_connect`` using the MOONCAKE_* env
        settings. ``stats`` (a ``HiddenCaptureStats``) receives the
        manifest-orphan counter; None keeps failures log-only."""
        self.store_id = store_id
        self.max_export_tokens = max_export_tokens
        if store is None:
            store, replicate_config, manifest_config = _connect(master_address)
        self._store = store
        self._put_config = replicate_config
        self._manifest_config = (
            manifest_config if manifest_config is not None else replicate_config
        )
        self._stats = stats
        self._manifest_prefix = f"{store_id}/_seq/{dp_rank}"
        self._next_manifest_seq = self._find_manifest_tail()

        # One registered staging buffer, reused for every sample: tensors are
        # serialized as raw bytes at increasing offsets, then put_from()
        # DMA-reads straight out of it.
        self._staging = torch.empty(max_export_tokens * row_bytes, dtype=torch.uint8)
        rc = self._store.register_buffer(
            self._staging.data_ptr(), self._staging.numel()
        )
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake register_buffer failed (status {rc})")
        logger.info(
            "MooncakeHiddenSink ready: store_id=%s, staging=%d MB registered",
            store_id,
            self._staging.numel() // (1024 * 1024),
        )

    # ------------------------------------------------------------------ sink

    def put(self, sample_id: str, record: Dict[str, Any]) -> bool:
        """Export one sample; False on duplicate id (first write wins)."""
        meta_key = self._key(sample_id, "meta")
        if int(self._store.is_exist(meta_key)) == 1:
            return False

        tensors = {
            _RECORD_TO_KEY_NAME[name]: record[name].contiguous()
            for name in _RECORD_TO_KEY_NAME
        }
        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if total_bytes > self._staging.numel():
            logger.warning(
                "hidden capture: sample %s (%d bytes) exceeds the registered "
                "staging buffer (%d bytes); capture miss. Raise "
                "SGLANG_HIDDEN_CAPTURE_MAX_EXPORT_TOKENS to fit longer samples",
                sample_id,
                total_bytes,
                self._staging.numel(),
            )
            raise SampleTooLargeError(sample_id)

        # Serialize into the registered staging buffer, then publish each
        # tensor's byte range. Safe to reuse per sample: put_from returns
        # after the transfer engine has consumed the source.
        offset = 0
        spans = {}
        for name, tensor in tensors.items():
            nbytes = tensor.numel() * tensor.element_size()
            flat = tensor.view(-1).view(torch.uint8)
            self._staging[offset : offset + nbytes] = flat
            spans[name] = (offset, nbytes)
            offset += nbytes
        for name, (start, nbytes) in spans.items():
            self._put_bytes(
                self._key(sample_id, name), self._staging.data_ptr() + start, nbytes
            )

        meta = {
            "rid": record["rid"],
            "num_tokens": int(record["input_ids"].numel()),
            "tensors": {
                name: {
                    "shape": list(tensors[name].shape),
                    "dtype": str(tensors[name].dtype),
                }
                for name in tensors
            },
            "loss_mask": "all_ones_placeholder",
        }
        # Meta goes last: its existence marks the sample complete, so a
        # consumer scanning for meta keys never sees partial samples.
        self._put_json(meta_key, meta)
        self._append_manifest(sample_id)
        return True

    def write_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        key = f"{self.store_id}/_fingerprint"
        if int(self._store.is_exist(key)) == 1:
            return  # idempotent across restarts and DP replicas
        self._put_json(key, fingerprint)

    def close(self) -> None:
        try:
            self._store.unregister_buffer(self._staging.data_ptr())
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------- internals

    def _key(self, sample_id: str, name: str) -> str:
        return f"{self.store_id}/{sample_id}/{_GEN_SEGMENT}/{name}"

    def _manifest_key(self, seq: int) -> str:
        return f"{self._manifest_prefix}/{seq}"

    def _find_manifest_tail(self) -> int:
        """Next free sequence number after the stream's current tail.

        Exponential probe + binary search over ``is_exist`` — O(log n) small
        RPCs, once per process start. Burnt numbers (failed puts) and, in
        degraded unpinned mode, evictions leave holes below the true tail,
        and the boundary search can land in one; a bounded scan past the
        candidate jumps over surviving entries so restart resumes after the
        last entry instead of resurrecting a hole consumers already skipped.
        """
        tail = 0
        if int(self._store.is_exist(self._manifest_key(0))) == 1:
            lo, hi = 0, 1
            while int(self._store.is_exist(self._manifest_key(hi))) == 1:
                lo, hi = hi, hi * 2
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if int(self._store.is_exist(self._manifest_key(mid))) == 1:
                    lo = mid
                else:
                    hi = mid
            tail = lo + 1
        while True:
            occupied = [
                seq
                for seq in range(tail, tail + _TAIL_SCAN_WINDOW)
                if int(self._store.is_exist(self._manifest_key(seq))) == 1
            ]
            if not occupied:
                return tail
            tail = max(occupied) + 1

    def _append_manifest(self, sample_id: str) -> None:
        """Publish one discovery entry; never raises (payload is already
        exported — a manifest failure orphans the sample, it must not be
        double-counted as a sink failure by the export worker).

        Every attempt burns its sequence number, success or not: mooncake
        ``put`` is first-write-wins (put-on-existing silently keeps the old
        value, verified against a live master), so retrying an ambiguous
        failure at the same number could silently no-op and lose the entry.
        A burnt number is just a hole, which consumers already tolerate via
        gap detection.
        """
        try:
            seq = self._next_manifest_seq
            # Last-line guard for a hole cluster wider than the tail scan
            # window: colliding with a survivor would silently lose this
            # entry, so skip forward instead.
            while int(self._store.is_exist(self._manifest_key(seq))) == 1:
                logger.warning(
                    "hidden capture manifest: sequence %s already occupied "
                    "(hole cluster wider than the restart scan window?); "
                    "skipping forward",
                    self._manifest_key(seq),
                )
                seq += 1
            self._next_manifest_seq = seq + 1
            rc = self._store.put(
                self._manifest_key(seq), sample_id.encode(), self._manifest_config
            )
            if rc is not None and int(rc) < 0:
                raise RuntimeError(f"manifest put failed (status {rc})")
        except Exception:
            if self._stats is not None:
                self._stats.bump("manifest_orphan_miss_ct")
            logger.warning(
                "hidden capture manifest: entry for sample %s failed; the "
                "sample is exported but undiscoverable (orphan, reclaimed by "
                "lease TTL)",
                sample_id,
                exc_info=True,
            )

    def _put_bytes(self, key: str, ptr: int, nbytes: int) -> None:
        rc = self._store.put_from(key, ptr, nbytes, self._put_config)
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake put_from failed (status {rc}) for {key}")

    def _put_json(self, key: str, obj: Dict[str, Any]) -> None:
        payload = json.dumps(obj, sort_keys=True).encode()
        rc = self._store.put(key, payload, self._put_config)
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake put failed (status {rc}) for {key}")


class SampleTooLargeError(Exception):
    """Sample exceeds the registered staging buffer; treated as a miss."""

    def __init__(self, sample_id: str) -> None:
        super().__init__(sample_id)
        self.sample_id = sample_id


def _connect(master_address: Optional[str]):
    """Connect a MooncakeDistributedStore client from MOONCAKE_* env settings."""
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
        _parse_global_segment_size,
    )

    master = master_address or envs.MOONCAKE_MASTER.get()
    if not master:
        raise RuntimeError(
            "SGLANG_HIDDEN_CAPTURE_SINK=mooncake requires MOONCAKE_MASTER"
        )
    store = MooncakeDistributedStore()
    rc = store.setup(
        envs.MOONCAKE_LOCAL_HOSTNAME.get(),
        envs.MOONCAKE_TE_META_DATA_SERVER.get(),
        _parse_global_segment_size(envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.get()),
        128 * 1024 * 1024,  # local transfer buffer
        envs.MOONCAKE_PROTOCOL.get(),
        envs.MOONCAKE_DEVICE.get() or "",
        master,
    )
    if int(rc) != 0:
        raise RuntimeError(f"mooncake store setup failed (status {rc})")
    config = ReplicateConfig()
    config.replica_num = 1
    # No pin for payloads: see the module docstring's lifecycle rationale.
    return store, config, _build_manifest_config(ReplicateConfig)


def _build_manifest_config(replicate_config_cls: Any) -> Any:
    """Manifest entries are the pinning exception: metadata must outlive the
    data it points at, or eviction silently orphans whole batches of samples.
    Entries are tiny and bounded, so pinning them cannot exhaust the store."""
    config = replicate_config_cls()
    config.replica_num = 1
    if hasattr(config, "with_hard_pin"):
        config.with_hard_pin = True
    else:
        logger.warning(
            "hidden capture: this mooncake binding has no with_hard_pin; "
            "manifest entries stay evictable and consumers must rely on gap "
            "detection to notice lost entries"
        )
    return config
