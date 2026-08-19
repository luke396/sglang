"""Mooncake sink: publish captured samples as immutable prefix segments.

Key layout (all tensor payloads are raw bytes; shape/dtype in the meta)::

    {store_id}/_segments/{writer_epoch}/{segment_id}/aux
    {store_id}/_segments/{writer_epoch}/{segment_id}/last_hidden
    {store_id}/_segments/{writer_epoch}/{segment_id}/meta
    {store_id}/_samples/{sample_id}/input_ids
    {store_id}/_samples/{sample_id}/meta   (written LAST: marks completeness)
    {store_id}/_seq/{dp_rank}/{n}          bare sample_id (manifest)
    {store_id}/_fingerprint                JSON: model/aux-layer/norm contract

Discovery: Mooncake is a flat KV store (no list/scan), so consumers tail the
per-writer manifest stream ``_seq/{w}/{n}``. Per-writer numbering because the
store is first-write-wins: a shared counter across DP replicas would swallow
entries. A missing ``n`` with ``n+1`` present means eviction or a failed put
— count and skip. Manifest entries are hard-pinned where supported (a lost
few-byte entry silently orphans a sample; pinned entries are tiny and
bounded). On restart the sink searches for the stream tail and resumes after
it; first-write-wins makes overwriting impossible by construction.

Payload lifecycle: payloads are NOT pinned. The trainer may lag or be
offline, and pinning would fill the store until every put fails; an evicted
sample is a capture-miss-equivalent (miss-over-backpressure). Revisit when
the trainer is in the loop.

Transport arenas (input-id staging, boundary read, segment lanes) are
registered once at startup — never per sample.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.state_capturer.hidden_prefix import (
    SAMPLE_SCHEMA,
    SEGMENT_SCHEMA,
    HiddenPrefixIndex,
    HiddenSegmentRef,
    build_segment_ref,
    context_id_for,
    prefix_hashes,
    segment_group_id_for,
)

logger = logging.getLogger(__name__)

# Constant generation segment: bypass capture writes each sample id once
# (first-write-wins), so the SpecForge re-put/supersede machinery is unused.

# Restart tail search: after the binary-search candidate, scan this many
# consecutive numbers for surviving entries before declaring the tail found.
# Holes come from failed manifest puts (burnt numbers) or degraded-mode
# eviction; a cluster wider than this window is handled by the pre-put
# existence check instead.
_TAIL_SCAN_WINDOW = 64


def _trace_span(trace: Any, kind: str):
    return trace.span(kind) if trace is not None else contextlib.nullcontext()


@dataclass
class _PrefixLane:
    index: int
    arena: torch.Tensor
    state: str = "FREE"
    state_since_ns: int = field(default_factory=time.monotonic_ns)


@dataclass
class _SegmentPutTask:
    lane: _PrefixLane
    payload_objects: List[Tuple[str, int, int]]
    meta_key: str
    meta: Dict[str, Any]
    group_id: str
    trace: Any = None
    done: threading.Event = field(default_factory=threading.Event)
    success: bool = False
    error: Optional[BaseException] = None


class MooncakeHiddenSink:
    """Registered-buffer zero-copy exporter with the file sink's interface.

    ``put`` / ``write_fingerprint`` are called from the single export thread;
    no internal locking is needed.
    """

    def __init__(
        self,
        *,
        store_id: str,
        max_export_tokens: int,
        dp_rank: int = 0,
        stats: Any = None,
        master_address: Optional[str] = None,
        store: Any = None,
        replicate_config: Any = None,
        manifest_config: Any = None,
        replicate_config_cls: Any = None,
        max_segment_rows: int = 256,
        prefix_lanes: int = 2,
        aux_width: Optional[int] = None,
        last_width: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        writer_epoch: Optional[str] = None,
    ) -> None:
        """``store``/``replicate_config``/``manifest_config`` are unit-test
        seams; production connects via ``_connect`` using the MOONCAKE_* env
        settings. ``stats`` (a ``HiddenCaptureStats``) receives the
        manifest-orphan counter; None keeps failures log-only."""
        self.store_id = store_id
        self.max_export_tokens = max_export_tokens
        self._owns_store = store is None
        if store is None:
            (
                store,
                replicate_config,
                manifest_config,
                replicate_config_cls,
            ) = _connect(master_address)
        self._store = store
        self._put_config = replicate_config
        self._replicate_config_cls = replicate_config_cls
        self._manifest_config = (
            manifest_config if manifest_config is not None else replicate_config
        )
        self._stats = stats
        # Prefix-segment publishing is the only Mooncake protocol.
        self.prefix_enabled = True
        self.max_segment_rows = int(max_segment_rows)
        self.prefix_lanes = int(prefix_lanes)
        self.aux_width = aux_width
        self.last_width = last_width
        self.dtype = dtype
        self.writer_epoch = writer_epoch or uuid.uuid4().hex
        self._prefix_index = HiddenPrefixIndex()
        self._context_id: Optional[str] = None
        self._boundary_staging: Optional[torch.Tensor] = None
        self._lanes: List[_PrefixLane] = []
        self._free_lanes: Optional[queue.Queue[_PrefixLane]] = None
        self._ready_tasks: Optional[queue.Queue[Any]] = None
        self._writer_stop_requested = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._registered_ptrs: List[int] = []
        self.registered_bytes = 0
        self._closed = False
        self._validate_prefix_configuration()
        self._manifest_prefix = f"{store_id}/_seq/{dp_rank}"
        self._next_manifest_seq = self._find_manifest_tail()

        try:
            registered_bytes = self._initialize_registered_arenas()
        except BaseException:
            self._unregister_registered_arenas()
            self._close_owned_store()
            raise
        self.registered_bytes = registered_bytes
        self._stats_observe_max("mooncake_registered_bytes_ct", registered_bytes)
        logger.info(
            "MooncakeHiddenSink ready: store_id=%s, registered=%d MB, "
            "max_segment_rows=%d, lanes=%d",
            store_id,
            registered_bytes // (1024 * 1024),
            self.max_segment_rows,
            self.prefix_lanes,
        )

    # ------------------------------------------------------------------ sink

    def put_prefix_sample(
        self,
        *,
        sample_id: str,
        rid: str,
        tokens: torch.Tensor,
        slots: torch.Tensor,
        prompt_len: int,
        own_slot_gens: Dict[int, int],
        sidecar: Any,
        trace: Any = None,
    ) -> Optional[bool]:
        """Publish one complete sample as immutable segment references.

        ``None`` is an attributable capture miss, ``False`` is a duplicate
        sample id, and ``True`` means sample meta was committed and a manifest
        append was attempted. This method runs only on the export worker.
        """
        if self._context_id is None:
            raise RuntimeError("prefix sink fingerprint was not initialized")

        sample_meta_key = self._sample_key(sample_id, "meta")
        if self._is_exist(sample_meta_key):
            return False

        tokens = tokens.to(dtype=torch.long, device="cpu").contiguous()
        slots = slots.to(dtype=torch.long, device="cpu").contiguous()
        token_values = tuple(int(token) for token in tokens.tolist())
        num_rows = len(token_values)
        if num_rows != int(slots.numel()):
            raise ValueError(
                f"prefix sample token/slot mismatch: {num_rows} != {slots.numel()}"
            )
        if num_rows > self.max_export_tokens:
            raise SampleTooLargeError(sample_id)

        self._bump("prefix_lookup_rows_ct", num_rows)
        hashes = prefix_hashes(self._context_id, token_values)
        lookup = self._lookup_live_prefix(token_values, hashes)
        if lookup is None:
            return None
        plan, stale_matched_until = lookup

        # A stale local hit is evidence that these matching warm rows once
        # belonged to an immutable object. If no alternative live segment now
        # covers them, token-only sidecar validation is not a trusted rebuild
        # source; exact own-generation rows remain safe.
        trusted_until = plan.matched_rows
        if stale_matched_until > trusted_until:
            stale_slots = slots[trusted_until:stale_matched_until].tolist()
            if any(int(slot) not in own_slot_gens for slot in stale_slots):
                self._bump("segment_exist_check_failed_miss_ct")
                return None

        refs: List[HiddenSegmentRef] = list(plan.full_segments)
        self._bump("prefix_reused_rows_ct", plan.reused_rows)
        boundary_rows = plan.matched_rows - plan.reused_rows
        self._bump("prefix_boundary_republished_rows_ct", boundary_rows)

        segment_tasks: List[_SegmentPutTask] = []
        synchronously_completed = 0
        start = plan.reused_rows
        while start < num_rows:
            end = min(start + self.max_segment_rows, num_rows)
            segment = build_segment_ref(
                context_id=self._context_id,
                hashes=hashes,
                tokens=token_values,
                start=start,
                end=end,
            )
            boundary_segment = (
                plan.partial_segment
                if start == plan.reused_rows and boundary_rows > 0
                else None
            )
            try:
                prepared = self._prepare_segment_publish(
                    segment=segment,
                    sample_start=start,
                    tokens=tokens,
                    slots=slots,
                    own_slot_gens=own_slot_gens,
                    sidecar=sidecar,
                    boundary_segment=boundary_segment,
                    boundary_rows=(
                        boundary_rows if boundary_segment is not None else 0
                    ),
                    trace=trace,
                )
            except BaseException:
                # Earlier segment tasks retain the same trace and registered
                # lanes. Settle them before unwinding so attribution closes
                # and no orphaned task writes into a later export's arena.
                self._finish_failed_segment_tasks(segment_tasks)
                raise
            if prepared is None:
                self._finish_failed_segment_tasks(segment_tasks)
                return None
            if isinstance(prepared, _SegmentPutTask):
                segment_tasks.append(prepared)
            elif prepared is True:
                synchronously_completed += 1
            refs.append(segment)
            start = end

        if not self._wait_segment_tasks(segment_tasks):
            self._bump("segment_orphan_ct", sum(task.success for task in segment_tasks))
            return None
        published_here = synchronously_completed + sum(
            task.success for task in segment_tasks
        )

        self._bump("suffix_published_rows_ct", max(0, num_rows - plan.matched_rows))
        try:
            with _trace_span(trace, "put"):
                self._publish_sample_input_ids(sample_id, tokens)
        except Exception:
            self._bump("sample_publish_failed_miss_ct")
            if published_here:
                self._bump("segment_orphan_ct", published_here)
            logger.warning(
                "hidden prefix capture: input_ids publish failed for %s",
                sample_id,
                exc_info=True,
            )
            return None

        input_ids_key = self._sample_key(sample_id, "input_ids")
        segment_component_keys = [
            key for segment in refs for key in self._segment_keys(segment.segment_id)
        ]
        try:
            complete = self._batch_exists([input_ids_key, *segment_component_keys])
        except Exception:
            self._bump("segment_exist_check_failed_miss_ct")
            self._bump("sample_publish_failed_miss_ct")
            if published_here:
                self._bump("segment_orphan_ct", published_here)
            logger.warning(
                "hidden prefix capture: final segment existence check failed "
                "for sample %s",
                sample_id,
                exc_info=True,
            )
            return None
        if not complete[0]:
            self._bump("sample_publish_failed_miss_ct")
            if published_here:
                self._bump("segment_orphan_ct", published_here)
            return None
        if not all(complete[1:]):
            self._bump("segment_component_missing_miss_ct")
            self._bump("sample_publish_failed_miss_ct")
            if published_here:
                self._bump("segment_orphan_ct", published_here)
            return None

        segment_offsets = list(_segments_with_offsets(refs))
        if sum(segment.rows for segment in refs) != num_rows or any(
            left.end_prefix_hash != right.start_prefix_hash
            for left, right in zip(refs, refs[1:])
        ):
            raise RuntimeError("internal prefix plan produced non-contiguous refs")
        sample_meta = {
            "schema": SAMPLE_SCHEMA,
            "sample_id": sample_id,
            "rid": rid,
            "context_id": self._context_id,
            "writer_epoch": self.writer_epoch,
            "num_rows": num_rows,
            "prompt_len": int(prompt_len),
            "input_ids": {
                "key": self._sample_key(sample_id, "input_ids"),
                "shape": [num_rows],
                "dtype": str(tokens.dtype),
            },
            "segments": [
                segment.sample_ref(offset) for segment, offset in segment_offsets
            ],
        }
        try:
            with _trace_span(trace, "put"):
                self._put_json(sample_meta_key, sample_meta)
        except Exception:
            self._bump("sample_publish_failed_miss_ct")
            if published_here:
                self._bump("segment_orphan_ct", published_here)
            logger.warning(
                "hidden prefix capture: sample meta publish failed for %s",
                sample_id,
                exc_info=True,
            )
            return None

        # Index only committed samples. Manifest remains last and may still
        # become an attributable orphan under the existing policy.
        self._prefix_index.add(self._context_id, refs)
        self._stats_observe_max(
            "prefix_index_segments_ct", self._prefix_index.segment_count
        )
        self._stats_observe_max("prefix_index_rows_ct", self._prefix_index.row_count)
        with _trace_span(trace, "put"):
            self._append_manifest(sample_id)
        return True

    def write_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        self._context_id = context_id_for(
            dict(fingerprint), "plain-text-extra-key-none"
        )
        key = f"{self.store_id}/_fingerprint"
        if not self._is_exist(key):
            self._put_json(key, fingerprint)

        # First-write-wins makes a concurrent DP writer's put look successful
        # even when it lost the race. Read back the authoritative value so a
        # reused store_id can never silently mix models or storage schemas.
        self._bump("mooncake_get_calls_ct")
        raw = self._store.get(key)
        if raw is None:
            raise RuntimeError("hidden capture fingerprint is not readable")
        if isinstance(raw, str):
            raw = raw.encode()
        try:
            stored = json.loads(bytes(raw))
        except Exception as error:
            raise RuntimeError("hidden capture fingerprint is invalid JSON") from error
        if stored != fingerprint:
            raise RuntimeError(
                "hidden capture store_id fingerprint mismatch; use a fresh "
                "SGLANG_HIDDEN_CAPTURE_STORE_ID for a different model or format"
            )

    def close(self) -> None:
        if self._closed:
            return
        if self._writer_thread is not None:
            self._writer_stop_requested.set()
            self._writer_thread.join(timeout=5.0)
            if self._writer_thread.is_alive():
                raise RuntimeError(
                    "hidden capture Mooncake writer did not stop; keeping "
                    "registered buffers alive"
                )
            self._writer_thread = None
        self._unregister_registered_arenas()
        self._close_owned_store()
        self._closed = True

    def state_snapshot(self) -> Dict[str, Any]:
        """Approximate live queue/lane state; counters remain authoritative."""
        lane_states: Dict[str, int] = {}
        for lane in self._lanes:
            lane_states[lane.state] = lane_states.get(lane.state, 0) + 1
        return {
            "closed": self._closed,
            "writer_alive": bool(
                self._writer_thread is not None and self._writer_thread.is_alive()
            ),
            "ready_tasks": self._ready_tasks.qsize() if self._ready_tasks else 0,
            "free_lanes": self._free_lanes.qsize() if self._free_lanes else 0,
            "lane_states": lane_states,
            "registered_bytes": self.registered_bytes,
            "prefix_index_segments": self._prefix_index.segment_count,
            "prefix_index_rows": self._prefix_index.row_count,
            "writer_epoch": self.writer_epoch,
            "store_id": self.store_id,
        }

    # ------------------------------------------------------------- internals

    def _validate_prefix_configuration(self) -> None:
        if self.max_segment_rows <= 0:
            raise ValueError("prefix max_segment_rows must be positive")
        if self.max_segment_rows > self.max_export_tokens:
            raise ValueError(
                "prefix max_segment_rows cannot exceed max_export_tokens: "
                f"{self.max_segment_rows} > {self.max_export_tokens}"
            )
        if self.prefix_lanes <= 0:
            raise ValueError("prefix_lanes must be positive")
        if not self.aux_width or not self.last_width or self.dtype is None:
            raise ValueError("prefix export requires aux_width, last_width, and dtype")
        config_cls = self._replicate_config_cls or type(self._put_config)
        try:
            probe = config_cls()
        except Exception as error:
            raise RuntimeError(
                "prefix V1 requires a constructible Mooncake ReplicateConfig"
            ) from error
        if not hasattr(probe, "group_ids"):
            raise RuntimeError(
                "prefix V1 requires Mooncake ReplicateConfig.group_ids "
                "(mooncake-transfer-engine 0.3.12.post1 or compatible)"
            )
        required_methods = ["batch_is_exist", "batch_get_into"]
        required_methods.append("batch_put_from")
        missing_methods = [
            name
            for name in required_methods
            if not callable(getattr(self._store, name, None))
        ]
        if missing_methods:
            raise RuntimeError(
                "prefix V1 requires Mooncake batch APIs: " + ", ".join(missing_methods)
            )
        self._replicate_config_cls = config_cls

    def _initialize_registered_arenas(self) -> int:
        """Allocate/register every arena before starting the writer thread."""
        # Prefix mode stages only int64 input-ids per sample.
        staging_bytes = self.max_export_tokens * 8
        self._staging = torch.empty(staging_bytes, dtype=torch.uint8)
        self._register_arena(self._staging, "sample staging")
        hidden_row_bytes = (self.aux_width + self.last_width) * self.dtype.itemsize
        self._boundary_staging = torch.empty(
            self.max_segment_rows * hidden_row_bytes,
            dtype=torch.uint8,
        )
        self._register_arena(self._boundary_staging, "boundary staging")
        self._free_lanes = queue.Queue(maxsize=self.prefix_lanes)
        self._ready_tasks = queue.Queue(maxsize=self.prefix_lanes)
        for index in range(self.prefix_lanes):
            lane = _PrefixLane(
                index=index,
                arena=torch.empty(
                    self.max_segment_rows * hidden_row_bytes,
                    dtype=torch.uint8,
                ),
            )
            self._register_arena(lane.arena, f"prefix lane {index}")
            self._lanes.append(lane)
            self._free_lanes.put_nowait(lane)
        self._writer_thread = threading.Thread(
            target=self._run_prefix_writer,
            name="hidden-capture-mooncake-writer",
            daemon=True,
        )
        self._writer_thread.start()
        registered_bytes = self._staging.numel() + sum(
            lane.arena.numel() for lane in self._lanes
        )
        if self._boundary_staging is not None:
            registered_bytes += self._boundary_staging.numel()
        return registered_bytes

    def _register_arena(self, arena: torch.Tensor, label: str) -> None:
        rc = self._store.register_buffer(arena.data_ptr(), arena.numel())
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake {label} register_buffer failed (status {rc})")
        self._registered_ptrs.append(arena.data_ptr())

    def _unregister_registered_arenas(self) -> None:
        for ptr in reversed(self._registered_ptrs):
            try:
                self._store.unregister_buffer(ptr)
            except Exception:  # pragma: no cover
                pass
        self._registered_ptrs.clear()

    def _close_owned_store(self) -> None:
        if not self._owns_store:
            return
        close_store = getattr(self._store, "close", None)
        if callable(close_store):
            close_store()
        self._owns_store = False

    def _group_config(self, group_id: str, count: int) -> Any:
        config = self._replicate_config_cls()
        for name in (
            "replica_num",
            "with_hard_pin",
            "with_soft_pin",
            "prefer_alloc_in_same_node",
            "preferred_segment",
            "preferred_segments",
            "data_type",
        ):
            if hasattr(config, name) and hasattr(self._put_config, name):
                setattr(config, name, getattr(self._put_config, name))
        config.group_ids = [group_id] * count
        return config

    def _sample_key(self, sample_id: str, name: str) -> str:
        return f"{self.store_id}/_samples/{sample_id}/{name}"

    def _segment_keys(self, segment_id: str) -> Tuple[str, str, str]:
        base = f"{self.store_id}/_segments/{self.writer_epoch}/{segment_id}"
        return f"{base}/aux", f"{base}/last_hidden", f"{base}/meta"

    def _batch_exists(self, keys: Sequence[str]) -> List[bool]:
        if not keys:
            return []
        self._bump("mooncake_batch_is_exist_calls_ct")
        self._bump("mooncake_batch_is_exist_keys_ct", len(keys))
        statuses = self._store.batch_is_exist(list(keys))
        if statuses is None or len(statuses) != len(keys):
            raise RuntimeError(
                "mooncake batch_is_exist returned an invalid status vector"
            )
        result = []
        for key, status in zip(keys, statuses):
            if status is None or int(status) < 0:
                raise RuntimeError(
                    f"mooncake is_exist failed for {key} (status {status})"
                )
            result.append(int(status) == 1)
        return result

    def _lookup_live_prefix(self, tokens: Sequence[int], hashes: Sequence[str]):
        excluded = set()
        stale_matched_until = 0
        while True:
            plan = self._prefix_index.lookup(
                context_id=self._context_id,
                tokens=tokens,
                hashes=hashes,
                excluded_segment_ids=excluded,
            )
            candidates = list(plan.full_segments)
            if plan.partial_segment is not None:
                candidates.append(plan.partial_segment)
            if not candidates:
                return plan, stale_matched_until
            keys = [
                key
                for segment in candidates
                for key in self._segment_keys(segment.segment_id)
            ]
            try:
                exists = self._batch_exists(keys)
            except Exception:
                self._bump("segment_exist_check_failed_miss_ct")
                logger.warning(
                    "hidden prefix capture: reuse existence check failed",
                    exc_info=True,
                )
                return None

            missing_ids = set()
            missing_components = 0
            for index, segment in enumerate(candidates):
                component_exists = exists[index * 3 : index * 3 + 3]
                if not all(component_exists):
                    missing_ids.add(segment.segment_id)
                    missing_components += component_exists.count(False)
            if not missing_ids:
                return plan, stale_matched_until

            self._bump("prefix_index_stale_hit_ct", len(missing_ids))
            self._bump("segment_component_missing_miss_ct", missing_components)
            offset = 0
            for segment in plan.full_segments:
                if segment.segment_id in missing_ids:
                    stale_matched_until = max(
                        stale_matched_until, offset + segment.rows
                    )
                offset += segment.rows
            if (
                plan.partial_segment is not None
                and plan.partial_segment.segment_id in missing_ids
            ):
                stale_matched_until = max(stale_matched_until, plan.matched_rows)
            excluded.update(missing_ids)

    def _prepare_segment_publish(
        self,
        *,
        segment: HiddenSegmentRef,
        sample_start: int,
        tokens: torch.Tensor,
        slots: torch.Tensor,
        own_slot_gens: Dict[int, int],
        sidecar: Any,
        boundary_segment: Optional[HiddenSegmentRef],
        boundary_rows: int,
        trace: Any = None,
    ):
        aux_key, last_key, meta_key = self._segment_keys(segment.segment_id)
        try:
            exists = self._batch_exists([aux_key, last_key, meta_key])
        except Exception:
            self._bump("segment_exist_check_failed_miss_ct")
            return None
        if all(exists):
            return False

        meta = self._segment_meta(segment)
        group_id = segment_group_id_for(
            store_id=self.store_id,
            writer_epoch=self.writer_epoch,
            segment_id=segment.segment_id,
        )
        missing_payload = [not exists[0], not exists[1]]
        if not any(missing_payload):
            try:
                with _trace_span(trace, "put"):
                    self._put_json_with_config(
                        meta_key, meta, self._group_config(group_id, 1)
                    )
                meta_bytes = len(json.dumps(meta, sort_keys=True).encode())
                self._bump("segment_object_count_ct")
                self._bump("segment_bytes_written_ct", meta_bytes)
                return True
            except Exception:
                self._bump("segment_publish_failed_miss_ct")
                self._bump("segment_orphan_ct")
                return None

        wait_started_ns = time.monotonic_ns()
        lane = self._free_lanes.get()
        self._bump("prefix_lane_wait_ns_ct", time.monotonic_ns() - wait_started_ns)
        self._transition_lane(lane, "GATHERING")
        gather_started_ns = time.monotonic_ns()
        try:
            aux_view, last_view = self._hidden_views(lane.arena, segment.rows)
            if boundary_segment is not None:
                if not self._read_boundary_into(
                    boundary_segment,
                    boundary_rows,
                    aux_view[:boundary_rows],
                    last_view[:boundary_rows],
                ):
                    self._transition_lane(lane, "FREE")
                    self._free_lanes.put(lane)
                    return None

            sidecar_start = sample_start + boundary_rows
            sidecar_rows = segment.rows - boundary_rows
            if sidecar_rows:
                read_args = {
                    "slots": slots[sidecar_start : sidecar_start + sidecar_rows],
                    "expected_tokens": tokens[
                        sidecar_start : sidecar_start + sidecar_rows
                    ],
                    "own_slot_gens": own_slot_gens,
                }
                valid = sidecar.read_rows_validated_into(
                    **read_args,
                    aux_dst=aux_view[boundary_rows:],
                    last_dst=last_view[boundary_rows:],
                )
                if not valid:
                    self._bump("prefix_invalid_miss_ct")
                    self._transition_lane(lane, "FREE")
                    self._free_lanes.put(lane)
                    return None

            aux_bytes = aux_view.numel() * aux_view.element_size()
            last_bytes = last_view.numel() * last_view.element_size()
            payload_objects = []
            if missing_payload[0]:
                payload_objects.append((aux_key, aux_view.data_ptr(), aux_bytes))
            if missing_payload[1]:
                payload_objects.append((last_key, last_view.data_ptr(), last_bytes))
            task = _SegmentPutTask(
                lane=lane,
                payload_objects=payload_objects,
                meta_key=meta_key,
                meta=meta,
                group_id=group_id,
                trace=trace,
            )
            self._transition_lane(lane, "READY")
            self._ready_tasks.put(task)
            self._stats_observe_max(
                "prefix_ready_queue_high_water_ct", self._ready_tasks.qsize()
            )
            return task
        except Exception:
            self._transition_lane(lane, "FREE")
            self._free_lanes.put(lane)
            raise
        finally:
            gather_ended_ns = time.monotonic_ns()
            self._bump("prefix_gather_busy_ns_ct", gather_ended_ns - gather_started_ns)
            if trace is not None:
                trace.record("gather", gather_started_ns, gather_ended_ns)

    def _segment_meta(self, segment: HiddenSegmentRef) -> Dict[str, Any]:
        return {
            "schema": SEGMENT_SCHEMA,
            "context_id": self._context_id,
            "segment_id": segment.segment_id,
            "start_prefix_hash": segment.start_prefix_hash,
            "end_prefix_hash": segment.end_prefix_hash,
            "num_rows": segment.rows,
            "tensors": {
                "aux": {
                    "shape": [segment.rows, self.aux_width],
                    "dtype": str(self.dtype),
                },
                "last_hidden": {
                    "shape": [segment.rows, self.last_width],
                    "dtype": str(self.dtype),
                },
            },
        }

    def _hidden_views(
        self, arena: torch.Tensor, rows: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_bytes = rows * self.aux_width * self.dtype.itemsize
        last_bytes = rows * self.last_width * self.dtype.itemsize
        if aux_bytes + last_bytes > arena.numel():
            raise SampleTooLargeError(f"segment-{rows}")
        aux = arena[:aux_bytes].view(self.dtype).view(rows, self.aux_width)
        last = (
            arena[aux_bytes : aux_bytes + last_bytes]
            .view(self.dtype)
            .view(rows, self.last_width)
        )
        return aux, last

    def _read_boundary_into(
        self,
        segment: HiddenSegmentRef,
        rows: int,
        aux_dst: torch.Tensor,
        last_dst: torch.Tensor,
    ) -> bool:
        aux_key, last_key, meta_key = self._segment_keys(segment.segment_id)
        try:
            self._bump("mooncake_get_calls_ct")
            raw_meta = self._store.get(meta_key)
            if raw_meta is None:
                self._bump("segment_component_missing_miss_ct")
                return False
            if isinstance(raw_meta, str):
                raw_meta = raw_meta.encode()
            meta = json.loads(bytes(raw_meta))
            if meta != self._segment_meta(segment):
                self._bump("segment_component_missing_miss_ct")
                return False

            boundary_aux, boundary_last = self._hidden_views(
                self._boundary_staging, segment.rows
            )
            keys = [aux_key, last_key]
            ptrs = [boundary_aux.data_ptr(), boundary_last.data_ptr()]
            sizes = [
                boundary_aux.numel() * boundary_aux.element_size(),
                boundary_last.numel() * boundary_last.element_size(),
            ]
            self._bump("mooncake_batch_get_calls_ct")
            self._bump("mooncake_batch_get_objects_ct", len(keys))
            statuses = self._store.batch_get_into(keys, ptrs, sizes)
            if statuses is None or len(statuses) != 2:
                self._bump("segment_component_missing_miss_ct")
                return False
            if any(int(status) != size for status, size in zip(statuses, sizes)):
                self._bump("segment_component_missing_miss_ct")
                return False
            aux_dst.copy_(boundary_aux[:rows])
            last_dst.copy_(boundary_last[:rows])
            self._bump("mooncake_batch_get_bytes_ct", sum(sizes))
            self._bump("segment_boundary_read_bytes_ct", sum(sizes))
            return True
        except Exception:
            self._bump("segment_component_missing_miss_ct")
            logger.warning(
                "hidden prefix capture: boundary segment read failed for %s",
                segment.segment_id,
                exc_info=True,
            )
            return False

    def _publish_sample_input_ids(self, sample_id: str, tokens: torch.Tensor) -> None:
        nbytes = tokens.numel() * tokens.element_size()
        self._staging[:nbytes].copy_(tokens.view(torch.uint8))
        self._put_batch_bytes(
            [
                (
                    self._sample_key(sample_id, "input_ids"),
                    self._staging.data_ptr(),
                    nbytes,
                )
            ]
        )

    def _wait_segment_tasks(self, tasks: Sequence[_SegmentPutTask]) -> bool:
        success = True
        for task in tasks:
            task.done.wait()
            if not task.success:
                success = False
                if task.error is not None:
                    logger.warning(
                        "hidden prefix capture: segment publish failed",
                        exc_info=(
                            type(task.error),
                            task.error,
                            task.error.__traceback__,
                        ),
                    )
        return success

    def _finish_failed_segment_tasks(self, tasks: Sequence[_SegmentPutTask]) -> None:
        self._wait_segment_tasks(tasks)
        completed = sum(task.success for task in tasks)
        if completed:
            self._bump("segment_orphan_ct", completed)

    def _run_prefix_writer(self) -> None:
        while True:
            if self._writer_stop_requested.is_set() and self._ready_tasks.empty():
                return
            try:
                task = self._ready_tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            self._transition_lane(task.lane, "PUTTING")
            put_started_ns = time.monotonic_ns()
            try:
                with _trace_span(task.trace, "put"):
                    self._execute_segment_put(task)
                task.success = True
            except BaseException as error:
                task.error = error
                self._bump("segment_publish_failed_miss_ct")
            finally:
                self._bump(
                    "prefix_put_busy_ns_ct", time.monotonic_ns() - put_started_ns
                )
                self._transition_lane(task.lane, "FREE")
                self._free_lanes.put(task.lane)
                task.done.set()

    def _execute_segment_put(self, task: _SegmentPutTask) -> None:
        keys = [key for key, _, _ in task.payload_objects]
        ptrs = [ptr for _, ptr, _ in task.payload_objects]
        sizes = [size for _, _, size in task.payload_objects]
        statuses = self._put_registered_objects(
            keys=keys,
            ptrs=ptrs,
            sizes=sizes,
            batch_config=self._group_config(task.group_id, len(keys)),
        )
        if statuses is None or len(statuses) != len(keys):
            self._bump("mooncake_batch_partial_failure_miss_ct")
            raise RuntimeError("invalid segment batch_put_from status vector")
        successful = [
            index
            for index, status in enumerate(statuses)
            if status is None or int(status) >= 0
        ]
        failed = [
            (keys[index], status)
            for index, status in enumerate(statuses)
            if status is not None and int(status) < 0
        ]
        if failed:
            self._bump("mooncake_batch_partial_failure_miss_ct")
            if successful:
                successful_bytes = sum(sizes[index] for index in successful)
                self._bump("mooncake_payload_objects_ct", len(successful))
                self._bump("mooncake_payload_bytes_ct", successful_bytes)
                self._bump("segment_orphan_ct")
                self._bump("segment_object_count_ct", len(successful))
                self._bump("segment_bytes_written_ct", successful_bytes)
            raise RuntimeError(f"segment payload partial failure: {failed}")

        self._bump("mooncake_payload_objects_ct", len(keys))
        self._bump("mooncake_payload_bytes_ct", sum(sizes))
        self._bump("segment_object_count_ct", len(keys))
        self._bump("segment_bytes_written_ct", sum(sizes))
        meta_payload = json.dumps(task.meta, sort_keys=True).encode()
        try:
            self._put_bytes_with_config(
                task.meta_key,
                meta_payload,
                self._group_config(task.group_id, 1),
            )
        except Exception:
            # Payload is immutable and already visible but cannot be
            # referenced until meta exists; account it as an orphan.
            self._bump("segment_orphan_ct")
            raise
        self._bump("segment_object_count_ct")
        self._bump("segment_bytes_written_ct", len(meta_payload))

    def _transition_lane(self, lane: _PrefixLane, state: str) -> None:
        now_ns = time.monotonic_ns()
        duration = now_ns - lane.state_since_ns
        if lane.state == "GATHERING":
            self._bump("prefix_lane_gathering_ns_ct", duration)
        elif lane.state == "READY":
            self._bump("prefix_lane_ready_ns_ct", duration)
        elif lane.state == "PUTTING":
            self._bump("prefix_lane_putting_ns_ct", duration)
        elif lane.state == "FREE":
            self._bump("prefix_lane_free_ns_ct", duration)
        lane.state = state
        lane.state_since_ns = now_ns

    def _stats_observe_max(self, name: str, value: int) -> None:
        observer = getattr(self._stats, "observe_max", None)
        if callable(observer):
            observer(name, value)

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
        if self._is_exist(self._manifest_key(0)):
            lo, hi = 0, 1
            while self._is_exist(self._manifest_key(hi)):
                lo, hi = hi, hi * 2
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self._is_exist(self._manifest_key(mid)):
                    lo = mid
                else:
                    hi = mid
            tail = lo + 1
        while True:
            occupied = [
                seq
                for seq in range(tail, tail + _TAIL_SCAN_WINDOW)
                if self._is_exist(self._manifest_key(seq))
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
            while self._is_exist(self._manifest_key(seq)):
                logger.warning(
                    "hidden capture manifest: sequence %s already occupied "
                    "(hole cluster wider than the restart scan window?); "
                    "skipping forward",
                    self._manifest_key(seq),
                )
                seq += 1
            self._next_manifest_seq = seq + 1
            payload = sample_id.encode()
            self._bump("mooncake_put_calls_ct")
            self._bump("mooncake_put_bytes_ct", len(payload))
            rc = self._store.put(
                self._manifest_key(seq), payload, self._manifest_config
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

    def _put_batch_bytes(self, objects) -> None:
        """Publish registered payload spans in one synchronous batch call.

        The pinned Mooncake binding consumes every source span before return.
        Per-key status is still load-bearing: any failure leaves only
        undiscoverable cold payload objects and prevents meta/manifest.
        """
        keys = [key for key, _, _ in objects]
        ptrs = [ptr for _, ptr, _ in objects]
        sizes = [size for _, _, size in objects]
        statuses = self._put_registered_objects(
            keys=keys,
            ptrs=ptrs,
            sizes=sizes,
            batch_config=self._put_config,
        )
        if statuses is None or len(statuses) != len(keys):
            self._bump("mooncake_batch_partial_failure_miss_ct")
            raise RuntimeError(
                "mooncake batch_put_from returned an invalid status vector: "
                f"expected {len(keys)}, got "
                f"{None if statuses is None else len(statuses)}"
            )
        failures = [
            (key, status)
            for key, status in zip(keys, statuses)
            if status is not None and int(status) < 0
        ]
        if failures:
            self._bump("mooncake_batch_partial_failure_miss_ct")
            successful = [
                index
                for index, status in enumerate(statuses)
                if status is None or int(status) >= 0
            ]
            self._bump("mooncake_payload_objects_ct", len(successful))
            self._bump(
                "mooncake_payload_bytes_ct",
                sum(sizes[index] for index in successful),
            )
            raise RuntimeError(
                "mooncake batch_put_from failed for "
                + ", ".join(f"{key} (status {status})" for key, status in failures)
            )
        self._bump("mooncake_payload_objects_ct", len(keys))
        self._bump("mooncake_payload_bytes_ct", sum(sizes))

    def _put_registered_objects(
        self,
        *,
        keys: Sequence[str],
        ptrs: Sequence[int],
        sizes: Sequence[int],
        batch_config: Any,
    ) -> Sequence[Any]:
        self._record_batch_put_attempt(sizes)
        return self._store.batch_put_from(
            list(keys), list(ptrs), list(sizes), batch_config
        )

    def _bump(self, name: str, delta: int = 1) -> None:
        if self._stats is not None:
            self._stats.bump(name, delta)

    def _is_exist(self, key: str) -> bool:
        self._bump("mooncake_is_exist_calls_ct")
        status = self._store.is_exist(key)
        if status is None or int(status) < 0:
            raise RuntimeError(f"mooncake is_exist failed for {key} (status {status})")
        return int(status) == 1

    def _record_batch_put_attempt(self, sizes: Sequence[int]) -> None:
        self._bump("mooncake_batch_put_calls_ct")
        self._bump("mooncake_batch_put_objects_attempted_ct", len(sizes))
        self._bump("mooncake_batch_put_bytes_attempted_ct", sum(sizes))

    def _put_json(self, key: str, obj: Dict[str, Any]) -> None:
        payload = json.dumps(obj, sort_keys=True).encode()
        self._bump("mooncake_put_calls_ct")
        self._bump("mooncake_put_bytes_ct", len(payload))
        rc = self._store.put(key, payload, self._put_config)
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake put failed (status {rc}) for {key}")

    def _put_json_with_config(self, key: str, obj: Dict[str, Any], config: Any) -> None:
        self._put_bytes_with_config(
            key, json.dumps(obj, sort_keys=True).encode(), config
        )

    def _put_bytes_with_config(self, key: str, payload: bytes, config: Any) -> None:
        self._bump("mooncake_put_calls_ct")
        self._bump("mooncake_put_bytes_ct", len(payload))
        rc = self._store.put(key, payload, config)
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake put failed (status {rc}) for {key}")


class SinkUnavailableError(RuntimeError):
    """The asynchronous Mooncake connector has not published a live sink."""


class AsyncMooncakeHiddenSink:
    """Fail-soft startup proxy with a bounded initial wait and background retry.

    Mooncake ``setup`` may spend roughly a minute in its own retry path when
    the master is absent. This proxy first performs a bounded TCP probe and
    builds the real sink on a daemon thread. ``write_fingerprint`` waits only
    the configured initial budget; serving then starts with capture pressure-
    degraded, and hooks become admissible automatically once ``ready`` flips.
    """

    def __init__(
        self,
        *,
        initial_probe_timeout_s: float,
        reconnect_interval_s: float,
        sink_factory: Any = None,
        probe_fn: Any = None,
        **sink_kwargs,
    ) -> None:
        from sglang.srt.environ import envs

        self._sink_kwargs = dict(sink_kwargs)
        self._stats = sink_kwargs.get("stats")
        self._master_address = (
            sink_kwargs.get("master_address") or envs.MOONCAKE_MASTER.get()
        )
        self._initial_probe_timeout_s = max(0.0, float(initial_probe_timeout_s))
        self._reconnect_interval_s = max(0.01, float(reconnect_interval_s))
        self._sink_factory = sink_factory or MooncakeHiddenSink
        self._probe_fn = probe_fn or _probe_master
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sink: Optional[MooncakeHiddenSink] = None
        self._fingerprint: Optional[Dict[str, Any]] = None
        self._closed = False
        self.prefix_enabled = True

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def registered_bytes(self) -> int:
        with self._lock:
            sink = self._sink
        return int(getattr(sink, "registered_bytes", 0)) if sink is not None else 0

    def write_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("asynchronous Mooncake sink is closed")
            self._fingerprint = dict(fingerprint)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._connect_loop,
                    name="hidden-capture-mooncake-connect",
                    daemon=True,
                )
                self._thread.start()
        if not self._ready.wait(timeout=self._initial_probe_timeout_s):
            self._bump("sink_initial_probe_timeout_ct")
            logger.warning(
                "hidden capture Mooncake was not ready within %.2fs; serving "
                "continues with capture degraded while background reconnect runs",
                self._initial_probe_timeout_s,
            )

    def put(self, sample_id: str, record: Dict[str, Any]) -> bool:
        return self._require_sink().put(sample_id, record)

    def put_prefix_sample(self, **kwargs) -> Optional[bool]:
        return self._require_sink().put_prefix_sample(**kwargs)

    def state_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            sink = self._sink
            thread = self._thread
        delegate_state = sink.state_snapshot() if sink is not None else None
        return {
            "connector_ready": self.ready,
            "connector_alive": bool(thread is not None and thread.is_alive()),
            "master_address": self._master_address,
            "closed": self._closed,
            "delegate": delegate_state,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            sink = self._sink
            self._sink = None
            self._ready.clear()
        self._stop.set()
        if thread is not None:
            thread.join(timeout=max(0.1, self._initial_probe_timeout_s))
            if thread.is_alive():
                logger.warning(
                    "hidden capture Mooncake connector is still inside setup; "
                    "leaving its daemon cleanup to complete asynchronously"
                )
        if sink is not None:
            sink.close()

    def _require_sink(self) -> MooncakeHiddenSink:
        with self._lock:
            sink = self._sink
        if sink is None:
            self._bump("sink_unavailable_miss_ct")
            raise SinkUnavailableError("Mooncake capture sink is not connected")
        return sink

    def _connect_loop(self) -> None:
        while not self._stop.is_set():
            self._bump("sink_reconnect_attempt_ct")
            try:
                reachable = self._probe_fn(
                    self._master_address, self._initial_probe_timeout_s
                )
            except Exception:
                reachable = False
                logger.debug("hidden capture Mooncake probe failed", exc_info=True)
            if not reachable:
                self._bump("sink_probe_failed_ct")
                if self._stop.wait(self._reconnect_interval_s):
                    return
                continue

            candidate = None
            try:
                candidate = self._sink_factory(**self._sink_kwargs)
                with self._lock:
                    fingerprint = dict(self._fingerprint or {})
                candidate.write_fingerprint(fingerprint)
            except BaseException:
                logger.warning(
                    "hidden capture Mooncake background connect failed; retrying",
                    exc_info=True,
                )
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        logger.debug("Mooncake candidate cleanup failed", exc_info=True)
                if self._stop.wait(self._reconnect_interval_s):
                    return
                continue

            with self._lock:
                if self._closed or self._stop.is_set():
                    publish = False
                else:
                    self._sink = candidate
                    self._ready.set()
                    publish = True
            if not publish:
                candidate.close()
                return
            self._bump("sink_reconnect_success_ct")
            logger.info("hidden capture Mooncake background connection is ready")
            return

    def _bump(self, name: str, delta: int = 1) -> None:
        if self._stats is not None:
            self._stats.bump(name, delta)


class SampleTooLargeError(Exception):
    """Sample exceeds the registered staging buffer; treated as a miss."""

    def __init__(self, sample_id: str) -> None:
        super().__init__(sample_id)
        self.sample_id = sample_id


def _probe_master(master_address: Optional[str], timeout_s: float) -> bool:
    """Bounded TCP reachability probe for the Mooncake master endpoint."""
    if not master_address:
        return False
    address = str(master_address)
    parsed = urlparse(address if "://" in address else f"tcp://{address}")
    host = parsed.hostname
    port = parsed.port
    if not host or port is None:
        logger.warning("invalid MOONCAKE_MASTER address for probe: %r", address)
        return False
    try:
        with socket.create_connection((host, port), timeout=max(0.01, timeout_s)):
            return True
    except OSError:
        return False


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
    try:
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
    except BaseException:
        close_store = getattr(store, "close", None)
        if callable(close_store):
            close_store()
        raise
    config = ReplicateConfig()
    config.replica_num = 1
    # No pin for payloads: see the module docstring's lifecycle rationale.
    return store, config, _build_manifest_config(ReplicateConfig), ReplicateConfig


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


def _segments_with_offsets(segments: Sequence[HiddenSegmentRef]):
    offset = 0
    for segment in segments:
        yield segment, offset
        offset += segment.rows
