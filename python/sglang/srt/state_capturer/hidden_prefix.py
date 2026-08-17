"""Pure producer-side metadata for Mooncake hidden-prefix segments.

The remote store remains a flat KV.  This module owns only the writer-local
hint index used to find immutable, whole-segment prefix reuse; losing the
index reduces reuse but never changes the persisted sample contract.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

SEGMENT_SCHEMA = "hidden-segment-v1"
SAMPLE_SCHEMA = "hidden-sample-view-v1"
SEGMENT_LAYOUT_VERSION = "hidden-segment-layout-v1"


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def context_id_for(fingerprint: dict, request_namespace: str) -> str:
    """Bind sharing to every capture semantic that affects hidden values."""
    payload = {
        "schema": SEGMENT_SCHEMA,
        "fingerprint": fingerprint,
        "request_namespace": request_namespace,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def prefix_hashes(context_id: str, tokens: Sequence[int]) -> List[str]:
    """Return chained hashes h[0..T] for one token sequence."""
    state = hashlib.sha256(b"hidden-prefix-v1\0" + bytes.fromhex(context_id)).digest()
    out = [state.hex()]
    for token in tokens:
        state = hashlib.sha256(
            b"hidden-prefix-token-v1\0" + state + struct.pack("<q", int(token))
        ).digest()
        out.append(state.hex())
    return out


def segment_id_for(
    *, context_id: str, start_prefix_hash: str, end_prefix_hash: str, rows: int
) -> str:
    payload = {
        "context_id": context_id,
        "end_prefix_hash": end_prefix_hash,
        "layout_version": SEGMENT_LAYOUT_VERSION,
        "rows": int(rows),
        "start_prefix_hash": start_prefix_hash,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def segment_group_id_for(*, store_id: str, writer_epoch: str, segment_id: str) -> str:
    payload = ["hidden-segment-v1", store_id, writer_epoch, segment_id]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class HiddenSegmentRef:
    segment_id: str
    start_prefix_hash: str
    end_prefix_hash: str
    rows: int
    tokens: Tuple[int, ...]

    def sample_ref(self, sample_offset: int) -> dict:
        return {
            "segment_id": self.segment_id,
            "sample_offset": int(sample_offset),
            "rows": self.rows,
        }


@dataclass(frozen=True)
class PrefixLookup:
    full_segments: Tuple[HiddenSegmentRef, ...]
    reused_rows: int
    matched_rows: int
    partial_segment: Optional[HiddenSegmentRef]


class HiddenPrefixIndex:
    """Writer-local exact-token index keyed by a segment start boundary."""

    def __init__(self) -> None:
        self._by_start: Dict[Tuple[str, str], List[HiddenSegmentRef]] = {}
        self._segment_ids: Set[str] = set()
        self._row_count = 0

    def add(self, context_id: str, segments: Iterable[HiddenSegmentRef]) -> None:
        for segment in segments:
            key = (context_id, segment.start_prefix_hash)
            bucket = self._by_start.setdefault(key, [])
            if all(existing.segment_id != segment.segment_id for existing in bucket):
                bucket.append(segment)
                self._segment_ids.add(segment.segment_id)
                self._row_count += segment.rows
                # Prefer the longest complete reusable edge at a boundary.
                bucket.sort(key=lambda item: (-item.rows, item.segment_id))

    @property
    def segment_count(self) -> int:
        return len(self._segment_ids)

    @property
    def row_count(self) -> int:
        return self._row_count

    def lookup(
        self,
        *,
        context_id: str,
        tokens: Sequence[int],
        hashes: Sequence[str],
        excluded_segment_ids: Optional[Set[str]] = None,
    ) -> PrefixLookup:
        excluded = excluded_segment_ids or set()
        token_tuple = tuple(int(token) for token in tokens)
        total = len(token_tuple)
        offset = 0
        full: List[HiddenSegmentRef] = []

        while offset < total:
            candidates = [
                segment
                for segment in self._by_start.get((context_id, hashes[offset]), ())
                if segment.segment_id not in excluded
            ]
            selected = next(
                (
                    segment
                    for segment in candidates
                    if offset + segment.rows <= total
                    and token_tuple[offset : offset + segment.rows] == segment.tokens
                ),
                None,
            )
            if selected is None:
                best_partial = None
                best_lcp = 0
                remaining = token_tuple[offset:]
                for segment in candidates:
                    lcp = _common_prefix_len(remaining, segment.tokens)
                    if 0 < lcp < segment.rows and lcp > best_lcp:
                        best_partial = segment
                        best_lcp = lcp
                return PrefixLookup(
                    full_segments=tuple(full),
                    reused_rows=offset,
                    matched_rows=offset + best_lcp,
                    partial_segment=best_partial,
                )
            full.append(selected)
            offset += selected.rows

        return PrefixLookup(
            full_segments=tuple(full),
            reused_rows=offset,
            matched_rows=offset,
            partial_segment=None,
        )


def build_segment_ref(
    *,
    context_id: str,
    hashes: Sequence[str],
    tokens: Sequence[int],
    start: int,
    end: int,
) -> HiddenSegmentRef:
    if not 0 <= start < end <= len(tokens):
        raise ValueError(f"invalid segment range [{start}, {end})")
    rows = end - start
    start_hash = hashes[start]
    end_hash = hashes[end]
    return HiddenSegmentRef(
        segment_id=segment_id_for(
            context_id=context_id,
            start_prefix_hash=start_hash,
            end_prefix_hash=end_hash,
            rows=rows,
        ),
        start_prefix_hash=start_hash,
        end_prefix_hash=end_hash,
        rows=rows,
        tokens=tuple(int(token) for token in tokens[start:end]),
    )


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit
