"""Fixed-schema reader for nsys sqlite exports of capture-attribution runs.

Turns the ad-hoc queries behind the 2026-08-20 capture-overhead attribution
(SpecLoop PR #1) into a checked-in, deterministic report so before/after nsys
comparisons for issue #10 levers are reproducible.  Stdlib-only on purpose:
sqlite exports are often analyzed on machines without an sglang environment.

Usage:
    python3 nsys_report.py ARM.sqlite                     # single-arm report
    python3 nsys_report.py ON.sqlite OFF.sqlite           # paired delta report
    python3 nsys_report.py ON.sqlite OFF.sqlite --json out.json
    ... [--meta-a run.json --meta-b run.json]             # adds per-request /
                                                          # per-captured-row
                                                          # normalization

Metric definitions (pinned; every consumer of these numbers cites this file):

- window_s: span of CUPTI_ACTIVITY_KIND_RUNTIME rows (first API start to last
  API end).  The collection window is bracketed by the runner (warmup outside),
  so this is the burst window.
- API counts/durations: CUPTI_ACTIVITY_KIND_RUNTIME grouped by API name.  The
  report carries the full per-API delta table ordered by |count delta| — no
  whitelist; the 2026-08-20 analysis whitelisted three APIs and missed a
  +5,189 cudaStreamIsCapturing delta.
- Thread attribution: rows grouped by globalTid.  Threads are labeled by their
  OS name (ThreadNames) when distinctive.  Python thread names do NOT reach
  the OS on CPython <= 3.13, so arms recorded before the capture workers got
  prctl OS names show every thread as the process comm ("sglang::schedul...");
  for those the fallback classification applies: the tid issuing kernel
  launches is "main", any other tid issuing CUDA APIs is a "worker".  The
  2026-08-20 comment attributed all +32,821 cudaEventQuery to "the scheduler
  thread"; per-tid attribution shows the main thread contributed only ~+0.4k
  and the two capture worker threads the rest.
- D2H: CUPTI_ACTIVITY_KIND_MEMCPY copyKind=2 (DTOH), per stream.  The
  "dedicated" stream is the one carrying the most DTOH bytes.  Two bandwidth
  figures: bw_all (all copies) and bw_large (copies >= 1 MiB only — the
  published "52 GB/s" is bw_large; small header copies drag bw_all lower).
  The size histogram separates header-sized copies from payload copies.
- Overlap (three definitions, all reported):
    d2h_duty_cycle   = DTOH busy on dedicated stream / window
    d2h_hidden_ratio = DTOH time overlapping kernel-busy time / DTOH busy
                       (how much of the copy cost is hidden behind compute)
    d2h_idle_occupancy = DTOH time inside kernel-idle gaps / total gap time
                       (the published "0.7%" is this at one decimal; the
                       exact value on the 2026-08-20 data is 0.65%)
  Kernel-busy is the merged union of all kernel intervals on the device.
- Kernel parity: per-arm total kernel count and busy time, plus the top-N
  kernels by per-name total duration.  Parity is judged on relative busy-time
  deltas, NOT equal counts: sampled workloads (temperature > 0) accept
  different draft lengths per run, so kernel counts differ by O(1%) between
  healthy arms.  Two tolerances: total busy time (tight, default 3%) and
  per-name (loose, default 10%, applied only to kernels >= 1% of total busy)
  because per-name times on a sampled workload inherit verify-step-count
  variance (the 2026-08-20 arms show -3..-6% on ~1%-share kernels with the
  80%-share main kernel matching to 0.2%).  The failure mode this gate
  guards — capture pushing work onto the GPU — shows up as arm A busy
  meaningfully ABOVE arm B, so the sign is reported.
- Normalization: with --meta-a/--meta-b (runner run.json), counts are also
  reported per completed request and per captured row.

Verified against ~/nsys-baseline-20260820/ (2026-08-20 arms): reproduces
+5,282 cudaMemcpyAsync / +32,821 cudaEventQuery / 11.88 GB DTOH /
idle_occupancy 0.65% (published as 0.7%).
"""

import argparse
import json
import sqlite3

LARGE_COPY_BYTES = 1 << 20
COPY_SIZE_BUCKETS = (
    ("<=4KB", 0, 4096),
    ("4-64KB", 4097, 65536),
    ("64KB-1MB", 65537, LARGE_COPY_BYTES),
    (">1MB", LARGE_COPY_BYTES + 1, 1 << 62),
)
# OS names set via prctl by the capture workers (15-char comm limit).
CAPTURE_WORKER_OS_NAMES = ("hcap-d2h-launch", "hcap-finalize")
# Blocking waits: their duration is time spent waiting on the GPU, not host
# CPU work, and it varies with workload timing — excluded from the host-cost
# total (still listed in the per-API delta table).
BLOCKING_WAIT_APIS = (
    "cudaEventSynchronize",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cuStreamSynchronize",
    "cuCtxSynchronize",
)
DTOH = 2  # ENUM_CUDA_MEMCPY_OPER: CUDA_MEMCPY_KIND_DTOH


def _query(db, sql, params=()):
    return list(db.execute(sql, params))


def _window(db):
    row = _query(db, "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME")[0]
    return {"start_ns": row[0], "end_ns": row[1], "window_s": (row[1] - row[0]) / 1e9}


def _api_table(db):
    rows = _query(
        db,
        """SELECT s.value, COUNT(*), SUM(r.end - r.start)
           FROM CUPTI_ACTIVITY_KIND_RUNTIME r
           JOIN StringIds s ON r.nameId = s.id
           GROUP BY s.value""",
    )
    return {name: {"count": count, "ms": dur / 1e6} for name, count, dur in rows}


def _thread_names(db):
    rows = _query(
        db,
        """SELECT t.globalTid, s.value FROM ThreadNames t
           JOIN StringIds s ON t.nameId = s.id""",
    )
    return dict(rows)


def _thread_table(db):
    """Per-tid CUDA API attribution with OS-name or fallback classification."""
    names = _thread_names(db)
    tids = _query(
        db,
        """SELECT globalTid, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME
           GROUP BY globalTid ORDER BY 2 DESC""",
    )
    launcher_tids = {
        tid
        for tid, in _query(
            db,
            """SELECT DISTINCT r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r
               JOIN StringIds s ON r.nameId = s.id
               WHERE s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cuLaunchKernel%'""",
        )
    }
    threads = []
    for tid, total in tids:
        os_name = names.get(tid, "")
        if os_name in CAPTURE_WORKER_OS_NAMES:
            role = os_name
        elif tid in launcher_tids:
            role = "main"
        else:
            role = "worker"
        apis = _query(
            db,
            """SELECT s.value, COUNT(*), SUM(r.end - r.start)
               FROM CUPTI_ACTIVITY_KIND_RUNTIME r
               JOIN StringIds s ON r.nameId = s.id
               WHERE r.globalTid = ? GROUP BY s.value ORDER BY 2 DESC""",
            (tid,),
        )
        threads.append(
            {
                "tid": tid & 0xFFFFFF,
                "os_name": os_name,
                "role": role,
                "total_api_rows": total,
                "apis": {
                    name: {"count": count, "ms": dur / 1e6} for name, count, dur in apis
                },
            }
        )
    return threads


def _kernel_intervals(db):
    return _query(db, "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")


def _merge_intervals(intervals):
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _kernel_table(db, top_n):
    total = _query(
        db, "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    )[0]
    top = _query(
        db,
        """SELECT s.value, COUNT(*), SUM(k.end - k.start)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.shortName = s.id
           GROUP BY s.value ORDER BY 3 DESC LIMIT ?""",
        (top_n,),
    )
    return {
        "count": total[0],
        "busy_ms": total[1] / 1e6,
        "top": {name: {"count": count, "ms": dur / 1e6} for name, count, dur in top},
    }


def _dtoh_table(db):
    per_stream = _query(
        db,
        """SELECT streamId, COUNT(*), COALESCE(SUM(bytes), 0), SUM(end - start)
           FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = ?
           GROUP BY streamId ORDER BY 3 DESC""",
        (DTOH,),
    )
    streams = [
        {"stream": sid, "count": count, "bytes": byt, "busy_ms": dur / 1e6}
        for sid, count, byt, dur in per_stream
    ]
    dedicated = streams[0]["stream"] if streams and streams[0]["bytes"] > 0 else None
    result = {"streams": streams, "dedicated_stream": dedicated}
    if dedicated is None:
        return result

    copies = _query(
        db,
        """SELECT bytes, start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
           WHERE copyKind = ? AND streamId = ?""",
        (DTOH, dedicated),
    )
    total_bytes = sum(byt for byt, _, _ in copies)
    total_dur = sum(end - start for _, start, end in copies)
    large = [(byt, end - start) for byt, start, end in copies if byt >= LARGE_COPY_BYTES]
    large_bytes = sum(byt for byt, _ in large)
    large_dur = sum(dur for _, dur in large)
    histogram = {}
    for label, low, high in COPY_SIZE_BUCKETS:
        bucket = [byt for byt, _, _ in copies if low <= byt <= high]
        histogram[label] = {"count": len(bucket), "bytes": sum(bucket)}
    result.update(
        {
            "bytes": total_bytes,
            "busy_ms": total_dur / 1e6,
            "bw_all_gbps": total_bytes / (total_dur / 1e9) / 1e9 if total_dur else 0.0,
            "bw_large_gbps": large_bytes / (large_dur / 1e9) / 1e9 if large_dur else 0.0,
            "large_copy_count": len(large),
            "size_histogram": histogram,
        }
    )
    return result


def _interval_overlap(sorted_a, sorted_b):
    """Total overlap between two sorted, disjoint interval lists."""
    overlap = 0
    b_index = 0
    for a_start, a_end in sorted_a:
        while b_index < len(sorted_b) and sorted_b[b_index][1] <= a_start:
            b_index += 1
        probe = b_index
        while probe < len(sorted_b) and sorted_b[probe][0] < a_end:
            overlap += min(a_end, sorted_b[probe][1]) - max(a_start, sorted_b[probe][0])
            probe += 1
    return overlap


def _overlap_table(db, dedicated_stream, window):
    if dedicated_stream is None:
        return None
    busy = _merge_intervals(_kernel_intervals(db))
    gaps = [
        (busy[i][1], busy[i + 1][0])
        for i in range(len(busy) - 1)
        if busy[i + 1][0] > busy[i][1]
    ]
    copies = sorted(
        _query(
            db,
            """SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY
               WHERE copyKind = ? AND streamId = ?""",
            (DTOH, dedicated_stream),
        )
    )
    copy_busy = sum(end - start for start, end in copies)
    window_ns = window["end_ns"] - window["start_ns"]
    gap_total = sum(end - start for start, end in gaps)
    in_gap = _interval_overlap(copies, gaps)
    hidden = _interval_overlap(copies, [tuple(pair) for pair in busy])
    return {
        "d2h_duty_cycle": copy_busy / window_ns if window_ns else 0.0,
        "d2h_hidden_ratio": hidden / copy_busy if copy_busy else 0.0,
        "d2h_idle_occupancy": in_gap / gap_total if gap_total else 0.0,
        "kernel_idle_gap_ms": gap_total / 1e6,
        "d2h_busy_ms": copy_busy / 1e6,
    }


def read_arm(path, top_kernels):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    window = _window(db)
    dtoh = _dtoh_table(db)
    return {
        "path": path,
        "window": window,
        "api": _api_table(db),
        "threads": _thread_table(db),
        "kernels": _kernel_table(db, top_kernels),
        "dtoh": dtoh,
        "overlap": _overlap_table(db, dtoh["dedicated_stream"], window),
    }


def _load_meta(path):
    if path is None:
        return None
    with open(path) as handle:
        return json.load(handle)


def _normalizers(meta):
    if meta is None:
        return {}
    result = {}
    completed = meta.get("completed")
    if completed:
        result["per_request"] = int(completed)
    captured_rows = meta.get("captured_rows")
    if captured_rows:
        result["per_captured_row"] = int(captured_rows)
    return result


def api_delta(arm_a, arm_b):
    names = set(arm_a["api"]) | set(arm_b["api"])
    deltas = []
    for name in names:
        stats_a = arm_a["api"].get(name, {"count": 0, "ms": 0.0})
        stats_b = arm_b["api"].get(name, {"count": 0, "ms": 0.0})
        deltas.append(
            {
                "api": name,
                "count_a": stats_a["count"],
                "count_b": stats_b["count"],
                "count_delta": stats_a["count"] - stats_b["count"],
                "ms_delta": stats_a["ms"] - stats_b["ms"],
            }
        )
    deltas.sort(key=lambda row: -abs(row["count_delta"]))
    return deltas


def kernel_parity(arm_a, arm_b, busy_tolerance, per_name_tolerance):
    kernels_a, kernels_b = arm_a["kernels"], arm_b["kernels"]
    busy_a, busy_b = kernels_a["busy_ms"], kernels_b["busy_ms"]
    busy_rel = (busy_a - busy_b) / busy_b if busy_b else 0.0
    significant_ms = 0.01 * max(busy_a, busy_b)
    per_name = []
    worst_rel = 0.0
    for name in set(kernels_a["top"]) | set(kernels_b["top"]):
        ms_a = kernels_a["top"].get(name, {}).get("ms", 0.0)
        ms_b = kernels_b["top"].get(name, {}).get("ms", 0.0)
        base = max(ms_a, ms_b)
        rel = (ms_a - ms_b) / base if base else 0.0
        if base >= significant_ms:
            worst_rel = max(worst_rel, abs(rel))
        per_name.append({"kernel": name, "ms_a": ms_a, "ms_b": ms_b, "rel": rel})
    per_name.sort(key=lambda row: -max(row["ms_a"], row["ms_b"]))
    return {
        "busy_ms_a": busy_a,
        "busy_ms_b": busy_b,
        "busy_rel_delta": busy_rel,
        "count_a": kernels_a["count"],
        "count_b": kernels_b["count"],
        "top_worst_rel_delta": worst_rel,
        "busy_tolerance": busy_tolerance,
        "per_name_tolerance": per_name_tolerance,
        "within_tolerance": abs(busy_rel) <= busy_tolerance
        and worst_rel <= per_name_tolerance,
        "per_name": per_name,
    }


def capture_worker_rows(arm):
    """CUDA API rows attributed to capture worker threads (named or fallback).

    The lever-2 acceptance assertion: after deleting the verify D2H launcher
    this must drop to zero rows for the launcher thread.
    """
    rows = {}
    for thread in arm["threads"]:
        if thread["role"] == "main":
            continue
        if thread["os_name"] in CAPTURE_WORKER_OS_NAMES:
            key = thread["os_name"]
        else:
            key = f"worker-tid-{thread['tid']}"
        rows[key] = {
            "total_api_rows": thread["total_api_rows"],
            "apis": thread["apis"],
        }
    return rows


def _fmt_bytes(value):
    return f"{value / 1e9:.2f}GB" if value >= 1e7 else f"{value / 1e3:.0f}KB"


def _print_arm(label, arm, normalizers):
    window = arm["window"]["window_s"]
    print(f"== arm {label}: {arm['path']}")
    print(f"   window={window:.1f}s")
    for thread in arm["threads"]:
        top = sorted(thread["apis"].items(), key=lambda kv: -kv[1]["count"])[:3]
        summary = ", ".join(f"{name}={stats['count']}" for name, stats in top)
        name = thread["os_name"] or "-"
        print(
            f"   thread tid={thread['tid']} role={thread['role']} os_name={name}: {summary}"
        )
    kernels = arm["kernels"]
    print(f"   kernels: count={kernels['count']} busy={kernels['busy_ms']:.0f}ms")
    dtoh = arm["dtoh"]
    if dtoh["dedicated_stream"] is not None and "bytes" in dtoh:
        print(
            f"   dtoh[stream {dtoh['dedicated_stream']}]: {_fmt_bytes(dtoh['bytes'])} "
            f"in {sum(s['count'] for s in dtoh['streams'])} copies, "
            f"bw_all={dtoh['bw_all_gbps']:.1f}GB/s "
            f"bw_large={dtoh['bw_large_gbps']:.1f}GB/s ({dtoh['large_copy_count']} copies >=1MiB)"
        )
        histogram = ", ".join(
            f"{label}:{bucket['count']}" for label, bucket in dtoh["size_histogram"].items()
        )
        print(f"   dtoh size histogram: {histogram}")
    if arm["overlap"]:
        overlap = arm["overlap"]
        print(
            f"   overlap: duty_cycle={overlap['d2h_duty_cycle'] * 100:.2f}% "
            f"hidden_ratio={overlap['d2h_hidden_ratio'] * 100:.1f}% "
            f"idle_occupancy={overlap['d2h_idle_occupancy'] * 100:.2f}%"
        )
    for norm_label, denom in normalizers.items():
        print(f"   normalizer {norm_label}: denominator={denom}")


def _print_delta(deltas, arm_a, arm_b, parity, normalizers_a, top_apis):
    print("== API delta (arm A - arm B), by |count delta| ==")
    print(f"   {'api':44s} {'A':>9s} {'B':>9s} {'delta':>9s} {'ms_delta':>10s}")
    for row in deltas[:top_apis]:
        print(
            f"   {row['api']:44s} {row['count_a']:9d} {row['count_b']:9d} "
            f"{row['count_delta']:+9d} {row['ms_delta']:+9.1f}ms"
        )
    per_request = normalizers_a.get("per_request")
    if per_request:
        host_ms = sum(
            row["ms_delta"]
            for row in deltas
            if not row["api"].startswith(BLOCKING_WAIT_APIS)
        )
        print(f"   host API time delta total (excl. blocking waits): {host_ms:+.1f}ms "
              f"= {host_ms / per_request:+.2f}ms/request ({per_request} requests)")
    print("== capture worker threads (arm A) ==")
    for name, info in capture_worker_rows(arm_a).items():
        top = sorted(info["apis"].items(), key=lambda kv: -kv[1]["count"])[:3]
        summary = ", ".join(f"{api}={stats['count']}" for api, stats in top)
        print(f"   {name}: {info['total_api_rows']} API rows ({summary})")
    print("== kernel parity ==")
    verdict = "OK" if parity["within_tolerance"] else "VIOLATION"
    print(
        f"   busy A={parity['busy_ms_a']:.0f}ms B={parity['busy_ms_b']:.0f}ms "
        f"rel={parity['busy_rel_delta'] * 100:+.2f}% (tol {parity['busy_tolerance'] * 100:.0f}%) "
        f"count A={parity['count_a']} B={parity['count_b']} "
        f"top_worst_rel={parity['top_worst_rel_delta'] * 100:.2f}% "
        f"(tol {parity['per_name_tolerance'] * 100:.0f}%, kernels >=1% busy) -> {verdict}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sqlite_a", help="arm A sqlite (capture-on for paired runs)")
    parser.add_argument("sqlite_b", nargs="?", help="arm B sqlite (capture-off)")
    parser.add_argument("--meta-a", help="runner run.json for arm A (normalization)")
    parser.add_argument("--meta-b", help="runner run.json for arm B")
    parser.add_argument("--json", dest="json_out", help="write full report as JSON")
    parser.add_argument("--top-apis", type=int, default=15)
    parser.add_argument("--top-kernels", type=int, default=15)
    parser.add_argument(
        "--kernel-busy-tolerance",
        type=float,
        default=0.03,
        help="relative total-busy-time delta allowed before kernel parity fails",
    )
    parser.add_argument(
        "--kernel-per-name-tolerance",
        type=float,
        default=0.10,
        help="relative per-kernel busy delta allowed (kernels >=1%% of busy)",
    )
    args = parser.parse_args()

    arm_a = read_arm(args.sqlite_a, args.top_kernels)
    normalizers_a = _normalizers(_load_meta(args.meta_a))
    _print_arm("A", arm_a, normalizers_a)
    report = {"arm_a": arm_a, "normalizers_a": normalizers_a}

    if args.sqlite_b:
        arm_b = read_arm(args.sqlite_b, args.top_kernels)
        normalizers_b = _normalizers(_load_meta(args.meta_b))
        _print_arm("B", arm_b, normalizers_b)
        deltas = api_delta(arm_a, arm_b)
        parity = kernel_parity(
            arm_a,
            arm_b,
            busy_tolerance=args.kernel_busy_tolerance,
            per_name_tolerance=args.kernel_per_name_tolerance,
        )
        _print_delta(deltas, arm_a, arm_b, parity, normalizers_a, args.top_apis)
        report.update(
            {
                "arm_b": arm_b,
                "normalizers_b": normalizers_b,
                "api_delta": deltas,
                "kernel_parity": parity,
                "capture_worker_rows_a": capture_worker_rows(arm_a),
                "capture_worker_rows_b": capture_worker_rows(arm_b),
            }
        )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"json report: {args.json_out}")


if __name__ == "__main__":
    main()
