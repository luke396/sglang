# Capture performance measurement (issue #11)

How capture-overhead changes (issue #10 levers and successors) are measured,
what runs when, and where results are archived.  The workload contract and
metric definitions live in the instruments' docstrings; this file is the
operating manual.

## Three instruments

| Question | Instrument | Output |
|---|---|---|
| Did anything regress? | `bench_hidden_capture_matrix.py` (regression gate, do not modify per-change) | 12-leg off/on matrix JSONL |
| How much did the symptom move? | `bench_capture_attribution.py --mode bare` | paired per-request TTFT/ITL by input-length bucket |
| What happened mechanically? | `bench_capture_attribution.py --mode trace` + `nsys_report.py` | per-thread API count/duration deltas, kernel parity, D2H bytes/bandwidth/overlap |

Why bare and trace are separate runs of the same leg: nsys intercepts every
host CUDA API call, and capture-on differs from capture-off by tens of
thousands of calls per burst, so traced latency deltas are systematically
inflated.  **Time numbers only from bare; counts/attribution only from
trace.**  Never quote a TTFT from a traced run.

## Tiered run rules

| Change type | What to run |
|---|---|
| Doesn't touch the capture path | nothing (unit tests / CI cover it) |
| Capture semantics/correctness change | unit tests + matrix `rand_high`/`warm_high` legs (4 legs, ~25 min) |
| Performance lever (issue #10 levers 1-4) | full set: matrix regression cells + bare on/off + trace on/off, nothing skipped |
| Milestone | full matrix incl. supplement + bare, roll the baseline |

Off arms are never skipped: every run is an off/on pair on the same GPU with
the same dataset file, seed and schedule.

## Per-commit archiving (required)

Every full set is archived to `~/capture-baselines/<date>-<commit>/`:

- runner `run-{on,off}.json` (contains `_source_identity()`: git revision +
  worktree diff sha256 — a dirty tree is visible in the archive),
- trace-mode `.nsys-rep` + `.sqlite`,
- matrix JSONL when the matrix ran,
- the `nsys_report.py --json` output for the pair.

Levers land serially, so the previous lever's post-change arms are the next
lever's pre-change baseline: steady-state marginal cost per lever is roughly
matrix-post 12 legs + bare-post 2 + trace-post 2 (~1.5 h on one GPU, ~40 min
sharded across four).

The dataset files themselves are sampled from private SpecLoop traffic and
stay out of this repo; `run.json` pins their sha256/row-count/length
distribution so any two archives can be checked for same-workload.

## Reading a before/after pair

```bash
# symptom (bare mode legs already print this at the end of an on+off run)
python3 bench_capture_attribution.py --report new/run-on.json new/run-off.json

# mechanism
python3 nsys_report.py new/serving-on.sqlite new/serving-off.sqlite \
    --meta-a new/run-on.json --meta-b new/run-off.json --json pair-report.json
```

Acceptance gates for issue #10 lever 2 (delete two-phase verify D2H), in
nsys_report.py terms:

- `capture_worker_rows`: the `hcap-d2h-launch` thread disappears (zero API
  rows attributed to it),
- `api_delta`: `cudaEventQuery` delta drops by the launcher's share,
- `kernel_parity`: still within tolerance (GPU stays innocent),
- `dtoh`: bytes may RISE (upper-bound copy is priced in; the explicit
  non-goal is optimizing bytes back down),
- bare mode: TTFT paired gap at 16-32k does not regress.

## History

The 2026-08-20 attribution numbers (+5,282 cudaMemcpyAsync / +32,821
cudaEventQuery / 11.9 GB D2H / 52 GB/s / "0.7%") were produced by an ad-hoc
session; `nsys_report.py`'s docstring records the pinned re-definitions and
the verification against the archived sqlite pair in
`~/nsys-baseline-20260820/`.  Two corrections to the published narrative made
by that verification: the +32,821 eventQuery polls were mostly on the two
capture worker threads, not the scheduler thread (whose own delta was ~+0.4k),
and 52 GB/s describes only the >=1 MiB payload copies (all-copy aggregate is
~49 GB/s).  Counts from the old sqlite pair are not comparable with
bench_serving-driven runs (different client protocol); they are kept as
reader-verification material only.
