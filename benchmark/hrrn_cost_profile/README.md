# HRRN prefill cost profiling

This tool calibrates the per-forward prefill cost of one deployment on its real P runtime:

    T = a + b·Σn_i + c·Σ(n_i·p_i + n_i²/2)

Here n_i is the extend (uncached) token count of request i in the batch, and p_i is its cached prefix. Batches may mix requests of any shape. Recalibrate after any change to model, hardware, parallelism, or backend. This round does not touch HRRN ordering, aging, or parameter wiring.

## What is measured

Set `SGLANG_FORWARD_COST_PROFILE_DIR=<dir>` on the prefill servers. Each `ModelRunner.forward` call on each DP rank then becomes one JSON line in `<dir>/forward_cost.dp<rank>.jsonl` (`python/sglang/srt/observability/forward_cost_profiler.py`). Each line has:

- `elapsed_s`, which is T
- `launch_unix_s`
- `failed`: the forward raised (the exception still propagates); its partial interval is never fitted
- `dp_rank`
- the batch `shape`, snapshotted at scope entry:
  - `forward_mode`
  - `batch_size`
  - per-request `extend_seq_lens` and `extend_prefix_lens`
  - the existing DP-gather `global_num_tokens`
  - `split_index` / `split_forward_count`
  - `waiting_reqs_before` / `waiting_reqs_after`: waiting queue counts snapshotted by the scheduler before sorting/admission and after admission removes selected requests (including any preempted requests returned to waiting). These travel read-only with the batch; event harvesting never reads the live queue. `null` means no admission snapshot is available, not zero waiters.

These counts describe the waiting queue, not token-budget utilization or remaining chunk work. `PrefillStats.num_pending_tokens` also includes the uncomputed remainder of a chunked request and is **not** a waiting-queue predicate. A lone continuing chunk can have pending tokens with `before=after=0`; several waiting requests can all enter this batch with `before>0, after=0`. Therefore `after=0` does not mean the batch started without waiting requests. No queue-based filtering happens during collection.

**T** is the elapsed time between two CUDA events, both recorded on the stream current at `ModelRunner.forward`'s profiling scope: one at entry, one at exit. The scope wraps `_forward_raw`, so every execution path is measured with the same boundaries:

- eager, prefill CUDA graph, and decode CUDA graph paths
- batch preparation and H2D (`_prepare_eager_forward_batch`, `load_batch`, attention metadata init)
- the model forward

**T includes** launch bubbles and dependency or collective waits inside the interval. Under DP attention this includes waiting for slower ranks at the MoE all-to-all. It also includes contention from KV transfer and HiCache, but only where that work overlaps or blocks this stream inside the interval.

**T is not** exclusive GPU time, and it is not one request's marginal blocking time. It excludes sampling (done after `forward`), scheduling, queueing, HTTP, and transfer time outside the interval. Work on side streams counts only through dependencies the runtime already creates. No synchronization is added for measurement, so T does not claim to end at the last kernel of every stream.

With overlap scheduling, the start event may wait for the previous batch, so T reflects this batch's stream occupancy. Without overlap, T also contains this batch's CPU preparation. Always profile with the deployment's own overlap setting.

**Lifecycle.** Samples are harvested asynchronously when a later forward exits its scope and on every scheduler idle iteration, and the file is flushed on idle. A graceful shutdown synchronizes once and drains the tail. An exception or kill path does not synchronize and writes only completed samples. The profiler logs a warning with the number of unwritten samples. If an event query raises on a faulted device, the file is still closed and the error propagates. The existing occupancy/FPM timer and its metrics are unchanged.

**Scope.** Supported: fixed-size DP/EP, no PP, no elastic EP. The server refuses to start with PP or elastic EP and this variable set. Only `attn_tp_rank == 0` and `attn_cp_rank == 0` of each DP rank writes. A file that already exists is an error, so use a fresh directory per run. The profiler uses `torch.cuda.Event` and needs a CUDA device.

## Run

1. Launch the target PD deployment unchanged, adding `SGLANG_FORWARD_COST_PROFILE_DIR` on the prefill instances. Keep its scheduling policy, overlap, chunking, cache, and parallelism settings; enabling HRRN is not required.
2. Note the start time with `date +%s`. Send representative traffic through the normal entrypoint using the existing benchmark client, for example:

       python -m sglang.benchmark.serving --backend sglang --base-url <router> \
           --dataset-name <dataset matching production prompt/prefix mix> \
           --num-prompts <N> --request-rate <rate>

   Focus on representative real load with requests still waiting after admission, covering different batch compositions, prompt/prefix mixes, and concurrency levels. Shared-prefix traffic is needed to exercise p > 0. Queueing does not imply a full token budget. The fit is valid only inside the shape range and load the traffic covers, and `fit_range` in the report states that range.
3. Stop traffic, wait for the servers to go idle so the tail is written, and shut down gracefully.
4. Fit:

       python fit.py --since <start time> --queued-only --out queued.json <dir>/forward_cost.dp*.jsonl

   `--queued-only` selects `waiting_reqs_after > 0`: requests remained waiting for a later batch after this admission. Unknown (`null`) queue state is excluded separately. Omit it for an all-eligible-samples comparison using the same raw files:

       python fit.py --since <start time> --out all.json <dir>/forward_cost.dp*.jsonl

   Both raw counts are retained for other future filters. The queue filter does not change M0/M1 or require dropping the constant term.

## What fit.py reports

- `coverage`: sample counts by category. Only successful full-model `EXTEND` samples at or after `--since` are fitted; `failed` samples are counted separately. `SPLIT_PREFILL` (partial layers), `MIXED`, `DECODE`, `IDLE` and other modes are counted and kept in the raw files, never mixed into the regression. An `EXTEND` sample without CPU lengths, or with inconsistent lengths, is an error.
- `filters`: the cutoff, success/mode requirements, and `queue_condition` (`waiting_reqs_after > 0`, or `null` with no queue filter). After the success/cutoff/mode and EXTEND-shape checks, `--queued-only` counts excluded known-empty queues under `coverage.not_queued` and unknown queues under `coverage.waiting_unknown`; `coverage.fit` counts only selected samples. Default analysis keeps both known-empty and unknown queues.
- `fit_range`: the k, n, p and T range that the fitted samples span.
- `models`:
  - M0 (`b` only, the baseline) and M1 (`a, b, c`), fitted by NNLS on relative error.
  - A model whose columns are linearly dependent on the samples (or with no samples) reports an error instead of coefficients. An exactly determined fit still reports coefficients; `cv` is `null` when some training split is not full rank.
  - `cv` reports K-fold held-out `mean_abs_rel_err` and `p90_abs_rel_err`.
  - Ranking metrics (Spearman / Kendall) are computed only among held-out samples of the same fold and the same k, then averaged. Comparisons therefore always come from one fitted model on comparable batches. Groups where a metric is undefined are skipped, and the metric is `null` when no group qualifies.
  - `dp_residual_vs_peer_excess` is the Spearman correlation between the relative residual (positive means slower than predicted) and `max(global_num_tokens) − Σn`. It is diagnostic evidence only. It neither proves that peer ranks cause the residual nor changes the model.

## Not verified (no GPU run yet)

- Real CUDA event timing and the overhead of profiling on the target P deployment.
- Whether the prefill CUDA graph is used there, and how T behaves with the deployment's overlap setting.
- How much KV transfer and HiCache overlap with the measured interval.
- Model fit quality on real data.

## Tests

    python -m pytest benchmark/hrrn_cost_profile                       # fit (numpy, scipy)
    python -m pytest test/registered/unit/observability/test_forward_cost_profiler.py
    python -m pytest test/registered/unit/managers/test_prefill_adder.py
