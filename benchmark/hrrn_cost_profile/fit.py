"""Fit the prefill forward cost model from SGLANG_FORWARD_COST_PROFILE_DIR samples.

One sample = one ModelRunner.forward on one DP rank (see
python/sglang/srt/observability/forward_cost_profiler.py):

    T = a + b * sum(n_i) + c * sum(n_i * p_i + n_i^2 / 2)

n_i = extend tokens, p_i = cached prefix tokens of request i in the batch.
Only full-model EXTEND forwards are fitted; every other sample is kept in the
input files and counted under "coverage".

Usage:
    python fit.py --since <unix_s> --out report.json /path/to/dir/forward_cost.dp*.jsonl
"""

import argparse
import json
from collections import Counter

import numpy as np
from scipy import optimize, stats

MODELS = {"M0": ["b"], "M1": ["a", "b", "c"]}


def load(paths):
    records = []
    for path in paths:
        with open(path) as f:
            records.extend(json.loads(line) for line in f)
    return records


def select(records, since, *, queued_only=False):
    """Return (fit samples as arrays, coverage counts by category)."""
    coverage = Counter()
    rows = []
    for rec in records:
        if rec["failed"]:
            coverage["failed"] += 1
            continue
        if rec["launch_unix_s"] < since:
            coverage["before_since"] += 1
            continue
        shape = rec["shape"]
        mode = shape["forward_mode"]
        if mode != "EXTEND":
            # IDLE / DECODE / MIXED / SPLIT_PREFILL etc. time a different workload.
            coverage[mode] += 1
            continue
        n, p = shape["extend_seq_lens"], shape["extend_prefix_lens"]
        if n is None or p is None:
            raise ValueError("EXTEND sample without CPU extend lengths")
        if not len(n) == len(p) == shape["batch_size"]:
            raise ValueError(f"inconsistent EXTEND shape: {shape}")
        if queued_only:
            waiting_after = shape["waiting_reqs_after"]
            if waiting_after is None:
                coverage["waiting_unknown"] += 1
                continue
            if waiting_after <= 0:
                coverage["not_queued"] += 1
                continue
        n, p = np.asarray(n, dtype=float), np.asarray(p, dtype=float)
        g = shape["global_num_tokens"]
        rows.append(
            (
                rec["dp_rank"],
                len(n),
                n.sum(),
                (n * p + n * n / 2).sum(),
                # Tokens by which the busiest DP rank exceeds this one; DP diagnostic only.
                max(g) - n.sum() if g is not None else np.nan,
                rec["elapsed_s"],
                n.max(),
                p.max(),
            )
        )
        coverage["fit"] += 1
    cols = np.array(rows, dtype=float).reshape(-1, 8).T
    names = ("dp_rank", "k", "sum_n", "sum_attn", "peer_excess", "T", "max_n", "max_p")
    return dict(zip(names, cols)), dict(coverage)


def design(terms, s):
    columns = {"a": np.ones_like(s["T"]), "b": s["sum_n"], "c": s["sum_attn"]}
    return np.column_stack([columns[t] for t in terms])


def fit_nnls(X, y):
    """NNLS on relative error (weight 1/T^2). Columns are scaled first: the
    weighted matrix reaches condition number ~1e9 on a wide shape range."""
    scale = X.max(axis=0)
    return optimize.nnls(X / scale / y[:, None], np.ones_like(y))[0] / scale


def identifiable(X):
    """Unique coefficients: full column rank (an exactly determined system qualifies)."""
    return len(X) > 0 and np.linalg.matrix_rank(X / X.max(axis=0)) == X.shape[1]


def cross_validate(terms, s, folds, seed):
    """K-fold over samples. Held-out predictions are ranked only against other
    held-out samples of the same fold (same fitted coefficients) and same k."""
    X, y = design(terms, s), s["T"]
    folds = min(folds, len(y))
    fold_of = np.empty(len(y), dtype=int)
    pred = np.empty_like(y)
    for f, held in enumerate(
        np.array_split(np.random.default_rng(seed).permutation(len(y)), folds)
    ):
        train = np.setdiff1d(np.arange(len(y)), held)
        if not identifiable(X[train]):
            return None
        pred[held] = X[held] @ fit_nnls(X[train], y[train])
        fold_of[held] = f
    # Positive = measured slower than the model predicts.
    rel_residual = (y - pred) / y
    return {
        "mean_abs_rel_err": float(np.mean(np.abs(rel_residual))),
        "p90_abs_rel_err": float(np.percentile(np.abs(rel_residual), 90)),
        **rank_metrics(pred, y, groups=np.stack([fold_of, s["k"]], axis=1)),
        "dp_residual_vs_peer_excess": dp_residual(rel_residual, s["peer_excess"]),
    }


def rank_metrics(pred, y, groups):
    """Mean Spearman / Kendall over groups with >= 2 samples and non-constant
    pred and y; None when no group qualifies."""
    rhos, taus = [], []
    for g in np.unique(groups, axis=0):
        m = (groups == g).all(axis=1)
        if m.sum() < 2 or np.ptp(pred[m]) == 0 or np.ptp(y[m]) == 0:
            continue
        rhos.append(stats.spearmanr(pred[m], y[m])[0])
        taus.append(stats.kendalltau(pred[m], y[m])[0])
    return {
        "spearman_within_fold_k": float(np.mean(rhos)) if rhos else None,
        "kendall_within_fold_k": float(np.mean(taus)) if taus else None,
        "num_rank_groups": len(rhos),
    }


def dp_residual(rel_residual, peer_excess):
    # Diagnostic only: correlation is not evidence that peers cause the residual.
    m = ~np.isnan(peer_excess)
    if m.sum() < 2 or np.ptp(peer_excess[m]) == 0 or np.ptp(rel_residual[m]) == 0:
        return None
    return {
        "spearman": float(stats.spearmanr(peer_excess[m], rel_residual[m])[0]),
        "num_samples": int(m.sum()),
    }


def build_report(records, since, folds, seed, *, queued_only=False):
    s, coverage = select(records, since, queued_only=queued_only)
    report = {
        "filters": {
            "since": since,
            "failed": False,
            "forward_mode": "EXTEND",
            "queue_condition": "waiting_reqs_after > 0" if queued_only else None,
        },
        "coverage": coverage,
    }
    if coverage.get("fit", 0):
        report["fit_range"] = {
            "dp_ranks": sorted({int(r) for r in s["dp_rank"]}),
            "k": [int(s["k"].min()), int(s["k"].max())],
            "max_request_n": int(s["max_n"].max()),
            "max_request_p": int(s["max_p"].max()),
            "T_s": [float(s["T"].min()), float(s["T"].max())],
        }
    report["models"] = {}
    for name, terms in MODELS.items():
        X = design(terms, s)
        if not identifiable(X):
            report["models"][name] = {
                "error": "terms not identifiable on these samples"
            }
            continue
        report["models"][name] = {
            "coefs": dict(zip(terms, fit_nnls(X, s["T"]).tolist())),
            "cv": cross_validate(terms, s, folds, seed),
        }
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="+", help="forward_cost.dp*.jsonl files")
    ap.add_argument(
        "--since",
        type=float,
        required=True,
        help="Unix time; earlier samples (startup warmup, cold caches) are not fitted",
    )
    ap.add_argument(
        "--queued-only",
        action="store_true",
        help="Fit only batches with waiting_reqs_after > 0; exclude unknown queue state",
    )
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = build_report(
        load(args.files),
        args.since,
        args.folds,
        args.seed,
        queued_only=args.queued_only,
    )
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
