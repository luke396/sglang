"""CPU tests for fit.py. Run: python -m pytest benchmark/hrrn_cost_profile"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fit  # noqa: E402

A, B, C = 2e-3, 3e-5, 4e-9


def true_T(n, p, a=A):
    n, p = np.asarray(n, float), np.asarray(p, float)
    return a + B * n.sum() + C * (n * p + n * n / 2).sum()


def record(
    n,
    p,
    T=None,
    *,
    mode="EXTEND",
    t0=100.0,
    global_tokens=None,
    dp_rank=0,
    batch_size=None,
    failed=False,
    waiting_before=None,
    waiting_after=None,
):
    return {
        "launch_unix_s": t0,
        "elapsed_s": true_T(n, p) if T is None else T,
        "dp_rank": dp_rank,
        "failed": failed,
        "shape": {
            "forward_mode": mode,
            "batch_size": len(n) if batch_size is None else batch_size,
            "extend_seq_lens": n,
            "extend_prefix_lens": p,
            "global_num_tokens": global_tokens,
            "split_index": 0,
            "split_forward_count": 1,
            "waiting_reqs_before": waiting_before,
            "waiting_reqs_after": waiting_after,
        },
    }


def hetero_records(m=200, noise=0.0, seed=0, a=A):
    """Mixed batches: k in 1..4, each request with its own n and p."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(m):
        k = int(rng.integers(1, 5))
        n = rng.integers(16, 4096, size=k).tolist()
        p = (rng.integers(0, 64, size=k) * 1024).tolist()
        recs.append(record(n, p, true_T(n, p, a) * (1 + noise * rng.standard_normal())))
    return recs


def test_recovers_coefficients_from_heterogeneous_batches():
    report = fit.build_report(hetero_records(), since=0, folds=5, seed=0)
    coefs = report["models"]["M1"]["coefs"]
    assert [coefs[t] for t in "abc"] == pytest.approx([A, B, C], rel=1e-6)


def test_cv_with_noise_beats_m0_and_ranks_within_fold_and_k():
    models = fit.build_report(
        hetero_records(noise=0.02, seed=1), since=0, folds=5, seed=0
    )["models"]
    m1, m0 = models["M1"]["cv"], models["M0"]["cv"]
    assert (
        m1["mean_abs_rel_err"] < 0.03
        and m1["mean_abs_rel_err"] * 3 < m0["mean_abs_rel_err"]
    )
    assert m1["kendall_within_fold_k"] > 0.9


def test_rank_metrics_never_compare_across_fold_or_k():
    # Each fold's own predictions order its samples perfectly, but fold 0's
    # coefficients predict lower overall. Pooling folds would mis-rank.
    pred = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([20.0, 21.0, 22.0, 1.0, 2.0, 3.0])
    groups = np.array([[0, 1]] * 3 + [[1, 1]] * 3)
    got = fit.rank_metrics(pred, y, groups)
    assert got["spearman_within_fold_k"] == 1.0 and got["num_rank_groups"] == 2
    # Same values split by k inside one fold: still two separate groups.
    assert (
        fit.rank_metrics(pred, y, np.array([[0, 1]] * 3 + [[0, 2]] * 3))[
            "kendall_within_fold_k"
        ]
        == 1.0
    )


def test_cross_validate_ranks_each_fold_with_its_own_coefficients(monkeypatch):
    # Every fold's fit orders its held-out samples correctly, but the folds'
    # coefficients differ by 1e3x, so pooling held-out predictions mis-ranks.
    calls = []

    def fold_dependent_fit(X, y):
        calls.append(None)
        return np.full(X.shape[1], 10.0 ** (3 * len(calls)))

    monkeypatch.setattr(fit, "fit_nnls", fold_dependent_fit)
    recs = [record([n], [0]) for n in range(100, 3100, 100)]
    s, _ = fit.select(recs, since=0)
    cv = fit.cross_validate(fit.MODELS["M0"], s, folds=3, seed=0)
    assert (
        cv["kendall_within_fold_k"] == pytest.approx(1.0) and cv["num_rank_groups"] == 3
    )


def test_rank_metrics_undefined_groups_return_none():
    # Singleton groups and constant predictions (e.g. M0 at fixed n, varying p).
    pred, y = np.array([1.0, 1.0, 5.0]), np.array([1.0, 2.0, 3.0])
    got = fit.rank_metrics(pred, y, np.array([[0, 1], [0, 1], [0, 2]]))
    assert got == {
        "spearman_within_fold_k": None,
        "kendall_within_fold_k": None,
        "num_rank_groups": 0,
    }


def test_only_full_model_extend_after_since_is_fitted():
    recs = hetero_records(m=20) + [
        record([8], [0], t0=50.0),  # warmup, before --since
        record([1], [100], mode="DECODE"),
        record([8, 1], [0, 100], mode="MIXED"),
        record([], [], mode="IDLE"),
        record([64], [0], mode="SPLIT_PREFILL"),  # partial layers
        record([512], [1024], T=1e-6, failed=True),  # forward raised mid-way
    ]
    report = fit.build_report(recs, since=60.0, folds=5, seed=0)
    assert report["coverage"] == {
        "fit": 20,
        "before_since": 1,
        "DECODE": 1,
        "MIXED": 1,
        "IDLE": 1,
        "SPLIT_PREFILL": 1,
        "failed": 1,
    }


@pytest.mark.parametrize(
    "bad",
    [
        record(None, None, T=0.01, batch_size=2),  # gpu_only path, no CPU lens
        record([8, 8], [0, 0], batch_size=3),
    ],
)
@pytest.mark.parametrize("queued_only", [False, True])
def test_malformed_extend_sample_is_an_error(bad, queued_only):
    with pytest.raises(ValueError):
        fit.select([bad], since=0, queued_only=queued_only)


def test_coefficient_identifiability_is_separate_from_cv_sample_count():
    # Empty input: nothing to fit.
    empty = fit.build_report([], since=0, folds=5, seed=0)["models"]
    assert all("error" in m for m in empty.values())
    # Rank-deficient for M1: fixed n with p = 0 makes the b and c columns proportional.
    same = [record([512], [0], T=true_T([512], [0]) * (1 + 0.01 * i)) for i in range(5)]
    models = fit.build_report(same, since=0, folds=2, seed=0)["models"]
    assert "error" in models["M1"] and "coefs" in models["M0"]
    # Exactly determined: three full-rank samples give unique M1 coefficients,
    # but any CV training split has fewer rows than terms, so CV is unavailable.
    three = [record([128], [0]), record([512], [1024]), record([2048], [16384])]
    m1 = fit.build_report(three, since=0, folds=5, seed=0)["models"]["M1"]
    assert [m1["coefs"][t] for t in "abc"] == pytest.approx([A, B, C], rel=1e-6)
    assert m1["cv"] is None


def test_queued_only_selects_remaining_waiters_and_reports_exclusions():
    recs = [
        record([128], [0], waiting_before=3, waiting_after=2),
        record([512], [1024], waiting_before=3, waiting_after=2),
        record([2048], [16384], waiting_before=3, waiting_after=2),
        # A continued long request is not another waiting candidate.
        record([512], [65536], T=0.5, waiting_before=0, waiting_after=0),
        # Several candidates can all fit into this batch.
        record([128, 256], [0, 0], T=0.5, waiting_before=2, waiting_after=0),
        record([128], [0], T=0.5),  # explicitly unknown queue state
        record([128], [0], failed=True),
        record([128], [0], t0=50),
        *[
            record([128], [0], mode=mode)
            for mode in ("IDLE", "DECODE", "MIXED", "SPLIT_PREFILL")
        ],
    ]
    all_samples = fit.build_report(recs, since=60, folds=5, seed=0)
    queued = fit.build_report(recs, since=60, folds=5, seed=0, queued_only=True)
    common = dict(failed=1, before_since=1, IDLE=1, DECODE=1, MIXED=1, SPLIT_PREFILL=1)
    assert all_samples["coverage"] == dict(common, fit=6)
    assert queued["coverage"] == dict(common, fit=3, not_queued=2, waiting_unknown=1)
    assert all_samples["filters"]["queue_condition"] is None
    assert queued["filters"] == dict(
        since=60,
        failed=False,
        forward_mode="EXTEND",
        queue_condition="waiting_reqs_after > 0",
    )
    m1 = queued["models"]["M1"]
    assert [m1["coefs"][t] for t in "abc"] == pytest.approx([A, B, C], rel=1e-6)
    assert m1["cv"] is None
    assert all_samples["models"]["M1"]["coefs"] != m1["coefs"]
    assert all_samples["fit_range"]["max_request_p"] == 65536
    assert queued["fit_range"]["max_request_p"] == 16384
    # Filtering away all samples must preserve the empty-fit error behavior.
    empty = fit.build_report(recs[3:6], since=60, folds=5, seed=0, queued_only=True)
    assert empty["coverage"] == dict(not_queued=2, waiting_unknown=1)
    assert all("error" in m for m in empty["models"].values())


@pytest.mark.parametrize("queued_only", [False, True])
def test_cli_passes_queue_filter_to_fit(tmp_path, monkeypatch, queued_only):
    import json

    source, out = tmp_path / "samples.jsonl", tmp_path / "report.json"
    recs = [
        record([128], [0], waiting_before=2, waiting_after=1),
        record([256], [0], waiting_before=1, waiting_after=0),
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in recs))
    argv = ["fit.py", "--since", "0", "--out", str(out), str(source)]
    if queued_only:
        argv.append("--queued-only")
    monkeypatch.setattr(sys, "argv", argv)
    fit.main()
    report = json.loads(out.read_text())
    assert report["coverage"] == (
        dict(fit=1, not_queued=1) if queued_only else dict(fit=2)
    )
    assert "coefs" in report["models"]["M0"]
    assert "error" in report["models"]["M1"]


def test_dp_residual_is_reported_only_with_global_tokens():
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(60):
        n, p = rng.integers(64, 2048, size=2).tolist(), [0, 4096]
        excess = int(rng.integers(0, 8192))
        # Slower when a peer rank carries more tokens.
        recs.append(
            record(
                n,
                p,
                true_T(n, p) * (1 + excess / 8192),
                global_tokens=[sum(n), sum(n) + excess],
            )
        )
    cv = fit.build_report(recs, since=0, folds=5, seed=0)["models"]["M1"]["cv"]
    assert cv["dp_residual_vs_peer_excess"]["spearman"] > 0.5
    cv = fit.build_report(hetero_records(m=40), since=0, folds=5, seed=0)["models"][
        "M1"
    ]["cv"]
    assert cv["dp_residual_vs_peer_excess"] is None
