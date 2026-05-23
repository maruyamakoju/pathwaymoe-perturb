"""Unit tests for pmoe.eval.stats (self-contained, CPU-only, no real data)."""
from __future__ import annotations

import numpy as np

from pmoe.eval.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    paired_cluster_bootstrap,
    holm_correction,
    effect_summary,
)


def _clustered_data(seed=1337, n_groups=20, per_group=30):
    """Strongly intra-cluster-correlated values: a per-group offset dominates tiny within noise."""
    rng = np.random.default_rng(seed)
    values, groups = [], []
    for g in range(n_groups):
        offset = rng.standard_normal() * 1.0          # large between-group variance
        members = offset + 0.01 * rng.standard_normal(per_group)  # tiny within-group variance
        values.append(members)
        groups.append(np.full(per_group, g))
    return np.concatenate(values), np.concatenate(groups)


def test_cluster_bootstrap_wider_than_naive():
    values, groups = _clustered_data()
    _, lo_n, hi_n = bootstrap_ci(values, n_boot=2000, seed=0)
    _, lo_c, hi_c = cluster_bootstrap_ci(values, groups, n_boot=2000, seed=0)
    width_naive = hi_n - lo_n
    width_cluster = hi_c - lo_c
    assert width_cluster > width_naive
    # The effective sample size collapses from n_obs to n_groups, so it should be much wider.
    assert width_cluster > 3 * width_naive


def test_holm_monotone_and_geq_raw():
    raw = {"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.04, "e": 0.5}
    corr = holm_correction(raw)
    # Each corrected p is >= its raw p.
    for k in raw:
        assert corr[k] >= raw[k] - 1e-12
        assert corr[k] <= 1.0 + 1e-12
    # Monotone in raw-p order (step-down enforces non-decreasing corrected p).
    order = sorted(raw, key=lambda k: raw[k])
    seq = [corr[k] for k in order]
    assert all(seq[i] <= seq[i + 1] + 1e-12 for i in range(len(seq) - 1))


def test_holm_smallest_matches_bonferroni():
    raw = {"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.04, "e": 0.5}
    corr = holm_correction(raw)
    assert abs(corr["a"] - min(1.0, 5 * 0.001)) < 1e-12


def test_paired_cluster_bootstrap_a_greater_than_b():
    rng = np.random.default_rng(7)
    n_groups, per_group = 25, 20
    a, b, groups = [], [], []
    for g in range(n_groups):
        base = rng.standard_normal() * 0.5
        a_g = base + 0.1 + 0.01 * rng.standard_normal(per_group)   # a consistently above b
        b_g = base + 0.01 * rng.standard_normal(per_group)
        a.append(a_g); b.append(b_g); groups.append(np.full(per_group, g))
    a = np.concatenate(a); b = np.concatenate(b); groups = np.concatenate(groups)
    res = paired_cluster_bootstrap(a, b, groups, n_boot=2000, seed=0)
    assert res["delta"] > 0
    assert res["p"] < 0.05
    assert res["ci"][0] < res["delta"] < res["ci"][1] or res["ci"][0] <= res["delta"] <= res["ci"][1]


def test_effect_summary_flags():
    # Big, significant delta -> significant_and_meaningful True.
    big = effect_summary(delta=0.05, ci=(0.03, 0.07), p_corrected=0.001)
    assert big["significant"] and big["meaningful"] and big["significant_and_meaningful"]

    # Tiny delta, significant p -> significant but NOT meaningful (audit G).
    tiny = effect_summary(delta=0.002, ci=(0.001, 0.003), p_corrected=0.001)
    assert tiny["significant"] and not tiny["meaningful"]
    assert not tiny["significant_and_meaningful"]

    # Big delta but not significant -> meaningful but not significant.
    nonsig = effect_summary(delta=0.05, ci=(-0.01, 0.11), p_corrected=0.2)
    assert nonsig["meaningful"] and not nonsig["significant"]
    assert not nonsig["significant_and_meaningful"]
