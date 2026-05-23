"""Statistics for the GRN-quality study.

Resampling units must be independent. Conditions cluster by drug / cell-line (audit C), so the
naive condition bootstrap underestimates CIs. We provide a **cluster bootstrap** that resamples
whole groups (clusters) with replacement, a **paired cluster bootstrap** for variant contrasts,
**Holm-Bonferroni** correction across the planned contrasts (audit D), and an **effect-size
summary** that separates statistical significance from a pre-registered minimum meaningful
effect (audit G).
"""
from __future__ import annotations

import numpy as np

from pmoe.config import SEED, MIN_MEANINGFUL_EFFECT


def bootstrap_ci(values, n_boot: int = 2000, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile CI of a per-condition metric, resampling CONDITIONS.

    Returns ``(mean, lo, hi)``.
    """
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def _group_members(values: np.ndarray, groups: np.ndarray):
    """Drop non-finite entries, then return (unique_groups, list-of-member-value-arrays)."""
    finite = np.isfinite(values)
    v = values[finite]
    g = groups[finite]
    uniq = np.unique(g)
    members = [v[g == u] for u in uniq]
    return uniq, members


def cluster_bootstrap_ci(values, groups, n_boot: int = 2000, seed: int = SEED,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Cluster bootstrap CI (audit C): resample whole GROUPS with replacement, then recompute the
    overall mean over the pooled members of the resampled groups.

    ``values`` and ``groups`` are aligned per-condition arrays. Returns ``(mean, lo, hi)``.
    Because correlated conditions move together within a cluster, this yields wider (honest) CIs
    than :func:`bootstrap_ci` when groups are internally correlated.
    """
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    uniq, members = _group_members(values, groups)
    n_groups = len(uniq)
    if n_groups == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.concatenate(members).mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        pooled = np.concatenate([members[j] for j in pick])
        boots[b] = pooled.mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def paired_cluster_bootstrap(a, b, groups, n_boot: int = 2000, seed: int = SEED,
                             alpha: float = 0.05) -> dict:
    """Paired cluster bootstrap for the contrast ``delta = mean(a) - mean(b)`` (audit C/D).

    ``a`` and ``b`` are aligned per-condition metric arrays for two variants on the SAME test set;
    ``groups`` is the per-condition cluster label. Resamples whole groups with replacement and
    recomputes the paired delta over the pooled members.

    Returns ``dict(delta, ci=(lo, hi), p)`` where ``p`` is the two-sided bootstrap p-value.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    groups = np.asarray(groups)
    m = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[m], b[m], groups[m]
    d = a - b
    uniq = np.unique(g)
    members = [d[g == u] for u in uniq]
    n_groups = len(uniq)
    if n_groups == 0:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")), "p": float("nan")}
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        boots[i] = np.concatenate([members[j] for j in pick]).mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided p: fraction of bootstrap deltas on the opposite side of 0, doubled.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta": delta, "ci": (float(lo), float(hi)), "p": float(min(1.0, p))}


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction (audit D) across a family of contrasts.

    Returns a dict (same keys) of corrected p-values. The correction is monotone in rank and each
    corrected p is >= its raw p (and clipped to <= 1.0).
    """
    keys = list(pvalues.keys())
    raw = np.asarray([pvalues[k] for k in keys], np.float64)
    n = len(keys)
    if n == 0:
        return {}
    order = np.argsort(raw, kind="stable")           # ascending raw p
    corrected = np.empty(n, np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * raw[idx]                   # Holm step-down multiplier
        running = max(running, adj)                   # enforce monotonicity
        corrected[idx] = min(1.0, running)
    return {k: float(corrected[i]) for i, k in enumerate(keys)}


def effect_summary(delta: float, ci, p_corrected: float,
                   min_effect: float = MIN_MEANINGFUL_EFFECT) -> dict:
    """Combine significance and effect size (audit G).

    A result is reported only when it is both statistically significant (``p_corrected < 0.05``)
    AND practically meaningful (``|delta| >= min_effect``). Returns booleans plus the inputs.
    """
    significant = bool(p_corrected < 0.05)
    meaningful = bool(abs(delta) >= min_effect)
    return {
        "delta": float(delta),
        "ci": (float(ci[0]), float(ci[1])),
        "p_corrected": float(p_corrected),
        "min_effect": float(min_effect),
        "significant": significant,
        "meaningful": meaningful,
        "significant_and_meaningful": significant and meaningful,
    }
