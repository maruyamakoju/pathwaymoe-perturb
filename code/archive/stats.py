"""Statistics for the GRN-quality study: per-condition metrics, bootstrap CIs, and paired
bootstrap significance tests. All resampling is over CONDITIONS (the independent test units)."""
from __future__ import annotations
import numpy as np
from metrics import _pearson, DEG_KS


def per_condition_deg_pearson(pred, true, k=50):
    """Array of per-condition DEG-Pearson@k (top-k genes by |true_lfc|)."""
    out = np.empty(len(true), np.float64)
    for i in range(len(true)):
        order = np.argsort(-np.abs(true[i]))[:k]
        out[i] = _pearson(pred[i][order], true[i][order])
    return out


def bootstrap_ci(values, n_boot=2000, seed=1337, alpha=0.05):
    """Mean and (1-alpha) percentile CI of a per-condition metric, resampling conditions."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(values_a, values_b, n_boot=2000, seed=1337):
    """Paired bootstrap over conditions for delta = a - b. Returns (mean_delta, lo, hi, p_two_sided).
    a and b must be aligned per-condition (same test set/order)."""
    a = np.asarray(values_a, np.float64); b = np.asarray(values_b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    mean_d = float(d.mean())
    # two-sided p: fraction of bootstrap means on the opposite side of 0, x2
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean_d, float(lo), float(hi), float(min(1.0, p))


def aggregate_seeds(per_cond_by_seed):
    """per_cond_by_seed: list of per-condition arrays (one per seed, same conditions).
    Returns mean-over-seeds per condition (for stable point estimate) + across-seed mean/std of the
    condition-mean."""
    stacked = np.vstack(per_cond_by_seed)            # (n_seeds, n_cond)
    seed_means = np.nanmean(stacked, axis=1)         # per-seed overall score
    per_cond_mean = np.nanmean(stacked, axis=0)      # averaged over seeds, per condition
    return per_cond_mean, float(seed_means.mean()), float(seed_means.std())
