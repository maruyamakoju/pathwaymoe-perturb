"""Unit tests for pmoe.eval.metrics (self-contained, CPU-only, no real data)."""
from __future__ import annotations

import numpy as np

from pmoe.eval.metrics import compute_metrics, deg_pearson_per_condition


def _make_data(seed=1337, n=8, N=2000, n_strong=100):
    rng = np.random.default_rng(seed)
    true = rng.standard_normal((n, N)).astype(np.float32)
    true[:, :n_strong] *= 4.0                     # planted strong DEGs
    pred_good = true + 0.3 * rng.standard_normal((n, N)).astype(np.float32)
    pred_zero = np.zeros_like(true)
    return true, pred_good, pred_zero


def test_good_pred_beats_zero_pred_deg50():
    true, pred_good, pred_zero = _make_data()
    good = compute_metrics(pred_good, true)
    zero = compute_metrics(pred_zero, true)
    assert good["pearson_deg50"] > zero["pearson_deg50"]
    # zero prediction is constant -> Pearson defined as 0.0
    assert abs(zero["pearson_deg50"]) < 1e-9
    assert good["pearson_deg50"] > 0.8


def test_compute_metrics_keys():
    true, pred_good, _ = _make_data()
    m = compute_metrics(pred_good, true)
    for k in ("pearson_all", "mse",
              "pearson_deg20", "pearson_deg50", "pearson_deg100",
              "direction_acc20", "direction_acc50", "direction_acc100"):
        assert k in m
        assert np.isfinite(m[k])


def test_compute_metrics_empty_input_returns_nan_dict():
    """Regression for the v1 IndexError on empty test sets.

    When an analysis filters out all conditions, compute_metrics should return a NaN-filled
    dict with the canonical key schema -- not crash with `rows[0].keys()` IndexError.
    """
    from pmoe.eval.metrics import METRIC_KEYS
    out = compute_metrics(np.empty((0, 20)), np.empty((0, 20)))
    assert set(out.keys()) == set(METRIC_KEYS)
    for k, v in out.items():
        assert np.isnan(v), f"{k}={v} should be NaN on empty input"


def test_deg_pearson_per_condition_shape_and_topk():
    true, pred_good, _ = _make_data()
    arr = deg_pearson_per_condition(pred_good, true, k=50)
    assert arr.shape == (true.shape[0],)
    assert np.all(np.isfinite(arr))
    # Per-condition mean matches the aggregate compute_metrics value.
    assert abs(np.nanmean(arr) - compute_metrics(pred_good, true)["pearson_deg50"]) < 1e-9
