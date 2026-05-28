"""Evaluation metrics for perturbation LFC prediction (ported from ``code/metrics.py``).

The field has converged on DEG-focused metrics because gene-wise Pearson over the full
transcriptome is dominated by housekeeping genes that do not move. We report four metric
families: full-transcriptome Pearson, DEG-Pearson@{20,50,100}, MSE, and direction accuracy.

DEG-Pearson@K ranks **all** genes by ``|true_lfc|`` and keeps the top-K (the biggest movers).
``deg_mask`` is retained in the schema as a record of significant DEGs but is not used to gate
the metric here (matches the corrected v1 behaviour).
"""
from __future__ import annotations

import numpy as np

DEG_KS = (20, 50, 100)
METRIC_KEYS = ("pearson_all", "mse",
               "pearson_deg20", "pearson_deg50", "pearson_deg100",
               "direction_acc20", "direction_acc50", "direction_acc100")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either input is (near-)constant, NaN when too small."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size < 2:
        return np.nan
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def deg_pearson_per_condition(pred: np.ndarray, true: np.ndarray, k: int = 50) -> np.ndarray:
    """Per-condition DEG-Pearson@k: rank all genes by ``|true_lfc|``, keep top-k, correlate.

    ``pred``/``true`` are (n_conditions, N). Returns a (n_conditions,) float64 array.
    Argsort uses ``kind="stable"`` so the top-k selection is reproducible when
    multiple genes tie in absolute LFC.
    """
    pred = np.atleast_2d(pred)
    true = np.atleast_2d(true)
    out = np.empty(true.shape[0], dtype=np.float64)
    for i in range(true.shape[0]):
        order = np.argsort(-np.abs(true[i]), kind="stable")[:k]
        if order.size >= 2:
            out[i] = _pearson(pred[i][order], true[i][order])
        else:
            out[i] = np.nan
    return out


def _condition_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    out = {"pearson_all": _pearson(pred, true), "mse": float(np.mean((pred - true) ** 2))}
    # DEG-Pearson@K = Pearson on the K genes with largest |true_lfc| (the biggest movers).
    order = np.argsort(-np.abs(true), kind="stable")
    for k in DEG_KS:
        sel = order[:k]
        if sel.size >= 2:
            out[f"pearson_deg{k}"] = _pearson(pred[sel], true[sel])
            same = np.sign(pred[sel]) == np.sign(true[sel])
            out[f"direction_acc{k}"] = float(np.mean(same))
        else:
            out[f"pearson_deg{k}"] = np.nan
            out[f"direction_acc{k}"] = np.nan
    return out


def compute_metrics(pred: np.ndarray, true: np.ndarray,
                    deg_mask: np.ndarray | None = None) -> dict:
    """pred/true: (n_conditions, N). ``deg_mask`` is accepted for schema compatibility but
    is unused — the DEG metrics are gated by top-K |true_lfc|, not by the supplied mask.

    Returns the nan-mean over conditions for each metric in :data:`METRIC_KEYS`. On empty
    input (``n_conditions == 0``) returns a NaN-filled dict with the same key schema
    rather than raising — important when an analysis filters out all conditions.
    """
    del deg_mask  # unused; see docstring
    pred = np.atleast_2d(pred)
    true = np.atleast_2d(true)
    if true.shape[0] == 0:
        return {k: float("nan") for k in METRIC_KEYS}
    rows = [_condition_metrics(pred[i], true[i]) for i in range(true.shape[0])]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in METRIC_KEYS}
