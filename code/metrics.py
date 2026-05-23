"""Evaluation metrics for perturbation LFC prediction.

The field has converged on DEG-focused metrics (Virtual Cell Challenge, Nature Methods 2025
baseline paper) because gene-wise Pearson over the full transcriptome is dominated by
housekeeping genes that don't move. We report all four metric families.

DEG definition (per condition): a gene is a DEG if |true_lfc| > LFC_THRESH AND it passed the
significance filter recorded in `deg_mask` during preprocessing (|true|>0.5 & padj<0.05).
Top-K DEGs = the K DEGs with largest |true_lfc|.
"""
from __future__ import annotations
import numpy as np

LFC_THRESH = 0.5
DEG_KS = (20, 50, 100)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.size < 2:
        return np.nan
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _condition_metrics(pred: np.ndarray, true: np.ndarray, deg_mask: np.ndarray | None) -> dict:
    out = {"pearson_all": _pearson(pred, true), "mse": float(np.mean((pred - true) ** 2))}
    # candidate DEG set
    if deg_mask is None:
        cand = np.abs(true) > LFC_THRESH
    else:
        cand = deg_mask.astype(bool) & (np.abs(true) > LFC_THRESH)
    cand_idx = np.where(cand)[0]
    order = cand_idx[np.argsort(-np.abs(true[cand_idx]))]  # by |true_lfc| desc
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


def compute_metrics(pred_lfc: np.ndarray, true_lfc: np.ndarray,
                    deg_mask: np.ndarray | None = None) -> dict:
    """pred/true: (n_conditions, N). deg_mask: (n_conditions, N) bool or None.
    Returns nan-mean over conditions for each metric."""
    pred_lfc = np.atleast_2d(pred_lfc); true_lfc = np.atleast_2d(true_lfc)
    rows = []
    for i in range(true_lfc.shape[0]):
        dm = None if deg_mask is None else deg_mask[i]
        rows.append(_condition_metrics(pred_lfc[i], true_lfc[i], dm))
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def evaluate(pred: np.ndarray, true: np.ndarray, deg_mask: np.ndarray | None = None,
             groups: np.ndarray | None = None) -> dict:
    """Overall metrics + per-group breakdown (groups = e.g. cell_line labels)."""
    res = {"overall": compute_metrics(pred, true, deg_mask)}
    if groups is not None:
        groups = np.asarray(groups)
        res["by_group"] = {}
        for g in np.unique(groups):
            m = groups == g
            dm = None if deg_mask is None else deg_mask[m]
            res["by_group"][str(g)] = compute_metrics(pred[m], true[m], dm)
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(1337)
    n, N = 8, 2000
    true = rng.standard_normal((n, N)).astype(np.float32)
    true[:, :100] *= 4  # some strong DEGs
    pred_good = true + 0.3 * rng.standard_normal((n, N)).astype(np.float32)
    pred_zero = np.zeros_like(true)
    dm = np.abs(true) > 0.5
    print("good :", {k: round(v, 3) for k, v in compute_metrics(pred_good, true, dm).items()})
    print("zero :", {k: round(v, 3) for k, v in compute_metrics(pred_zero, true, dm).items()})
