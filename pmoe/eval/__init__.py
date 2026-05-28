"""Evaluation subpackage: DEG-focused metrics, cluster-bootstrap statistics, and the single
canonical model loader (audit C/D/G/I)."""
from __future__ import annotations

from pmoe.eval.metrics import (
    deg_pearson_per_condition,
    compute_metrics,
    DEG_KS,
    METRIC_KEYS,
)
from pmoe.eval.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    paired_cluster_bootstrap,
    holm_correction,
    effect_summary,
)

__all__ = [
    "deg_pearson_per_condition",
    "compute_metrics",
    "DEG_KS",
    "METRIC_KEYS",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "paired_cluster_bootstrap",
    "holm_correction",
    "effect_summary",
]
