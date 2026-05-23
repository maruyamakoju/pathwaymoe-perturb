"""Headline regression test for AUDIT item A: data-derived GRNs must NOT see the test set.

Strategy:
  1. Build a tiny synthetic conditions DataFrame (no real dataset needed) by passing genes= and df=.
  2. Build coexpr_lfc from train_idx -> record edge set.
  3. Drastically MUTATE the test-set rows' lfc, rebuild from the SAME train_idx.
  4. Assert the edge set is IDENTICAL (proves test rows never entered the GRN).
Also assert build_grn(coexpr_lfc, train_idx=None) raises ValueError (the leakage guard).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pmoe.priors.grn import build_grn
from pmoe.config import GRNVariant

N_GENES = 50
N_COND = 40
DATASET = "synthetic_smoke"   # only used for path/IO of the tiny artifact


def _fake_df(seed=0):
    """A minimal conditions DataFrame with lfc/pert_mean list columns."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_COND):
        rows.append({
            "treatment": f"drug{i % 7}",
            "lfc": rng.standard_normal(N_GENES).astype(np.float32),
            "pert_mean": rng.standard_normal(N_GENES).astype(np.float32),
        })
    return pd.DataFrame(rows)


def _edge_set(m: sp.csr_matrix) -> set:
    coo = m.tocoo()
    return set(zip(coo.row.tolist(), coo.col.tolist()))


def test_coexpr_lfc_ignores_test_rows():
    genes = [f"G{i:05d}" for i in range(N_GENES)]
    df = _fake_df(seed=1)

    idx = np.arange(N_COND)
    train_idx = idx[: int(N_COND * 0.6)]
    test_idx = idx[int(N_COND * 0.6):]

    m1 = build_grn(DATASET, GRNVariant.COEXPR_LFC, train_idx=train_idx, df=df,
                   genes=genes, split="unseen_drug")
    edges_before = _edge_set(m1)
    assert edges_before, "expected a non-empty coexpr_lfc GRN"

    # Mutate ONLY the test rows' lfc, drastically.
    df2 = df.copy()
    big = np.arange(N_GENES, dtype=np.float32) * 1000.0 + 12345.0
    for t in test_idx:
        df2.at[t, "lfc"] = big.copy() * (1 + t)

    m2 = build_grn(DATASET, GRNVariant.COEXPR_LFC, train_idx=train_idx, df=df2,
                   genes=genes, split="unseen_drug")
    edges_after = _edge_set(m2)

    assert edges_after == edges_before, (
        "GRN changed after mutating ONLY test-set rows -> test leakage!"
    )


def test_coexpr_train_idx_none_raises():
    genes = [f"G{i:05d}" for i in range(N_GENES)]
    df = _fake_df(seed=2)
    with pytest.raises(ValueError):
        build_grn(DATASET, GRNVariant.COEXPR_LFC, train_idx=None, df=df, genes=genes)
    with pytest.raises(ValueError):
        build_grn(DATASET, GRNVariant.COEXPR, train_idx=None, df=df, genes=genes)
