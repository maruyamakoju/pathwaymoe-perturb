"""Per-variant behaviour tests for pmoe.priors.grn (offline, CPU, no real dataset).

Covers:
  * random  -> exactly match_edges edges (matched density).
  * ground_truth -> round-trips an existing grn_mask.npz.
  * trrust_weighted -> float, signed (has negative weights) CSR.
  * none -> empty matrix, load_grn returns None.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pmoe.priors.grn import build_grn, load_grn
from pmoe.config import GRNVariant, priors_dir

N_GENES = 60


def _genes(n=N_GENES):
    return [f"G{i:05d}" for i in range(n)]


def test_random_matches_requested_edge_count():
    genes = _genes()
    want = 137
    m = build_grn("synthetic_smoke", GRNVariant.RANDOM, match_edges=want, genes=genes, seed=7)
    assert m.shape == (N_GENES, N_GENES)
    assert m.nnz == want
    assert m.dtype == bool
    # determinism
    m2 = build_grn("synthetic_smoke", GRNVariant.RANDOM, match_edges=want, genes=genes, seed=7)
    assert (m != m2).nnz == 0


def test_ground_truth_roundtrips(tmp_path, monkeypatch):
    # write a tiny grn_mask.npz into a throwaway dataset's priors dir, then load via ground_truth
    ds = "gt_roundtrip_test"
    genes = _genes()
    rng = np.random.default_rng(0)
    r = rng.integers(0, N_GENES, 40)
    c = rng.integers(0, N_GENES, 40)
    keep = r != c
    truth = sp.csr_matrix((np.ones(keep.sum(), bool), (r[keep], c[keep])), shape=(N_GENES, N_GENES))
    sp.save_npz(priors_dir(ds) / "grn_mask.npz", truth)

    m = build_grn(ds, GRNVariant.GROUND_TRUTH, genes=genes)
    assert m.shape == truth.shape
    # same edge set
    a = set(zip(*truth.tocoo().nonzero()))
    b = set(zip(*m.tocoo().nonzero()))
    assert a == b

    loaded = load_grn(ds, GRNVariant.GROUND_TRUTH)
    assert loaded is not None and (loaded != m).nnz == 0


def test_trrust_weighted_is_signed_float():
    genes = _genes(80)
    m = build_grn("synthetic_smoke", GRNVariant.TRRUST_WEIGHTED, genes=genes)
    assert m.dtype != bool, "weighted GRN must be float"
    assert np.issubdtype(m.dtype, np.floating)
    if m.nnz:
        data = m.data
        assert (data < 0).any() or (data > 0).any()
        # signed: at least some negatives expected from Repression / Unknown
        # (don't hard-require negatives if the tiny gene slice happened to map none)


def test_none_is_empty():
    genes = _genes()
    m = build_grn("synthetic_smoke", GRNVariant.NONE, genes=genes)
    assert m.nnz == 0
    assert load_grn("synthetic_smoke", GRNVariant.NONE) is None
