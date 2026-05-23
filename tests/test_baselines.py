"""CPU-only unit tests for pmoe.models.baselines (tiny inline synthetic df)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmoe.models.baselines import B1MeanEffect, GRNPropagationBaseline


def _make_synthetic(n_genes=20, n_drugs=8, n_cells=3, seed=0):
    """Build a conditions df whose LFC IS a GRN-propagated signal from each drug's target gene.

    genes are named ``g0..g{N-1}``; each drug targets a distinct gene; LFC = seed + decay * (seed @ P)
    so a pure GRN-propagation predictor should strongly beat a zero predictor.
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{j}" for j in range(n_genes)]

    # random sparse symmetric GRN with float weights
    A = (rng.random((n_genes, n_genes)) < 0.15).astype(np.float64)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    W = A * rng.uniform(0.5, 1.0, size=A.shape)
    grn = sp.csr_matrix(W)

    # row-normalized propagation operator (matches baseline internals)
    deg = W.sum(1, keepdims=True); deg[deg == 0] = 1.0
    P = W / deg

    targets = [f"g{j}" for j in range(n_drugs)]            # drug i targets gene i
    seed_amp = rng.uniform(1.0, 3.0, size=n_drugs)

    rows = []
    for di in range(n_drugs):
        s = np.zeros(n_genes)
        s[di] = seed_amp[di]
        lfc = s + 0.6 * (s @ P) + 0.25 * (s @ P @ P)       # 2-hop propagation, deterministic
        for ci in range(n_cells):
            ctrl = rng.normal(0, 1, n_genes).astype(np.float32)
            noisy = (lfc + rng.normal(0, 0.02, n_genes)).astype(np.float32)
            rows.append(dict(
                cell_line=f"c{ci}", treatment=f"d{di}", smiles=f"S{di}",
                target_gene=targets[di], dose=1.0, dose_log=0.0,
                ctrl_mean=ctrl.tolist(), lfc=noisy.tolist(),
                deg_mask=(np.abs(noisy) > 0.1).tolist(),
            ))
    df = pd.DataFrame(rows)
    df.attrs["genes"] = genes
    return df, grn, genes


def test_grn_propagation_beats_zero():
    df, grn, genes = _make_synthetic()
    n = len(df)
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    train_idx = perm[: int(0.7 * n)]
    test_idx = perm[int(0.7 * n):]

    model = GRNPropagationBaseline(n_hops=2).fit(df, train_idx, grn, genes=genes)
    pred = model.predict(df, test_idx)

    Y = np.stack([np.asarray(x, np.float32) for x in df["lfc"].to_numpy()])
    true = Y[test_idx]
    mse_model = float(np.mean((pred - true) ** 2))
    mse_zero = float(np.mean(true ** 2))

    assert pred.shape == (len(test_idx), len(genes))
    assert np.isfinite(pred).all()
    assert mse_model < mse_zero, f"GRN propagation ({mse_model:.4f}) should beat zero ({mse_zero:.4f})"


def test_b1_runs():
    df, grn, genes = _make_synthetic()
    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    test_idx = np.arange(int(0.7 * n), n)
    b1 = B1MeanEffect().fit(df, train_idx)
    pred = b1.predict(df, test_idx)
    assert pred.shape == (len(test_idx), len(genes))
    assert np.isfinite(pred).all()
