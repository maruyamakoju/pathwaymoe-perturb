"""Build a gene-regulatory-network mask aligned to a dataset's gene list.

Source: DoRothEA tier A+B via `decoupler` (TF -> target edges). If decoupler/network is
unavailable, falls back to a seeded random sparse mask of plausible density and records
source="fallback" so results are never silently based on fake structure.

Output: priors/<dataset>/grn_mask.npz (scipy sparse bool CSR, NxN; [i,j]=True => gene i regulates j).
"""
from __future__ import annotations
import argparse, json
import numpy as np
import scipy.sparse as sp

from config import DATA_ROOT
from loader import load_genes


def from_dorothea(genes):
    import decoupler as dc
    net = None
    for fn in ("get_dorothea", "op_dorothea"):
        if hasattr(dc, fn):
            try:
                net = getattr(dc, fn)(organism="human", levels=["A", "B"]); break
            except Exception:
                pass
    if net is None and hasattr(dc, "op"):           # decoupler>=2 API
        net = dc.op.dorothea(organism="human", levels=["A", "B"])
    gidx = {g: i for i, g in enumerate(genes)}
    rows, cols = [], []
    for s, t in zip(net["source"], net["target"]):
        if s in gidx and t in gidx:
            rows.append(gidx[s]); cols.append(gidx[t])
    n = len(genes)
    m = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    return m, f"dorothea_AB:{m.nnz}_edges"


def fallback(genes, density=0.01, seed=1337):
    n = len(genes); rng = np.random.default_rng(seed)
    n_tf = max(20, n // 10)
    tfs = rng.choice(n, n_tf, replace=False)
    rows, cols = [], []
    for tf in tfs:
        k = rng.integers(8, 30)
        tg = rng.choice(n, k, replace=False)
        rows += [tf] * k; cols += list(tg)
    m = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(n, n))
    return m, "fallback"


def build(dataset: str):
    genes = load_genes(dataset)
    try:
        m, source = from_dorothea(genes)
        if m.nnz == 0:
            raise RuntimeError("no overlapping edges")
    except Exception as e:
        m, source = fallback(genes)
        source = f"fallback({type(e).__name__})"
    pri = DATA_ROOT / "data" / "priors" / dataset; pri.mkdir(parents=True, exist_ok=True)
    sp.save_npz(pri / "grn_mask.npz", m.astype(bool))
    (pri / "grn_mask.meta.json").write_text(json.dumps(
        {"source": source, "n_edges": int(m.nnz), "n_genes": len(genes)}, indent=2))
    print(f"GRN[{dataset}] source={source} edges={m.nnz} density={m.nnz/len(genes)**2:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="synthetic")
    build(ap.parse_args().dataset)
