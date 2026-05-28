"""Build the GRN-quality spectrum (the controlled independent variable of the study).

All variants are aligned to the SAME gene space and matched to the same edge count (TRRUST's), so
the only thing that differs is *which* edges exist — isolating GRN quality, not density.

Variants (worst -> best expected):
  random      random edges (matched density)              -> "any sparse mask" control
  trrust      TRRUST v2 curated TF->target (generic)       -> generic prior
  coexpr      gene-gene correlation of control expression  -> data-derived, observational
  coexpr_lfc  gene-gene correlation of LFC across conds    -> data-derived, perturbation-response
  genie3      (optional) GENIE3/GRNBoost2 inferred TF->target (isolated venv) -> inferred regulatory

Each saved to priors/<dataset>/grn_<variant>.npz (bool CSR, NxN). train.py/eval pick via --grn-file.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import scipy.sparse as sp

from config import DATA_ROOT
from loader import load_conditions, load_genes, stack_arrays

PRIORS = DATA_ROOT / "data" / "priors"
VOCAB = DATA_ROOT / "data" / "tahoe100m" / "metadata" / "gene_vocabulary.jsonl"


def _sym2ens():
    import json as _j
    m = {}
    with open(VOCAB) as f:
        for line in f:
            r = _j.loads(line); s, e = r.get("gene_symbol"), r.get("ensembl_id")
            if s and e and s != e:
                m[s] = e
    return m


def trrust_edges(genes):
    import pandas as pd
    gidx = {g: i for i, g in enumerate(genes)}
    s2e = _sym2ens()
    net = pd.read_csv(PRIORS / "trrust_human.tsv", sep="\t", header=None,
                      names=["source", "target", "mode", "pmid"])
    e = set()
    for s, t in zip(net["source"].astype(str), net["target"].astype(str)):
        es, et = s2e.get(s, s), s2e.get(t, t)
        if es in gidx and et in gidx and es != et:
            e.add((gidx[es], gidx[et]))
    return e


def _save(genes, edges, dataset, variant, source):
    n = len(genes)
    if edges:
        r, c = zip(*edges)
        m = sp.csr_matrix((np.ones(len(edges), bool), (list(r), list(c))), shape=(n, n))
    else:
        m = sp.csr_matrix((n, n), dtype=bool)
    out = PRIORS / dataset; out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / f"grn_{variant}.npz", m)
    (out / f"grn_{variant}.meta.json").write_text(json.dumps(
        {"variant": variant, "source": source, "n_edges": int(m.nnz), "n_genes": n}, indent=2))
    print(f"  grn_{variant}: {m.nnz} edges  ({source})")
    return m


def _topk_corr_edges(M, k, exclude_self=True):
    """Top-k absolute-correlation gene pairs from matrix M (rows=samples, cols=genes)."""
    M = M.astype(np.float32)
    M = M - M.mean(0, keepdims=True)
    sd = M.std(0) + 1e-6
    C = (M.T @ M) / M.shape[0] / np.outer(sd, sd)        # gene x gene correlation
    np.fill_diagonal(C, 0.0)
    A = np.abs(C)
    # take top-k pairs overall (upper triangle to avoid double counting, then symmetrize)
    iu = np.triu_indices_from(A, k=1)
    vals = A[iu]
    if k >= len(vals):
        sel = np.arange(len(vals))
    else:
        sel = np.argpartition(-vals, k)[:k]
    edges = set()
    for s in sel:
        i, j = iu[0][s], iu[1][s]
        edges.add((i, j)); edges.add((j, i))            # symmetric
    return edges


def build(dataset, with_genie3=False, ground_truth=False):
    genes = load_genes(dataset)
    n = len(genes)
    df = load_conditions(dataset)
    print(f"[{dataset}] {n} genes, {len(df)} conditions")

    if ground_truth:
        # synthetic: the existing grn_mask.npz IS the true GRN; use its density as the budget.
        gt = sp.load_npz(PRIORS / dataset / "grn_mask.npz").tocoo()
        gt_edges = set(zip(gt.row.tolist(), gt.col.tolist()))
        _save(genes, gt_edges, dataset, "ground_truth", "synthetic_ground_truth")
        n_edges = max(len(gt_edges), 100)
        print(f"matched edge budget = {n_edges} (ground-truth GRN)")
    else:
        tr = trrust_edges(genes)
        n_edges = max(len(tr), 100)
        print(f"matched edge budget = {n_edges} (TRRUST)")
        _save(genes, tr, dataset, "trrust", "TRRUST_v2")
    rng = np.random.default_rng(1337)
    rand = set()
    while len(rand) < n_edges:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j:
            rand.add((i, j))
    _save(genes, rand, dataset, "random", "random_matched_density")

    pert = stack_arrays(df, "pert_mean")
    lfc = stack_arrays(df, "lfc")
    _save(genes, _topk_corr_edges(pert, n_edges // 2), dataset, "coexpr", "coexpr_expression")
    _save(genes, _topk_corr_edges(lfc, n_edges // 2), dataset, "coexpr_lfc", "coexpr_LFC_response")

    if with_genie3:
        try:
            build_genie3(dataset, genes, df, n_edges)
        except Exception as e:
            print(f"  genie3 skipped: {type(e).__name__}: {e}")


def build_genie3(dataset, genes, df, n_edges):
    """Inferred regulatory net via arboreto GRNBoost2 in the isolated decoupler venv (has deps)."""
    raise NotImplementedError("run separately via isolated venv if desired")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe_full")
    ap.add_argument("--genie3", action="store_true")
    ap.add_argument("--ground-truth", action="store_true", help="synthetic: include true GRN variant")
    a = ap.parse_args()
    build(a.dataset, a.genie3, a.ground_truth)
