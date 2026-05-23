"""Memory-bounded streaming preprocessor for Tahoe-100M at scale.

Processes ALL downloaded parquet shards in a single pass without holding cells in RAM. Memory is
bounded by (#cell_lines x #drugs) x #genes, NOT by #cells, so it scales to the full 100M-cell dataset.

Pipeline:
  for each shard: build sparse cells x genes, CPM-normalize + log1p per cell, group by
  (cell_line, drug) and accumulate pseudobulk SUM + cell COUNT on the FULL gene vocabulary; controls
  (drug==control_label) accumulated per cell_line.
  Then pseudobulk mean = sum/count; LFC = pert_mean - matched control_mean; select top-`n_hvg` genes
  by VARIANCE OF LFC ACROSS CONDITIONS (genes that respond to perturbations, the relevant signal);
  subset and write conditions.parquet for dataset `tahoe_full`.

DEG mask at scale = |LFC| > 0.5 (deep pseudobulk -> magnitude is a reliable significance proxy).
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pyarrow.parquet as pq

from config import DATA_ROOT, dataset_dir
from metrics import LFC_THRESH


def gene_symbols():
    import json as _j
    vocab = _j.loads((DATA_ROOT / "data" / "tahoe100m" / "metadata" / "gene_vocabulary.json").read_text())
    items = list(vocab.items())
    id2 = {v: k for k, v in vocab.items()} if isinstance(items[0][1], int) else {int(k): v for k, v in vocab.items()}
    n = max(id2) + 1
    return [id2.get(i, f"GENE{i}") for i in range(n)], n


def _norm_log(X: sp.csr_matrix) -> sp.csr_matrix:
    """CPM->1e4 + log1p per cell, on sparse (log1p(0)=0 keeps sparsity)."""
    X = X.tocsr().astype(np.float32)
    rs = np.asarray(X.sum(1)).ravel(); rs[rs == 0] = 1.0
    inv = (1e4 / rs).astype(np.float32)
    X = sp.diags(inv) @ X
    np.clip(X.data, 0, None, out=X.data)        # guard against any negative/garbage values
    X.data = np.log1p(X.data)
    return X


def run(dataset="tahoe_full", n_hvg=4000, control_label="DMSO_TF", min_cells=50,
        min_ctrl=50, shard_limit=None, log_every=25):
    syms, n_vocab = gene_symbols()
    files = sorted(glob.glob(str(DATA_ROOT / "data" / "tahoe100m" / "data" / "*.parquet")))
    if shard_limit:
        files = files[:shard_limit]
    print(f"streaming {len(files)} shards, vocab={n_vocab}", flush=True)

    cond_sum: dict[tuple, np.ndarray] = {}
    cond_cnt: dict[tuple, int] = {}
    cond_smiles: dict[tuple, str] = {}
    ctrl_sum: dict[str, np.ndarray] = {}
    ctrl_cnt: dict[str, int] = {}
    t0 = time.time(); n_cells = 0
    for si, f in enumerate(files):
        t = pq.read_table(f, columns=["genes", "expressions", "drug", "cell_line_id",
                                      "canonical_smiles"]).to_pandas()
        # build sparse matrix for this shard
        lens = t["genes"].map(len).to_numpy()
        indptr = np.concatenate([[0], np.cumsum(lens)])
        indices = np.concatenate(t["genes"].to_numpy()).astype(np.int32)
        data = np.concatenate(t["expressions"].to_numpy()).astype(np.float32)
        X = sp.csr_matrix((data, indices, indptr), shape=(len(t), n_vocab))
        X = _norm_log(X)
        n_cells += len(t)
        smi_arr = t["canonical_smiles"].astype(str).to_numpy()
        # group rows by (cell_line, drug) via pandas groupby on row indices
        t = t.assign(_row=np.arange(len(t)))
        for (c, d), g in t.groupby(["cell_line_id", "drug"], sort=False):
            rows = g["_row"].to_numpy()
            c = str(c); d = str(d)
            gsum = np.asarray(X[rows].sum(0)).ravel().astype(np.float32)
            if d == control_label:
                if c not in ctrl_sum:
                    ctrl_sum[c] = np.zeros(n_vocab, np.float32); ctrl_cnt[c] = 0
                ctrl_sum[c] += gsum; ctrl_cnt[c] += len(rows)
            else:
                key = (c, d)
                if key not in cond_sum:
                    cond_sum[key] = np.zeros(n_vocab, np.float32); cond_cnt[key] = 0
                    cond_smiles[key] = smi_arr[rows[0]]
                cond_sum[key] += gsum; cond_cnt[key] += len(rows)
        if si % log_every == 0:
            el = time.time() - t0
            print(f"shard {si}/{len(files)} cells={n_cells:,} conds={len(cond_sum)} "
                  f"ctrls={len(ctrl_sum)} {el:.0f}s", flush=True)

    # build conditions with matched controls + cell thresholds
    print(f"aggregation done: {n_cells:,} cells, {len(cond_sum)} raw conditions", flush=True)
    keys = [k for k in cond_sum if cond_cnt[k] >= min_cells and k[0] in ctrl_cnt
            and ctrl_cnt[k[0]] >= min_ctrl]
    print(f"{len(keys)} conditions pass thresholds (min_cells={min_cells})", flush=True)
    ctrl_mean = {c: ctrl_sum[c] / ctrl_cnt[c] for c in ctrl_sum}
    # LFC matrix on full genes for HVG-by-variance
    lfc_full = np.empty((len(keys), n_vocab), np.float32)
    pert_means = np.empty((len(keys), n_vocab), np.float32)
    for i, k in enumerate(keys):
        pm = cond_sum[k] / cond_cnt[k]
        pert_means[i] = pm
        lfc_full[i] = pm - ctrl_mean[k[0]]
    hv = np.argsort(-lfc_full.var(0))[:n_hvg]
    hv.sort()
    genes = [syms[g] for g in hv]
    print(f"selected {n_hvg} HVGs by cross-condition LFC variance", flush=True)

    import pandas as pd
    rows = []
    for i, k in enumerate(keys):
        c, d = k
        lfc = lfc_full[i, hv]
        rows.append(dict(condition_id=f"{c}|{d}|1.0", cell_line=c, treatment=d,
                         smiles=cond_smiles[k], target_gene="", dose=1.0,
                         dose_log=float(np.log1p(1.0)), n_cells=int(cond_cnt[k]),
                         n_ctrl_cells=int(ctrl_cnt[c]),
                         ctrl_mean=ctrl_mean[c][hv].astype(np.float32),
                         pert_mean=pert_means[i, hv].astype(np.float32),
                         lfc=lfc.astype(np.float32),
                         deg_mask=(np.abs(lfc) > LFC_THRESH)))
    df = pd.DataFrame(rows)
    out = dataset_dir(dataset); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "conditions.parquet")
    (out / "genes.txt").write_text("\n".join(genes))
    (out / "meta.json").write_text(json.dumps(dict(
        dataset=dataset, N_GENES=len(genes), cell_lines=sorted({k[0] for k in keys}),
        drugs=[], dose_unit="uM", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_conditions=len(df), n_cells_total=int(n_cells), n_shards=len(files),
        notes=f"streaming Tahoe, HVG by cross-condition LFC variance, deg=|lfc|>0.5"), indent=2))
    print(f"wrote {len(df)} conditions x {len(genes)} genes -> {out}", flush=True)
    print(f"mean cells/cond={np.mean([cond_cnt[k] for k in keys]):.0f}, "
          f"mean DEGs/cond={np.mean([r['deg_mask'].sum() for r in rows]):.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe_full")
    ap.add_argument("--n-hvg", type=int, default=4000)
    ap.add_argument("--control-label", default="DMSO_TF")
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--shard-limit", type=int, default=None)
    a = ap.parse_args()
    run(a.dataset, a.n_hvg, a.control_label, a.min_cells, shard_limit=a.shard_limit)
