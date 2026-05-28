"""Preprocess REAL Tahoe-100M (HuggingFace parquet shards) into the condition-level schema.

Tahoe-100M reality (differs from the spec's h5ad assumption):
  - 3388 parquet shards, ~71 MB / 28k cells each (~240 GB total).
  - Each row is ONE cell with SPARSE expression: `genes` (list<int64> vocab IDs) +
    `expressions` (list<float> raw counts), plus `drug`, `cell_line_id` (Cellosaurus CVCL),
    `canonical_smiles`, `moa-fine`, `plate`, `sample`. No explicit dose column.
  - Controls are DMSO cells (drug label configurable, default "DMSO_TF"); they live on specific
    plates, so you must download shards that actually contain controls for your target cell lines.
    LFC = pseudobulk(treated) - pseudobulk(matched DMSO, same cell line).

Strategy: stream a chosen set of shards, accumulate a sparse cells×genes CSR for a target set of
cell lines, pick HVGs on the pooled counts, then pseudobulk per (cell_line, drug) with the
cell-line-matched DMSO control. Writes <dataset>/processed/{conditions.parquet, genes.txt, meta.json}.

This is the real-data path; the synthetic dataset is the fully-validated one. Run `--list-controls`
first to discover which shards contain DMSO and how many cells per cell line.
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import DATA_ROOT, dataset_dir
from metrics import LFC_THRESH

TAHOE_REPO = "tahoebio/Tahoe-100M"


def load_gene_vocab():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(TAHOE_REPO, "metadata/gene_vocabulary.json", repo_type="dataset",
                        local_dir=str(DATA_ROOT / "data" / "tahoe100m"))
    vocab = json.loads(Path(p).read_text())
    # vocab maps symbol/ensembl -> id (or id->symbol); normalize to id->symbol
    items = list(vocab.items())
    if isinstance(items[0][1], int):
        id2sym = {v: k for k, v in vocab.items()}
    else:
        id2sym = {int(k): v for k, v in vocab.items()}
    n = max(id2sym) + 1
    syms = [id2sym.get(i, f"GENE{i}") for i in range(n)]
    return syms


def download_shards(n_shards, start=0):
    from huggingface_hub import hf_hub_download
    paths = []
    for i in range(start, start + n_shards):
        fn = f"data/train-{i:05d}-of-03388.parquet"
        p = hf_hub_download(TAHOE_REPO, fn, repo_type="dataset",
                            local_dir=str(DATA_ROOT / "data" / "tahoe100m"))
        paths.append(p)
    return paths


def _shard_files(limit=None, start=0):
    files = sorted(glob.glob(str(DATA_ROOT / "data" / "tahoe100m" / "data" / "*.parquet")))
    if limit:
        files = files[start:start + limit]
    return files


def list_controls(control_label="DMSO_TF", limit=None):
    import pyarrow.parquet as pq
    files = _shard_files(limit)
    print(f"scanning {len(files)} shards for control='{control_label}'...")
    tot = {}
    for f in files:
        t = pq.read_table(f, columns=["drug", "cell_line_id"]).to_pandas()
        c = t[t["drug"].astype(str) == control_label]
        if len(c):
            for cl, n in c["cell_line_id"].value_counts().items():
                tot[cl] = tot.get(cl, 0) + int(n)
            print(f"  {Path(f).name}: {len(c)} control cells")
    print(f"control cells per cell line (top): "
          f"{dict(sorted(tot.items(), key=lambda kv:-kv[1])[:10])}")
    return tot


def run(dataset="tahoe", n_hvg=2000, control_label="DMSO_TF", min_cells=20,
        target_cell_lines=None, shard_limit=None, shard_start=0, min_genes=500):
    import pyarrow.parquet as pq
    syms = load_gene_vocab()
    n_vocab = len(syms)
    files = _shard_files(shard_limit, shard_start)
    if not files:
        print("no shards downloaded; use download_shards or --download N"); return
    print(f"{len(files)} shards, vocab={n_vocab} genes")

    # accumulate sparse cells x genes (CSR) + per-cell meta for target cell lines
    rows_g, rows_e, rows_ptr = [], [], [0]
    meta = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        if target_cell_lines:
            t = t[t["cell_line_id"].isin(target_cell_lines)]
        for genes, expr, drug, cl, smi, moa, plate in zip(
                t["genes"], t["expressions"], t["drug"], t["cell_line_id"],
                t["canonical_smiles"], t["moa-fine"], t["plate"]):
            g = np.asarray(genes); e = np.asarray(expr, np.float32)
            if g.size < min_genes:
                continue
            rows_g.append(g); rows_e.append(e); rows_ptr.append(rows_ptr[-1] + g.size)
            meta.append((str(drug), str(cl), str(smi), str(moa), str(plate)))
    if not meta:
        print("no cells matched filters"); return
    data = np.concatenate(rows_e); indices = np.concatenate(rows_g)
    X = sp.csr_matrix((data, indices, np.array(rows_ptr)), shape=(len(meta), n_vocab))
    meta = pd.DataFrame(meta, columns=["drug", "cell_line", "smiles", "moa", "plate"])
    print(f"accumulated {X.shape[0]} cells x {X.shape[1]} genes; "
          f"cell_lines={meta['cell_line'].nunique()} drugs={meta['drug'].nunique()} "
          f"controls={(meta['drug']==control_label).sum()}")

    # normalize -> 1e4, log1p ; HVG via variance of log-normalized (seurat_v3 needs counts layer)
    import scanpy as sc, anndata as ad
    A = ad.AnnData(X=X.tocsr(), obs=meta.reset_index(drop=True))
    A.var_names = [str(s) for s in syms]
    A.layers["counts"] = A.X.copy()
    try:
        sc.pp.highly_variable_genes(A, flavor="seurat_v3", n_top_genes=n_hvg, layer="counts")
        hv = A.var["highly_variable"].to_numpy()
    except Exception as e:
        print(f"seurat_v3 unavailable ({type(e).__name__}); sparse top-variance HVG fallback")
        Xc = A.layers["counts"].tocsc()           # raw counts, sparse (no densify)
        n = Xc.shape[0]
        mean = np.asarray(Xc.mean(0)).ravel()
        sq = np.asarray(Xc.multiply(Xc).mean(0)).ravel()
        var = sq - mean ** 2                       # per-gene variance on counts
        hv = np.zeros(A.shape[1], bool); hv[np.argsort(-var)[:n_hvg]] = True
    A = A[:, hv].copy()
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    genes = A.var_names.tolist()
    Xn = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    obs = A.obs.reset_index(drop=True)

    out_rows = []
    for cl, cdf in obs.groupby("cell_line"):
        cmask = (cdf["drug"] == control_label).to_numpy()
        cidx = cdf.index[cmask].to_numpy()
        if len(cidx) < min_cells:
            print(f"  {cl}: only {len(cidx)} controls, skipping"); continue
        ctrl_mean = Xn[cidx].mean(0)
        for drug, gdf in cdf[~cmask].groupby("drug"):
            gi = gdf.index.to_numpy()
            if len(gi) < min_cells:
                continue
            pert_mean = Xn[gi].mean(0)
            lfc = (pert_mean - ctrl_mean).astype(np.float32)
            pooled_sd = np.sqrt((Xn[gi].var(0) + Xn[cidx].var(0)) / 2 + 1e-6)
            n = min(len(gi), len(cidx)); se = pooled_sd / max(1.0, np.sqrt(n))
            deg = (np.abs(lfc) > LFC_THRESH) & (np.abs(lfc) > 3.0 * se)
            smi = gdf["smiles"].iloc[0]
            out_rows.append(dict(condition_id=f"{cl}|{drug}|1.0", cell_line=str(cl),
                                 treatment=str(drug), smiles=str(smi), target_gene="",
                                 dose=1.0, dose_log=float(np.log1p(1.0)),
                                 n_cells=len(gi), n_ctrl_cells=len(cidx),
                                 ctrl_mean=ctrl_mean.astype(np.float32),
                                 pert_mean=pert_mean.astype(np.float32), lfc=lfc, deg_mask=deg))
    df = pd.DataFrame(out_rows)
    if df.empty:
        print("no conditions met thresholds (need matched controls + >= min_cells)"); return
    out = dataset_dir(dataset); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "conditions.parquet")
    (out / "genes.txt").write_text("\n".join(genes))
    (out / "meta.json").write_text(json.dumps(dict(
        dataset=dataset, N_GENES=len(genes), cell_lines=sorted(df["cell_line"].unique().tolist()),
        drugs=[], dose_unit="uM", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_conditions=len(df), notes=f"real Tahoe-100M, {len(files)} shards"), indent=2))
    print(f"wrote {len(df)} conditions, {len(genes)} genes -> {out}")
    print("NOTE: target_gene blank; map drug->target separately for B3/model target features.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=int, default=0, help="download N shards first")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-limit", type=int, default=None)
    ap.add_argument("--list-controls", action="store_true")
    ap.add_argument("--control-label", default="DMSO_TF")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--cell-lines", nargs="*", default=None)
    ap.add_argument("--dataset", default="tahoe")
    a = ap.parse_args()
    if a.download:
        print("downloading", a.download, "shards..."); download_shards(a.download, a.shard_start)
    if a.list_controls:
        list_controls(a.control_label, a.shard_limit)
    else:
        run(a.dataset, a.n_hvg, a.control_label, target_cell_lines=a.cell_lines,
            shard_limit=a.shard_limit, shard_start=a.shard_start)
