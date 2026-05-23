"""Preprocess real Tahoe-100M (or any perturbation h5ad collection) into the condition-level
schema (see data/schema.md).

Pipeline (section 3 of the spec):
  1. load .h5ad shards in backed mode
  2. QC filter: n_genes_by_counts >= min_genes, pct_counts_mt <= max_mt
  3. normalize total-count -> 1e4, log1p; select top HVG (seurat_v3 on raw counts)
  4. for each (cell_line, treatment, dose): pseudobulk mean of perturbed cells AND of the matched
     DMSO control cells from the same cell line -> ctrl_mean, pert_mean, lfc, deg_mask
  5. write conditions.parquet + genes.txt + meta.json under <dataset>/processed/

Obs column names are configurable because dataset releases vary; defaults match Tahoe-100M's
metadata fields. Use --dry-run to print the plan without writing. This is best-effort on real data;
the synthetic dataset is the fully-validated path.
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA_ROOT, dataset_dir
from metrics import LFC_THRESH


def _deg_mask(lfc, pert_cells, ctrl_cells, pooled_sd, z=3.0):
    n = min(pert_cells, ctrl_cells)
    se = pooled_sd / max(1.0, np.sqrt(n))
    return (np.abs(lfc) > LFC_THRESH) & (np.abs(lfc) > z * se)


def run(input_dir, dataset="tahoe", n_hvg=2000, min_genes=500, max_mt=20.0,
        cell_line_col="cell_line", drug_col="drug", dose_col="dose",
        control_value="DMSO_TF", max_cells_per_cond=10000, dry_run=False, limit_shards=None):
    import scanpy as sc, anndata as ad
    files = sorted(glob.glob(str(Path(input_dir) / "**" / "*.h5ad"), recursive=True))
    if limit_shards:
        files = files[:limit_shards]
    print(f"found {len(files)} h5ad shards under {input_dir}")
    if not files:
        print("no shards; nothing to do"); return
    if dry_run:
        a = sc.read_h5ad(files[0], backed="r")
        print(f"[dry-run] first shard: {a.shape}; obs cols: {list(a.obs.columns)[:20]}")
        print(f"[dry-run] would HVG->{n_hvg}, group by ({cell_line_col},{drug_col},{dose_col}), "
              f"control='{control_value}'"); return

    # --- pass 1: concatenate (subset) and QC, pick HVGs on raw counts ---
    adatas = []
    for f in files:
        a = sc.read_h5ad(f)
        sc.pp.calculate_qc_metrics(a, percent_top=None, inplace=True,
                                   qc_vars=["mt"] if "mt" in a.var.columns else [])
        if "n_genes_by_counts" in a.obs:
            a = a[a.obs["n_genes_by_counts"] >= min_genes]
        adatas.append(a)
    A = ad.concat(adatas, join="inner")
    print(f"concatenated: {A.shape}")
    A.layers["counts"] = A.X.copy()
    sc.pp.highly_variable_genes(A, flavor="seurat_v3", n_top_genes=n_hvg, layer="counts")
    A = A[:, A.var["highly_variable"]].copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    genes = A.var_names.tolist()

    obs = A.obs
    X = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    rows = []
    for cl, cdf in obs.groupby(cell_line_col):
        ctrl_mask = (cdf[drug_col].astype(str) == control_value).to_numpy()
        ctrl_idx = cdf.index[ctrl_mask]
        if len(ctrl_idx) == 0:
            print(f"  {cl}: no controls, skipping"); continue
        ci = obs.index.get_indexer(ctrl_idx)
        ctrl_mean = X[ci].mean(0)
        for (drug, dose), gdf in cdf[~ctrl_mask].groupby([drug_col, dose_col]):
            gi = obs.index.get_indexer(gdf.index[:max_cells_per_cond])
            if len(gi) < 5:
                continue
            pert_mean = X[gi].mean(0)
            lfc = (pert_mean - ctrl_mean).astype(np.float32)
            pooled_sd = np.sqrt((X[gi].var(0) + X[ci].var(0)) / 2 + 1e-6)
            deg = _deg_mask(lfc, len(gi), len(ci), pooled_sd)
            rows.append(dict(condition_id=f"{cl}|{drug}|{dose}", cell_line=str(cl),
                             treatment=str(drug), smiles="", target_gene="",
                             dose=float(dose) if np.isreal(dose) else 0.0,
                             dose_log=float(np.log1p(float(dose) if np.isreal(dose) else 0.0)),
                             n_cells=len(gi), n_ctrl_cells=len(ci),
                             ctrl_mean=ctrl_mean.astype(np.float32),
                             pert_mean=pert_mean.astype(np.float32), lfc=lfc, deg_mask=deg))
    df = pd.DataFrame(rows)
    out = dataset_dir(dataset); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "conditions.parquet")
    (out / "genes.txt").write_text("\n".join(genes))
    (out / "meta.json").write_text(json.dumps(dict(
        dataset=dataset, N_GENES=len(genes), cell_lines=sorted(df["cell_line"].unique().tolist()),
        drugs=[], dose_unit="uM", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_conditions=len(df), notes=f"real preprocess from {input_dir}"), indent=2))
    print(f"wrote {len(df)} conditions, {len(genes)} genes -> {out}")
    print("NOTE: SMILES/target_gene are blank; map drug names -> SMILES/targets before B2/B3.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(DATA_ROOT / "data" / "tahoe100m"))
    ap.add_argument("--dataset", default="tahoe")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--cell-line-col", default="cell_line")
    ap.add_argument("--drug-col", default="drug")
    ap.add_argument("--dose-col", default="dose")
    ap.add_argument("--control-value", default="DMSO_TF")
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run(a.input_dir, a.dataset, a.n_hvg, cell_line_col=a.cell_line_col, drug_col=a.drug_col,
        dose_col=a.dose_col, control_value=a.control_value, dry_run=a.dry_run,
        limit_shards=a.limit_shards)
