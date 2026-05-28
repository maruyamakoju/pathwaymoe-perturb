"""Assemble the comparison table: baselines B1/B2/B3 vs trained PathwayMoE across the three splits.

Reuses the SAME split seed as training so test sets match. For each split it loads the matching
checkpoint E:/checkpoints/<dataset>_<split>_<size>.pt if present. Writes
results/comparison_<dataset>.json and a markdown table to results/comparison_<dataset>.md.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from config import CKPT_DIR, RESULTS_DIR, SEED, DATA_ROOT
from loader import load_conditions, load_genes, stack_arrays
from splits import make_splits
from metrics import evaluate
from data import build_shared, make_loader
from model import ModelConfig, PathwayMoEPerturb, count_params
import baselines as B
import scipy.sparse as sp

SPLITS = ["unseen_drug", "unseen_cell_line", "unseen_both"]
KEY = "pearson_deg50"


def model_predict(dataset, df, shared, test_idx, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    pri = DATA_ROOT / "data" / "priors" / dataset
    grn = sp.load_npz(pri / "grn_mask.npz").toarray().astype(bool) if (pri / "grn_mask.npz").exists() else None
    gp = sp.load_npz(pri / "pathways.npz").toarray() if (pri / "pathways.npz").exists() else None
    m = PathwayMoEPerturb(cfg, grn, gp).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    loader = make_loader(dataset, test_idx, df, shared, batch_size=16, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                preds.append(m(b).float().cpu().numpy())
    return np.concatenate(preds), count_params(m)


def run(dataset, size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_conditions(dataset)
    Y = stack_arrays(df, "lfc"); deg = stack_arrays(df, "deg_mask")
    shared = build_shared(dataset, df)
    fz = B.Featurizer(dataset, df)

    table = {}
    for split in SPLITS:
        sp_ = make_splits(df, split, seed=SEED)
        tr, va, te = sp_["train"], sp_["val"], sp_["test"]
        groups = df.iloc[te]["cell_line"].to_numpy()
        row = {}
        for Bl in (B.B1MeanEffect(), B.B2Ridge(), B.B3RidgeBio()):
            Bl.fit(df, tr, val_idx=va, fz=fz, deg=deg)
            pred = Bl.predict(df, te)
            m = evaluate(pred, Y[te], deg[te], groups)["overall"]
            row[Bl.name] = {"params": fz_params(Bl), **{k: round(float(v), 4) for k, v in m.items()}}
        ckpt = CKPT_DIR / f"{dataset}_{split}_{size}.pt"
        if ckpt.exists():
            pred, n = model_predict(dataset, df, shared, te, ckpt, device)
            m = evaluate(pred, Y[te], deg[te], groups)["overall"]
            row["PathwayMoE"] = {"params": int(n), **{k: round(float(v), 4) for k, v in m.items()}}
        table[split] = row
        print(f"\n[{split}]")
        for k, v in row.items():
            print(f"  {k:16s} DEG50={v.get('pearson_deg50'):.3f}  all={v.get('pearson_all'):.3f}  "
                  f"dir50={v.get('direction_acc50'):.3f}")

    (RESULTS_DIR / f"comparison_{dataset}.json").write_text(json.dumps(table, indent=2))
    _write_md(dataset, table)
    return table


def fz_params(bl):
    return {"B1_mean_effect": 0, "B2_ridge": 1_000_000, "B3_ridge_bio": 1_000_000}.get(bl.name, 0)


def _write_md(dataset, table):
    models = ["B1_mean_effect", "B2_ridge", "B3_ridge_bio", "PathwayMoE"]
    lines = [f"# Comparison — {dataset} (metric: DEG-Pearson @top-50)\n",
             "| Model | Params | unseen_drug | unseen_cell_line | unseen_both |",
             "|---|---|---|---|---|"]
    for mdl in models:
        cells = []
        params = None
        for split in SPLITS:
            v = table[split].get(mdl)
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v['pearson_deg50']:.3f}")
                params = v.get("params")
        if any(c != "—" for c in cells):
            pstr = "0" if params == 0 else (f"{params/1e6:.0f}M" if params and params >= 1e6 else str(params))
            lines.append(f"| {mdl} | {pstr} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"comparison_{dataset}.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--size", default="base")
    run(ap.parse_args().dataset, ap.parse_args().size)
