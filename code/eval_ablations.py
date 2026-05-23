"""Evaluate ablation checkpoints on the unseen_drug TEST set (not val), for a rigorous ablation
table. Each ablation checkpoint stores its own cfg (with the disabled component), so we rebuild
the matching model. Writes results/ablations_<dataset>.json + markdown."""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from config import CKPT_DIR, RESULTS_DIR, SEED, DATA_ROOT
from loader import load_conditions, stack_arrays
from splits import make_splits
from data import build_shared, make_loader
from model import ModelConfig, PathwayMoEPerturb
from metrics import evaluate
import scipy.sparse as sp

VARIANTS = {
    "full": "{ds}_unseen_drug_base.pt",
    "no_GRN_mask": "{ds}_unseen_drug_base_nogrn.pt",
    "no_MoE": "{ds}_unseen_drug_base_nomoe.pt",
    "no_pathway_prior": "{ds}_unseen_drug_base_noprior.pt",
    "pert_as_token": "{ds}_unseen_drug_base_perttoken.pt",
}


def predict(ckpt, dataset, df, shared, idx, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    pri = DATA_ROOT / "data" / "priors" / dataset
    grn = sp.load_npz(pri / "grn_mask.npz").toarray().astype(bool)
    gp = sp.load_npz(pri / "pathways.npz").toarray()
    m = PathwayMoEPerturb(cfg, grn, gp).to(device); m.load_state_dict(ck["model"]); m.eval()
    loader = make_loader(dataset, idx, df, shared, batch_size=32, shuffle=False)
    out = []
    with torch.no_grad():
        for b in loader:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(m(bb).float().cpu().numpy())
    return np.concatenate(out)


def run(dataset="synthetic"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_conditions(dataset); Y = stack_arrays(df, "lfc"); deg = stack_arrays(df, "deg_mask")
    shared = build_shared(dataset, df)
    sp_ = make_splits(df, "unseen_drug", seed=SEED); te = sp_["test"]
    rows = {}
    full = None
    for name, pat in VARIANTS.items():
        ckpt = CKPT_DIR / pat.format(ds=dataset)
        if not ckpt.exists():
            print(f"missing {ckpt}"); continue
        pred = predict(ckpt, dataset, df, shared, te, device)
        m = evaluate(pred, Y[te], deg[te])["overall"]
        rows[name] = {k: round(float(v), 4) for k, v in m.items()}
        if name == "full":
            full = m["pearson_deg50"]
    for name in rows:
        rows[name]["delta_vs_full"] = round(rows[name]["pearson_deg50"] - full, 4) if full else None
    (RESULTS_DIR / f"ablations_{dataset}.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Ablations — unseen_drug TEST set\n",
             "| Variant | DEG-Pearson@50 | Δ vs full | all-Pearson | dir@50 |",
             "|---|---|---|---|---|"]
    for name, v in rows.items():
        lines.append(f"| {name} | {v['pearson_deg50']:.3f} | {v.get('delta_vs_full',0):+.3f} | "
                     f"{v['pearson_all']:.3f} | {v['direction_acc50']:.3f} |")
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"ablations_{dataset}.md").write_text(md)
    print(md)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="synthetic")
    run(ap.parse_args().dataset)
