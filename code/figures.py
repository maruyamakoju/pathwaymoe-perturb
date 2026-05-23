"""Figures for the report: (1) comparison bar chart, (2) training curves, (3) expert specialization
heatmap (interpretability). All read artifacts from results/ and runs/; no training here."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR, RUNS_DIR

SPLITS = ["unseen_drug", "unseen_cell_line", "unseen_both"]


def comparison_bar(dataset):
    p = RESULTS_DIR / f"comparison_{dataset}.json"
    if not p.exists():
        print(f"no {p}"); return
    table = json.loads(p.read_text())
    models = ["B1_mean_effect", "B2_ridge", "B3_ridge_bio", "PathwayMoE"]
    present = [m for m in models if any(m in table[s] for s in SPLITS)]
    x = np.arange(len(SPLITS)); w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(present):
        vals = [table[s].get(m, {}).get("pearson_deg50", np.nan) for s in SPLITS]
        ax.bar(x + i * w, vals, w, label=m)
    ax.set_xticks(x + w * (len(present) - 1) / 2); ax.set_xticklabels(SPLITS)
    ax.set_ylabel("DEG-Pearson @top-50"); ax.set_title(f"Perturbation prediction — {dataset}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    f = RESULTS_DIR / f"fig_comparison_{dataset}.png"; fig.tight_layout(); fig.savefig(f, dpi=150)
    print(f"wrote {f}")


def training_curves(pattern="*"):
    files = sorted(RUNS_DIR.glob(f"{pattern}.jsonl"))
    if not files:
        print("no run logs"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in files:
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        ep = [r["epoch"] for r in recs]; v = [r["val_deg50"] for r in recs]
        ax.plot(ep, v, marker=".", label=f.stem)
    ax.set_xlabel("epoch"); ax.set_ylabel("val DEG-Pearson@50"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3); ax.set_title("Training curves")
    f = RESULTS_DIR / "fig_training_curves.png"; fig.tight_layout(); fig.savefig(f, dpi=150)
    print(f"wrote {f}")


def pathway_response(dataset, split="unseen_drug", size="base", max_drugs=40):
    """Interpretability: per-drug PREDICTED-LFC magnitude aggregated by pathway -> heatmap
    (drug x pathway). A correct model should concentrate each drug's predicted response in its
    own pathway (e.g., MEK/MAPK inhibitors -> MAPK genes). Tied directly to predictions, not to
    internal routing mechanics."""
    import torch
    from config import CKPT_DIR, DATA_ROOT
    from loader import load_conditions
    from data import build_shared, make_loader
    from model import ModelConfig, PathwayMoEPerturb
    import scipy.sparse as sp
    ckpt = CKPT_DIR / f"{dataset}_{split}_{size}.pt"
    if not ckpt.exists():
        print(f"no checkpoint {ckpt}"); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    pri = DATA_ROOT / "data" / "priors" / dataset
    grn = sp.load_npz(pri / "grn_mask.npz").toarray().astype(bool)
    gp = sp.load_npz(pri / "pathways.npz").toarray().astype(bool)   # (N, P)
    pw_names = (pri / "pathway_names.txt").read_text().splitlines()
    m = PathwayMoEPerturb(cfg, grn, gp).to(device); m.load_state_dict(ck["model"]); m.eval()

    df = load_conditions(dataset); shared = build_shared(dataset, df)
    loader = make_loader(dataset, np.arange(len(df)), df, shared, batch_size=32, shuffle=False)
    pw_size = gp.sum(0) + 1e-6
    resp = {}
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = m(b).float().cpu().numpy()
            mag = np.abs(pred)                                  # (B, N)
            pw_mag = mag @ gp / pw_size                         # (B, P) mean |pred| per pathway
            for r, drug in enumerate(batch["treatment"]):
                resp.setdefault(drug, []).append(pw_mag[r])
    drugs = list(resp.keys())[:max_drugs]
    M = np.stack([np.mean(resp[d], 0) for d in drugs])
    M = M / (M.max(1, keepdims=True) + 1e-9)                    # row-normalize for display
    fig, ax = plt.subplots(figsize=(10, max(3, len(drugs) * 0.28)))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(drugs))); ax.set_yticklabels(drugs, fontsize=7)
    ax.set_xticks(range(len(pw_names))); ax.set_xticklabels(pw_names, rotation=90, fontsize=6)
    ax.set_title("Predicted perturbation response by pathway (row-normalized)"); fig.colorbar(im)
    f = RESULTS_DIR / f"fig_pathway_response_{dataset}.png"; fig.tight_layout(); fig.savefig(f, dpi=150)
    print(f"wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--what", default="all", choices=["all", "comparison", "curves", "experts"])
    a = ap.parse_args()
    if a.what in ("all", "comparison"): comparison_bar(a.dataset)
    if a.what in ("all", "curves"): training_curves()
    if a.what in ("all", "experts"):
        try: pathway_response(a.dataset)
        except Exception as e: print(f"pathway-response fig skipped: {e}")
