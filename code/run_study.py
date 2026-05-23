"""Orchestrate the GRN-quality controlled study: train PathwayMoE across GRN variants x splits x
seeds, then analyze with bootstrap CIs and paired significance tests.

Variants map to train.py flags:
  none        --no-grn
  random      --grn-file grn_random.npz
  trrust      --grn-file grn_trrust.npz
  coexpr      --grn-file grn_coexpr.npz
  coexpr_lfc  --grn-file grn_coexpr_lfc.npz

Resumable: skips a run if its checkpoint already exists. Use --train-only / --analyze-only.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

from config import CKPT_DIR, RESULTS_DIR, SEED, DATA_ROOT
from loader import load_conditions, stack_arrays
from splits import make_splits
from data import build_shared, make_loader
from model import ModelConfig, PathwayMoEPerturb
import scipy.sparse as sp
from stats import per_condition_deg_pearson, bootstrap_ci, paired_bootstrap, aggregate_seeds
import baselines as B
from metrics import evaluate

PY = str(Path(".venv/Scripts/python.exe").resolve())
VARIANT_FLAGS = {
    "none": ["--no-grn"],
    "random": ["--grn-file", "grn_random.npz"],
    "trrust": ["--grn-file", "grn_trrust.npz"],
    "coexpr": ["--grn-file", "grn_coexpr.npz"],
    "coexpr_lfc": ["--grn-file", "grn_coexpr_lfc.npz"],
}


def ckpt_name(dataset, split, variant, seed, size="base"):
    name = f"{dataset}_{split}_{size}"
    if variant not in ("none",):
        name += f"_{variant.replace('grn_','')}"          # matches train.py grn_tag (file stem minus 'grn_')
    if seed != SEED:
        name += f"_s{seed}"
    if variant == "none":
        name += "_nogrn"
    return name


def train_one(dataset, split, variant, seed, epochs, size="base", batch=64):
    name = ckpt_name(dataset, split, variant, seed, size)
    ckpt = CKPT_DIR / f"{name}.pt"
    if ckpt.exists():
        print(f"[skip] {name} exists"); return name
    cmd = [PY, "-u", "code/train.py", "--dataset", dataset, "--split", split, "--size", size,
           "--batch-size", str(batch), "--epochs", str(epochs), "--patience", "10",
           "--eval-every", "2", "--seed", str(seed)] + VARIANT_FLAGS[variant]
    print(f"[train] {name}  ::  {' '.join(VARIANT_FLAGS[variant])} seed={seed}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join(r.stdout.splitlines()[-3:])
    print(tail, flush=True)
    if not ckpt.exists():
        print(f"[WARN] {name} produced no checkpoint:\n{r.stdout[-800:]}\n{r.stderr[-400:]}", flush=True)
    return name


VARIANT_GRNFILE = {"none": None, "random": "grn_random.npz", "trrust": "grn_trrust.npz",
                   "coexpr": "grn_coexpr.npz", "coexpr_lfc": "grn_coexpr_lfc.npz"}


def predict_test(dataset, name, variant, df, shared, test_idx, device):
    ck = torch.load(CKPT_DIR / f"{name}.pt", map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    pri = DATA_ROOT / "data" / "priors" / dataset
    # CRITICAL: reload the SAME GRN variant used in training (grn_bias is a non-persistent buffer).
    gf = VARIANT_GRNFILE.get(variant)
    grn = sp.load_npz(pri / gf).toarray().astype(bool) if gf and (pri / gf).exists() else None
    gp = sp.load_npz(pri / "pathways.npz").toarray()
    m = PathwayMoEPerturb(cfg, grn, gp).to(device); m.load_state_dict(ck["model"]); m.eval()
    loader = make_loader(dataset, test_idx, df, shared, batch_size=32, shuffle=False)
    out = []
    with torch.no_grad():
        for b in loader:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(m(bb).float().cpu().numpy())
    return np.concatenate(out)


def analyze(dataset, splits, variants, seeds, size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_conditions(dataset); Y = stack_arrays(df, "lfc"); deg = stack_arrays(df, "deg_mask")
    shared = build_shared(dataset, df)
    fz = B.Featurizer(dataset, df)
    study = {}
    for split in splits:
        sp_ = make_splits(df, split, seed=SEED); tr, va, te = sp_["train"], sp_["val"], sp_["test"]
        Yte = Y[te]
        res = {"n_test": int(len(te)), "variants": {}, "baselines": {}, "tests": {}}
        # baselines
        for Bl in (B.B1MeanEffect(), B.B2Ridge(), B.B3RidgeBio()):
            Bl.fit(df, tr, val_idx=va, fz=fz, deg=deg)
            pc = per_condition_deg_pearson(Bl.predict(df, te), Yte, k=50)
            mean, lo, hi = bootstrap_ci(pc)
            res["baselines"][Bl.name] = {"deg50": mean, "ci": [lo, hi], "_pc": pc.tolist()}
        # model variants (avg over seeds)
        pc_by_variant = {}
        for variant in variants:
            seed_pc = []
            for seed in seeds:
                name = ckpt_name(dataset, split, variant, seed, size)
                if not (CKPT_DIR / f"{name}.pt").exists():
                    print(f"[analyze] missing {name}, skip seed"); continue
                pred = predict_test(dataset, name, variant, df, shared, te, device)
                seed_pc.append(per_condition_deg_pearson(pred, Yte, k=50))
            if not seed_pc:
                continue
            per_cond_mean, smean, sstd = aggregate_seeds(seed_pc)
            mean, lo, hi = bootstrap_ci(per_cond_mean)
            pc_by_variant[variant] = per_cond_mean
            res["variants"][variant] = {"deg50": mean, "ci": [lo, hi],
                                        "seed_mean": smean, "seed_std": sstd, "n_seeds": len(seed_pc)}
        # paired significance vs none + trrust vs coexpr_lfc
        def pair(a, b):
            if a in pc_by_variant and b in pc_by_variant:
                d, lo, hi, p = paired_bootstrap(pc_by_variant[a], pc_by_variant[b])
                return {"delta": d, "ci": [lo, hi], "p": p}
            return None
        res["tests"] = {f"{a}_vs_{b}": pair(a, b) for a, b in
                        [("trrust", "none"), ("coexpr_lfc", "none"), ("coexpr_lfc", "random"),
                         ("coexpr_lfc", "trrust"), ("coexpr", "none")]}
        study[split] = res
        # console summary
        print(f"\n=== {split} (n_test={len(te)}) ===")
        for k, v in res["baselines"].items():
            print(f"  {k:16s} DEG50={v['deg50']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
        for k, v in res["variants"].items():
            print(f"  GRN={k:12s} DEG50={v['deg50']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] "
                  f"(seed {v['seed_mean']:.3f}±{v['seed_std']:.3f}, n={v['n_seeds']})")
        for k, v in res["tests"].items():
            if v: print(f"  {k}: Δ={v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p={v['p']:.3f}")
    # strip per-condition arrays before saving
    for s in study.values():
        for d in s["baselines"].values(): d.pop("_pc", None)
    (RESULTS_DIR / f"study_{dataset}.json").write_text(json.dumps(study, indent=2))
    print(f"\nwrote {RESULTS_DIR / f'study_{dataset}.json'}")
    return study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe_full")
    ap.add_argument("--splits", nargs="+", default=["unseen_drug"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_FLAGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        for split in a.splits:
            for variant in a.variants:
                for seed in a.seeds:
                    train_one(a.dataset, split, variant, seed, a.epochs, batch=a.batch)
    if not a.train_only:
        analyze(a.dataset, a.splits, a.variants, a.seeds)


if __name__ == "__main__":
    main()
