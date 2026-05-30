"""Drug->target mechanistic-feature experiment.

The one lever the GRN-quality study never tested: instead of a generic gene-gene regulatory prior,
give the model *which gene each drug hits* (external biological knowledge from Tahoe drug_metadata).
The architecture already consumes a per-condition ``target_idx`` (``pathway_moe`` concatenates a
target-gene embedding into the perturbation MLP); it has been inert only because ``target_gene`` was
empty. This experiment populates it (via ``with_targets=True``) and asks: does drug-specific target
information help OOD-to-novel-drug, where GRN priors demonstrably do not?

Design (leakage-free): the target gene is prior knowledge, not derived from test expression, so it is
valid even on unseen_drug. We compare, same architecture / GRN=none:
  * ``none``  — no target info (existing checkpoints, target_idx=-1)
  * ``none + drug-target`` (tag=dtgt) — target_idx populated for the 56/212 drugs with an in-space target
paired cluster bootstrap (resampling drug clusters) + the study's effect-size convention.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pmoe.config import RESULTS_DIR, SEED, RunSpec, TrainConfig
from pmoe.data.dataset import build_shared
from pmoe.data.loader import load_conditions, stack_arrays
from pmoe.data.splits import make_splits
from pmoe.eval.loading import predict_test
from pmoe.eval.metrics import deg_pearson_per_condition
from pmoe.eval.stats import cluster_bootstrap_ci, effect_summary, paired_cluster_bootstrap
from pmoe.experiments.train import train
from pmoe.priors.drugs import build_drug_targets
from pmoe.config import MIN_MEANINGFUL_EFFECT

TAG = "dtgt"


def run_train(dataset, split, seeds, tc, size="base"):
    build_drug_targets(dataset)  # ensure annotation exists
    for seed in seeds:
        run = RunSpec(dataset, split, "none", seed, size, tag=TAG)
        if run.ckpt_path.exists():
            print(f"[skip] {run.name}", flush=True)
            continue
        train(dataset, split, "none", seed, tc, size, with_targets=True, tag=TAG)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _seed_pred(dataset, df, shared, te, split, seeds, size, device, tag):
    """Average per-condition DEG-Pearson over seeds for the none-variant with the given tag."""
    Y = stack_arrays(df, "lfc")[te]
    accs = []
    for seed in seeds:
        run = RunSpec(dataset, split, "none", seed, size, tag=tag)
        if not run.ckpt_path.exists():
            continue
        pred = predict_test(run, dataset, df, shared, te, "none", split, device)
        accs.append(deg_pearson_per_condition(pred, Y, 50))
    if not accs:
        return None
    return np.nanmean(np.vstack(accs), 0)


def analyze(dataset, split, seeds, size="base", device="cpu"):
    df = load_conditions(dataset)
    sp_ = make_splits(df, split, seed=SEED)
    te, groups = sp_["test"], sp_["groups"]
    gte = groups[te]

    shared_no = build_shared(dataset, df, with_targets=False)
    shared_tg = build_shared(dataset, df, with_targets=True)

    pc_none = _seed_pred(dataset, df, shared_no, te, split, seeds, size, device, tag="")
    pc_dtgt = _seed_pred(dataset, df, shared_tg, te, split, seeds, size, device, tag=TAG)

    res = {"n_test": int(len(te)), "n_clusters": int(len(np.unique(gte))), "model": {}}
    for name, pc in [("none", pc_none), ("none+drug_target", pc_dtgt)]:
        if pc is None:
            continue
        m, lo, hi = cluster_bootstrap_ci(pc, gte)
        res["model"][name] = {"deg50": m, "ci": [lo, hi]}
    if pc_none is not None and pc_dtgt is not None:
        d = paired_cluster_bootstrap(pc_dtgt, pc_none, gte)
        res["contrast_dtgt_vs_none"] = {**d, **effect_summary(
            d["delta"], d["ci"], d["p"], MIN_MEANINGFUL_EFFECT)}

    print(f"\n=== {split} drug->target (n_test={res['n_test']}, clusters={res['n_clusters']}) ===")
    for k, v in res["model"].items():
        print(f"  {k:18s} {v['deg50']:.4f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
    if "contrast_dtgt_vs_none" in res:
        c = res["contrast_dtgt_vs_none"]
        print(f"  dtgt vs none: Δ={c['delta']:+.4f} ci=[{c['ci'][0]:.3f},{c['ci'][1]:.3f}] "
              f"p={c['p']:.3f} sig={c.get('significant')}")

    out = RESULTS_DIR / f"drug_target_{dataset}_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe_full")
    ap.add_argument("--split", default="unseen_drug")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--size", default="base")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--eval-device", default="cpu", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    tc = TrainConfig(epochs=a.epochs, batch_size=a.batch)
    if not a.analyze_only:
        run_train(a.dataset, a.split, a.seeds, tc, a.size)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.eval_device == "auto" else a.eval_device
    analyze(a.dataset, a.split, a.seeds, a.size, device=dev)


if __name__ == "__main__":
    main()
