"""Mechanism comparison: does injecting the GRN as SOFT message-passing help where the hard
attention-MASK did not? Uses the ground-truth GRN on the well-powered novel-target synthetic
(the regime where the GRN should matter most + full statistical power).

Conditions (all reuse the SAME ground_truth GRN where a GRN is used):
  none        no GRN at all                         (reuse existing checkpoints)
  gt_mask     GRN as attention mask (current)       (reuse existing ground_truth checkpoints)
  gt_prop     GRN as soft propagation, mask OFF     (new: tag=proponly, use_grn_mask=False)
  gt_maskprop GRN as mask + propagation             (new: tag=maskprop)
Analysis: cluster-bootstrap CIs + paired cluster bootstrap vs none, Holm-corrected, effect sizes.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from pmoe.config import RunSpec, TrainConfig, SEED, RESULTS_DIR, MIN_MEANINGFUL_EFFECT
from pmoe.data.loader import load_conditions, stack_arrays
from pmoe.data.splits import make_splits
from pmoe.data.dataset import build_shared
from pmoe.eval.metrics import deg_pearson_per_condition
from pmoe.eval.stats import cluster_bootstrap_ci, paired_cluster_bootstrap, holm_correction, effect_summary
from pmoe.eval.loading import predict_test
from pmoe.experiments.train import train

# (label, variant, tag, grn_propagation, use_grn_mask_override)
CONDITIONS = [
    ("none",        "none",         "",         False, None),
    ("gt_mask",     "ground_truth", "",         False, None),     # mask on (variant!=none)
    ("gt_prop",     "ground_truth", "proponly", True,  False),    # propagation only
    ("gt_maskprop", "ground_truth", "maskprop", True,  True),     # mask + propagation
]


def run(dataset, split, seeds, epochs, batch, size="base"):
    tc = TrainConfig(epochs=epochs, batch_size=batch)
    for label, variant, tag, prop, mask in CONDITIONS:
        for seed in seeds:
            r = RunSpec(dataset, split, variant, seed, size, tag=tag)
            if r.ckpt_path.exists():
                print(f"[skip] {r.name}", flush=True); continue
            print(f"[train] {label}: {r.name}", flush=True)
            try:
                train(dataset, split, variant, seed, tc, size,
                      grn_propagation=prop, use_grn_mask=mask, tag=tag)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e:
                import traceback; print(f"[FAIL] {r.name}: {e}", flush=True); traceback.print_exc()


def analyze(dataset, split, seeds, size="base", eval_device="auto"):
    """Cluster-bootstrap analyze of mechanism conditions.

    ``eval_device``: "auto" picks cuda when available, "cpu" forces CPU eval. CPU eval is
    SLOW but bitwise-deterministic across process invocations. On synthetic_hard_big the
    per-condition DEG50 metric exhibits ~5% run-to-run variance on CUDA (suspected MoE
    index_add_ atomic-add ordering); use eval_device="cpu" for publication-grade numbers.
    See AUDIT.md item N.
    """
    if eval_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = eval_device
    df = load_conditions(dataset); Y = stack_arrays(df, "lfc")
    shared = build_shared(dataset, df)
    sp_ = make_splits(df, split, seed=SEED); te, groups = sp_["test"], sp_["groups"]
    gte, Yte = groups[te], Y[te]
    pc, res = {}, {"dataset": dataset, "split": split, "n_test": int(len(te)),
                   "n_clusters": int(len(np.unique(gte))), "conditions": {}, "contrasts": {}}
    for label, variant, tag, prop, mask in CONDITIONS:
        seed_pc = []
        for seed in seeds:
            r = RunSpec(dataset, split, variant, seed, size, tag=tag)
            if not r.ckpt_path.exists():
                continue
            pred = predict_test(r, dataset, df, shared, te, variant, split, device)
            seed_pc.append(deg_pearson_per_condition(pred, Yte, 50))
        if not seed_pc:
            print(f"[analyze] no checkpoints for {label}"); continue
        per_cond = np.nanmean(np.vstack(seed_pc), 0); pc[label] = per_cond
        m, lo, hi = cluster_bootstrap_ci(per_cond, gte)
        res["conditions"][label] = {"deg50": m, "ci": [lo, hi], "n_seeds": len(seed_pc)}
    raw, details = {}, {}
    for label in ("gt_mask", "gt_prop", "gt_maskprop"):
        if label in pc and "none" in pc:
            d = paired_cluster_bootstrap(pc[label], pc["none"], gte)
            raw[f"{label}_vs_none"] = d["p"]; details[f"{label}_vs_none"] = d
    if "gt_prop" in pc and "gt_mask" in pc:
        d = paired_cluster_bootstrap(pc["gt_prop"], pc["gt_mask"], gte)
        raw["gt_prop_vs_gt_mask"] = d["p"]; details["gt_prop_vs_gt_mask"] = d
    corr = holm_correction(raw) if raw else {}
    for k, d in details.items():
        res["contrasts"][k] = {**d, "p_holm": corr.get(k),
                               **effect_summary(d["delta"], d["ci"], corr.get(k, 1.0), MIN_MEANINGFUL_EFFECT)}
    (RESULTS_DIR / f"mechanism_{dataset}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n=== mechanism: {dataset}/{split} (n_test={res['n_test']}, clusters={res['n_clusters']}) ===")
    for k, v in res["conditions"].items():
        print(f"  {k:12s} DEG50={v['deg50']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] (n={v['n_seeds']})")
    for k, v in res["contrasts"].items():
        flag = "***" if v.get("significant_and_meaningful") else ("*" if v.get("significant") else "")
        print(f"  {k}: Δ={v['delta']:+.3f} p_holm={v.get('p_holm'):.3f} {flag}")
    print(f"wrote {RESULTS_DIR / f'mechanism_{dataset}.json'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic_hard_big")
    ap.add_argument("--split", default="unseen_drug")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1, 2])
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--eval-device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Evaluation device. 'cpu' forces deterministic CPU eval (slow but bit-reproducible).")
    a = ap.parse_args()
    if not a.analyze_only:
        run(a.dataset, a.split, a.seeds, a.epochs, a.batch)
    if not a.train_only:
        analyze(a.dataset, a.split, a.seeds, eval_device=a.eval_device)


if __name__ == "__main__":
    main()
