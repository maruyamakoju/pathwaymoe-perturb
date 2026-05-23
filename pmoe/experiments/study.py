"""GRN-quality controlled study (v2): leakage-free, cluster-bootstrapped, multiple-comparison
corrected, with effect sizes and a GRN-only baseline. Trains via the importable train() (robust).

Per (split): build TRAIN-ONLY data-derived GRNs, train PathwayMoE across GRN variants x seeds,
then analyze with cluster bootstrap (resampling drugs/cell-lines) + Holm correction + effect sizes.
Also evaluates the GRN-propagation baseline across variants (audit E) and linear baselines.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch

from pmoe.config import (RunSpec, TrainConfig, SEED, RESULTS_DIR, GRNVariant, DATA_DERIVED,
                         MIN_MEANINGFUL_EFFECT)
from pmoe.data.loader import load_conditions, load_genes, stack_arrays
from pmoe.data.splits import make_splits
from pmoe.data.dataset import build_shared
from pmoe.priors.grn import build_grn, load_grn
from pmoe.priors.pathways import build_pathways, load_pathways
from pmoe.priors.drugs import load_drug_feats
from pmoe.models import baselines as BL
from pmoe.eval.metrics import deg_pearson_per_condition
from pmoe.eval.stats import (cluster_bootstrap_ci, paired_cluster_bootstrap, holm_correction,
                             effect_summary)
from pmoe.eval.loading import predict_test
from pmoe.experiments.train import train
from pmoe.io import data_manifest


def ensure_static_priors(dataset):
    """Build the priors that do NOT depend on the split (pathways, drugs, non-data-derived GRNs)."""
    pd = __import__("pmoe.priors.pathways", fromlist=["load_pathways"])
    try:
        load_pathways(dataset)
    except Exception:
        build_pathways(dataset)
    load_drug_feats(dataset)
    for v in [GRNVariant.RANDOM, GRNVariant.TRRUST, GRNVariant.TRRUST_WEIGHTED]:
        try:
            load_grn(dataset, v)
        except Exception:
            build_grn(dataset, v)
    # ground_truth only if a base grn_mask exists (synthetic)
    from pmoe.config import priors_dir
    if (priors_dir(dataset) / "grn_mask.npz").exists():
        try: load_grn(dataset, GRNVariant.GROUND_TRUTH)
        except Exception: build_grn(dataset, GRNVariant.GROUND_TRUTH)


def build_split_grns(dataset, df, split, train_idx, variants):
    """Build TRAIN-ONLY data-derived GRNs for this split (leakage-free)."""
    for v in variants:
        if GRNVariant(v) in DATA_DERIVED:
            build_grn(dataset, v, train_idx=train_idx, df=df, split=split)


def run_grid(dataset, splits, variants, seeds, tc: TrainConfig, size="base"):
    df = load_conditions(dataset)
    ensure_static_priors(dataset)
    for split in splits:
        sp_ = make_splits(df, split, seed=SEED)
        build_split_grns(dataset, df, split, sp_["train"], variants)
        for variant in variants:
            for seed in seeds:
                run = RunSpec(dataset, split, variant, seed, size)
                if run.ckpt_path.exists():
                    print(f"[skip] {run.name}", flush=True); continue
                try:
                    train(dataset, split, variant, seed, tc, size)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    print(f"[FAIL] {run.name}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


def analyze(dataset, splits, variants, seeds, size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_conditions(dataset)
    Y = stack_arrays(df, "lfc"); deg = stack_arrays(df, "deg_mask")
    shared = build_shared(dataset, df)
    fz = BL.Featurizer(dataset, df) if hasattr(BL, "Featurizer") else None
    study = {"manifest": data_manifest(dataset), "splits": {}}
    for split in splits:
        sp_ = make_splits(df, split, seed=SEED)
        tr, va, te, groups = sp_["train"], sp_["val"], sp_["test"], sp_["groups"]
        gte = groups[te]; Yte = Y[te]
        res = {"n_test": int(len(te)), "n_clusters": int(len(np.unique(gte))),
               "baselines": {}, "grn_prop": {}, "model": {}, "contrasts": {}}
        pc = {}  # per-condition arrays for paired tests

        # linear baselines
        for Bl in (BL.B1MeanEffect(), BL.B2Ridge(), BL.B3RidgeBio()):
            try:
                Bl.fit(df, tr, val_idx=va, fz=fz, deg=deg)
                v = deg_pearson_per_condition(Bl.predict(df, te), Yte, 50)
                m, lo, hi = cluster_bootstrap_ci(v, gte)
                res["baselines"][Bl.name] = {"deg50": m, "ci": [lo, hi]}
            except Exception as e:
                res["baselines"][getattr(Bl, "name", str(Bl))] = {"error": str(e)[:120]}

        # GRN-propagation baseline across variants (does a GRN-only model benefit from GRN quality?)
        for variant in variants:
            if GRNVariant(variant) == GRNVariant.NONE:
                continue
            try:
                grn = load_grn(dataset, variant, split)
                gp = BL.GRNPropagationBaseline()
                gp.fit(df, tr, grn, val_idx=va, genes=load_genes(dataset))
                v = deg_pearson_per_condition(gp.predict(df, te), Yte, 50)
                m, lo, hi = cluster_bootstrap_ci(v, gte)
                res["grn_prop"][variant] = {"deg50": m, "ci": [lo, hi]}
            except Exception as e:
                res["grn_prop"][variant] = {"error": str(e)[:140]}

        # trained PathwayMoE per variant (avg per-condition over seeds)
        for variant in variants:
            seed_pc = []
            for seed in seeds:
                run = RunSpec(dataset, split, variant, seed, size)
                if not run.ckpt_path.exists():
                    continue
                pred = predict_test(run, dataset, df, shared, te, variant, split, device)
                seed_pc.append(deg_pearson_per_condition(pred, Yte, 50))
            if not seed_pc:
                continue
            arr = np.vstack(seed_pc)
            per_cond = np.nanmean(arr, 0)
            pc[variant] = per_cond
            m, lo, hi = cluster_bootstrap_ci(per_cond, gte)
            res["model"][variant] = {"deg50": m, "ci": [lo, hi], "n_seeds": len(seed_pc),
                                     "seed_mean": float(np.nanmean(arr, 1).mean()),
                                     "seed_std": float(np.nanmean(arr, 1).std())}

        # planned contrasts (paired cluster bootstrap) + Holm correction + effect sizes
        contrasts = [("random", "none"), ("trrust", "none"), ("trrust_weighted", "none"),
                     ("coexpr", "none"), ("coexpr_lfc", "none"), ("ground_truth", "none"),
                     ("coexpr_lfc", "trrust"), ("coexpr_lfc", "random"), ("trrust", "random")]
        raw, details = {}, {}
        for a, b in contrasts:
            if a in pc and b in pc:
                d = paired_cluster_bootstrap(pc[a], pc[b], gte)
                key = f"{a}_vs_{b}"; raw[key] = d["p"]; details[key] = d
        corr = holm_correction(raw) if raw else {}
        for key, d in details.items():
            res["contrasts"][key] = {**d, "p_holm": corr.get(key),
                                     **effect_summary(d["delta"], d["ci"], corr.get(key, 1.0),
                                                      MIN_MEANINGFUL_EFFECT)}
        study["splits"][split] = res
        _print_split(split, res)

    out = RESULTS_DIR / f"study_{dataset}_v2.json"
    out.write_text(json.dumps(study, indent=2, default=float))
    print(f"\nwrote {out}", flush=True)
    return study


def _print_split(split, res):
    print(f"\n=== {split}  (n_test={res['n_test']}, clusters={res['n_clusters']}) ===", flush=True)
    for k, v in res["baselines"].items():
        if "deg50" in v: print(f"  baseline {k:16s} {v['deg50']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
    for k, v in res["grn_prop"].items():
        if "deg50" in v: print(f"  GRNprop[{k:14s}] {v['deg50']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
    for k, v in res["model"].items():
        print(f"  MoE[{k:14s}]   {v['deg50']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] (n={v['n_seeds']})")
    for k, v in res["contrasts"].items():
        flag = "***" if v.get("significant_and_meaningful") else ("*" if v.get("significant") else "")
        print(f"  {k}: Δ={v['delta']:+.3f} p_holm={v.get('p_holm'):.3f} {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe_full")
    ap.add_argument("--splits", nargs="+", default=["unseen_drug"])
    ap.add_argument("--variants", nargs="+",
                    default=["none", "random", "trrust", "trrust_weighted", "coexpr", "coexpr_lfc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--size", default="base")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    tc = TrainConfig(epochs=a.epochs, batch_size=a.batch)
    if not a.analyze_only:
        run_grid(a.dataset, a.splits, a.variants, a.seeds, tc, a.size)
    if not a.train_only:
        analyze(a.dataset, a.splits, a.variants, a.seeds, a.size)


if __name__ == "__main__":
    main()
