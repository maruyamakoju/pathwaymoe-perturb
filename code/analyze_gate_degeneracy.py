"""Step 0: Is the LRI gate a real, condition-specific signal or a degenerate artifact?

This is the decision point for the whole MoA direction. We test four things on the
trained synthetic latent checkpoint:

  (A) SATURATION   - distribution of all gate entries. If ~all are ~1.0 (or ~0.5),
                     the >0.8 thresholding in visualize_gates is meaningless.
  (B) SPECIFICITY  - does the gate matrix actually change across drugs? If gates are
                     ~identical regardless of perturbation, the LRI ignores its input.
  (C) NOISE>SIGNAL - the LRI samples eps even in eval(). We compare gate variance from
                     pure sampling noise (same drug, many samples) vs across-drug
                     variance. If noise >= drug-signal, "discovered edges" are random.
  (D) RANK-1       - gate[i,j]=sigmoid(a_i+b_j). We check whether 'top edges' are just
                     the Cartesian product of high-activity sources x high-receptivity
                     targets (i.e. structurally forced, not learned topology).

Outputs results/interpretability/gate_degeneracy_report.json and prints a verdict.
"""
import json
from pathlib import Path

import numpy as np
import torch

from pmoe.config import (ArchitectureConfig, CKPT_DIR, GRNConfig, ModelConfig,
                         MoEConfig, PerturbationConfig, RESULTS_DIR)
from pmoe.data.dataset import build_shared
from pmoe.data.loader import load_conditions, load_genes
from pmoe.models.pathway_moe import PathwayMoEPerturb

import sys

RUN = "synthetic_hard_v2__unseen_drug__small__ground_truth__long_latent"
DATASET = "synthetic_hard_v2"
torch.manual_seed(0)


def dict_to_cfg(d):
    return ModelConfig(
        arch=ArchitectureConfig(**d["arch"]),
        pert=PerturbationConfig(**d["pert"]),
        grn=GRNConfig(**d["grn"]),
        moe=MoEConfig(**d["moe"]),
    )


def build_pert_emb(model, shared, idx):
    cb = torch.from_numpy(shared["chemberta"][idx]).unsqueeze(0).float()
    ti = torch.tensor([shared["target_idx"][idx]])
    dl = torch.tensor([shared["dose_log"][idx]]).float()
    with torch.no_grad():
        temb = model.gene_emb(ti.clamp(min=0))
        temb = torch.where((ti >= 0).unsqueeze(-1), temb, torch.zeros_like(temb))
        feats = torch.cat([cb, temb, dl.unsqueeze(-1)], dim=-1)
        return model.pert_mlp(feats)  # (1, D)


def lri_params(model, pert_emb):
    """Deterministic mean gate + the per-gene a (TF activity) / b (receptivity)."""
    n = model.cfg.n_genes
    with torch.no_grad():
        params = model.lri.infer(pert_emb).reshape(1, n, 2, 2)
        mu = params[..., 0]          # (1, N, 2)
        logvar = params[..., 1]
        a = mu[0, :, 0].numpy()      # TF activity
        b = mu[0, :, 1].numpy()      # receptivity
        gate_mean = torch.sigmoid(mu[0, :, 0:1] + mu[0, :, 1].unsqueeze(0)).numpy()
        return gate_mean, a, b, logvar[0].numpy()


def sample_gate(model, pert_emb):
    """Force the LRI to sample eps (train mode) to quantify how noisy the *original*
    (buggy) eval-time extraction was. After the eval-determinism fix, model.eval()
    gives identical gates, so we set just the LRI module to train() here."""
    was_training = model.lri.training
    model.lri.train()
    try:
        with torch.no_grad():
            gate, _ = model.lri(pert_emb)  # stochastic
            return gate[0].numpy()
    finally:
        model.lri.train(was_training)


def main(run=RUN, dataset=DATASET, report_suffix=""):
    global DATASET
    DATASET = dataset
    ckpt_path = CKPT_DIR / f"{run}.pt"
    if not ckpt_path.exists():
        print(f"FATAL: checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict_to_cfg(ckpt["cfg"])
    model = PathwayMoEPerturb(cfg)  # grn=None ok; learnable params loaded below
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = load_conditions(DATASET)
    genes = load_genes(DATASET)
    shared = build_shared(DATASET, df)
    n = len(genes)

    # pick a panel of distinct drugs
    drugs = list(df["treatment"].unique())
    rng = np.random.default_rng(0)
    panel = ["trametinib", "nutlin-3"] + [d for d in rng.choice(drugs, 10, replace=False)
                                          if d not in ("trametinib", "nutlin-3")]
    panel = panel[:12]
    idx_of = {d: int(df[df["treatment"] == d].index[0]) for d in panel if (df["treatment"] == d).any()}

    report = {"run": run, "n_genes": n, "panel": list(idx_of.keys())}

    # ---- (A) SATURATION + (D) rank-1, per drug (deterministic mean gate) ----
    mean_gates = {}
    a_vecs, b_vecs = {}, {}
    for d, idx in idx_of.items():
        pe = build_pert_emb(model, shared, idx)
        g, a, b, _ = lri_params(model, pe)
        mean_gates[d] = g
        a_vecs[d], b_vecs[d] = a, b

    # saturation stats on trametinib's deterministic gate
    g0 = mean_gates[list(idx_of)[0]]
    report["saturation"] = {
        "drug": list(idx_of)[0],
        "mean": float(g0.mean()), "std": float(g0.std()),
        "frac_gt_0.8": float((g0 > 0.8).mean()),
        "frac_gt_0.5": float((g0 > 0.5).mean()),
        "frac_lt_0.2": float((g0 < 0.2).mean()),
        "pctile": {p: float(np.percentile(g0, p)) for p in (1, 25, 50, 75, 99)},
    }

    # ---- (B) SPECIFICITY: across-drug variation of the deterministic gate ----
    stack = np.stack([mean_gates[d] for d in idx_of])  # (D, N, N)
    across_drug_std = stack.std(axis=0)  # (N,N) std per edge across drugs
    overall_std = stack.std()
    within_drug_spatial_std = float(np.mean([g.std() for g in mean_gates.values()]))
    report["specificity"] = {
        "mean_across_drug_std_per_edge": float(across_drug_std.mean()),
        "within_drug_spatial_std": within_drug_spatial_std,
        # if across-drug variation << the gate's own edge-to-edge variation, the gate is
        # essentially the same matrix regardless of which drug -> not condition-specific.
        "specificity_ratio_across_over_within": float(across_drug_std.mean() /
                                                      (within_drug_spatial_std + 1e-12)),
        "overall_gate_std": float(overall_std),
        "tram_vs_nutlin_mean_abs_diff": float(np.abs(mean_gates.get("trametinib", g0) -
                                                     mean_gates.get("nutlin-3", g0)).mean()),
    }

    # ---- (C) NOISE vs SIGNAL: sampling variance for ONE drug, K samples ----
    d0 = list(idx_of)[0]
    pe0 = build_pert_emb(model, shared, idx_of[d0])
    samples = np.stack([sample_gate(model, pe0) for _ in range(8)])  # (8,N,N)
    sampling_std_per_edge = samples.std(axis=0)  # noise from eps alone
    report["noise_vs_signal"] = {
        "_note": "sampling_std measured with LRI forced to train() (eps on); quantifies how "
                 "noisy the ORIGINAL buggy eval-time extraction was vs real across-drug signal.",
        "sampling_std_mean": float(sampling_std_per_edge.mean()),
        "across_drug_std_mean": float(across_drug_std.mean()),
        "ratio_signal_over_noise": float(across_drug_std.mean() /
                                         (sampling_std_per_edge.mean() + 1e-12)),
    }

    # ---- (D) RANK-1 confound: do top-receptivity targets dominate 'top edges'? ----
    # For trametinib mean gate, take top-100 edges and see how concentrated targets are,
    # then check those targets == argmax(b) (receptivity), independent of source.
    flat_idx = np.argsort(g0.ravel())[::-1][:100]
    top_src = flat_idx // n
    top_tgt = flat_idx % n
    top_b_targets = set(np.argsort(b_vecs[d0])[::-1][:10].tolist())
    report["rank1_confound"] = {
        "unique_targets_in_top100": int(len(set(top_tgt.tolist()))),
        "unique_sources_in_top100": int(len(set(top_src.tolist()))),
        "frac_top_edges_into_top10_receptivity_genes":
            float(np.mean([t in top_b_targets for t in top_tgt])),
    }

    # ---- verdict ----
    sig_noise = report["noise_vs_signal"]["ratio_signal_over_noise"]
    sat = report["saturation"]["frac_gt_0.8"]
    spec = report["specificity"]["mean_across_drug_std_per_edge"]
    verdict = []
    if sat > 0.9:
        verdict.append(f"SATURATED: {sat:.0%} of gates >0.8 -> thresholding meaningless")
    if spec < 0.02:
        verdict.append(f"NON-SPECIFIC: across-drug std={spec:.4f} ~ gate barely moves with drug")
    if sig_noise < 1.5:
        verdict.append(f"NOISE-DOMINATED: signal/noise={sig_noise:.2f} -> 'edges' ~ sampling noise")
    if report["rank1_confound"]["frac_top_edges_into_top10_receptivity_genes"] > 0.7:
        verdict.append("RANK-1 CONFOUND: top edges are just high-receptivity hub columns")
    report["VERDICT"] = verdict if verdict else ["gates appear to carry usable signal"]

    out = RESULTS_DIR / "interpretability" / f"gate_degeneracy_report{report_suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=" * 70)
    print("STEP 0 GATE DEGENERACY REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70)
    print("VERDICT:")
    for v in report["VERDICT"]:
        print("  -", v)


if __name__ == "__main__":
    _run = sys.argv[1] if len(sys.argv) > 1 else RUN
    _ds = sys.argv[2] if len(sys.argv) > 2 else DATASET
    _suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    main(_run, _ds, _suffix)
