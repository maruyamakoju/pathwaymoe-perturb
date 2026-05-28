"""Steelman the dynamic-GRN (LRI) idea before declaring it dead.

Step 0 showed the LRI collapsed to a no-op (gate=0.5, posterior collapse) when a
static ground-truth GRN was ALSO present (the gate was redundant). This script gives
the idea its best shot on the novel-target regime:

  E1 nogrn        - no regulatory prior at all (floor)
  E2 gateonly     - latent_grn=True, use_grn_mask=False: the gate is the ONLY
                    regulatory pathway, so the model is FORCED to use it. kl=0.001 (default)
  E3 gateonly_lowkl - same but kl=1e-5 to fight posterior collapse
  E4 static       - static ground-truth GRN mask, no latent (matched baseline)

All use deterministic eval (LRI now uses mu at eval; see pmoe/models/layers.py).
After training, each gate-bearing model is re-run through the degeneracy analyzer.

Outputs results/steelman_results.json
"""
import json
import subprocess
import sys
from pathlib import Path

from pmoe.config import RESULTS_DIR
from pmoe.experiments.train import TrainConfig, train

DATASET = "synthetic_hard_v2"
SPLIT = "unseen_drug"


def common_tc():
    # 20 epochs: give the gate room to develop beyond the 15ep collapse point.
    return TrainConfig(epochs=20, batch_size=64, lr=5e-4, warmup=200,
                       patience=20, eval_every=2)


def main():
    results = {}

    print("\n===== E1: no-GRN floor =====")
    results["nogrn"] = train(DATASET, SPLIT, variant="none", tc=common_tc(),
                             size="small", latent_grn=False, use_grn_mask=False,
                             tag="steel_nogrn")

    print("\n===== E2: gate-only (latent is the only GRN), kl=0.001 =====")
    results["gateonly"] = train(DATASET, SPLIT, variant="ground_truth", tc=common_tc(),
                                size="small", latent_grn=True, use_grn_mask=False,
                                tag="steel_gateonly")

    print("\n===== E3: gate-only, kl=1e-5 (anti-collapse) =====")
    results["gateonly_lowkl"] = train(DATASET, SPLIT, variant="ground_truth", tc=common_tc(),
                                      size="small", latent_grn=True, use_grn_mask=False,
                                      latent_grn_kl_weight=1e-5, tag="steel_gateonly_lowkl")

    print("\n===== E4: static GRN baseline (matched) =====")
    results["static"] = train(DATASET, SPLIT, variant="ground_truth", tc=common_tc(),
                              size="small", latent_grn=False, tag="steel_static")

    summary = {k: {"name": v["name"], "best_val_deg50": v["best_val_deg50"]}
               for k, v in results.items()}
    (RESULTS_DIR / "steelman_results.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("STEELMAN RESULTS (val DEG50)")
    for k, v in summary.items():
        print(f"  {k:20s} {v['best_val_deg50']:.4f}")
    print("=" * 60)

    # Degeneracy re-analysis on the two gate-bearing models (deterministic eval now).
    for tag, run in (("gateonly", results["gateonly"]["name"]),
                     ("gateonly_lowkl", results["gateonly_lowkl"]["name"])):
        print(f"\n--- degeneracy re-analysis: {tag} ---")
        subprocess.run([sys.executable, "code/analyze_gate_degeneracy.py",
                        run, DATASET, f"_{tag}"], check=False)


if __name__ == "__main__":
    main()
