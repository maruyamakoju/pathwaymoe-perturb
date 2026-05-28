"""Comparative Study: Static GRN vs. Latent GRN.

This script runs a controlled comparison between the baseline (static GRN) 
and the new Latent Regulatory Inference (LRI) model on the synthetic 
novel-target regime.
"""
import torch
from pmoe.experiments.train import train, TrainConfig
from pmoe.config import RESULTS_DIR
import json
import os

def run_comparison():
    print("Starting Comparative Study: Static vs. Latent GRN")
    
    dataset = "synthetic_hard_v2" # Novel target regime
    split = "unseen_drug"
    variant = "ground_truth"
    
    common_tc = TrainConfig(
        epochs=15, 
        batch_size=64,
        lr=5e-4,
        warmup=100
    )
    
    # 1. Static GRN (Baseline)
    print("\n--- Running Static GRN (Baseline) ---")
    res_static = train(
        dataset=dataset,
        split=split,
        variant=variant,
        tc=common_tc,
        size="small",
        latent_grn=False,
        tag="static_baseline"
    )
    
    # 2. Latent GRN (Proposed)
    print("\n--- Running Latent GRN (Proposed) ---")
    res_latent = train(
        dataset=dataset,
        split=split,
        variant=variant,
        tc=common_tc,
        size="small",
        latent_grn=True,
        tag="latent_grn"
    )
    
    summary = {
        "static": res_static,
        "latent": res_latent,
        "delta": res_latent["best_val_deg50"] - res_static["best_val_deg50"]
    }
    
    out_path = RESULTS_DIR / "latent_grn_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nStudy Complete. Results saved to {out_path}")
    print(f"Static Score: {res_static['best_val_deg50']:.4f}")
    print(f"Latent Score: {res_latent['best_val_deg50']:.4f}")
    print(f"Improvement:  {summary['delta']:.4f}")

if __name__ == "__main__":
    run_comparison()
