"""Long-Horizon Comparative Study: Static vs. Latent GRN.

Demonstrating the performance crossover on synthetic data with extended training.
"""
import torch
from pmoe.experiments.train import train, TrainConfig
from pmoe.config import RESULTS_DIR, SEED
import json

def run_long_study():
    dataset = "synthetic_hard_v2"
    split = "unseen_drug"
    variant = "ground_truth"
    
    # Plan A: 50 Epochs to show convergence/crossover
    tc = TrainConfig(
        epochs=50, 
        batch_size=64,
        lr=5e-4,
        warmup=200,
        patience=15, # Be more patient
        eval_every=2
    )
    
    print("\n--- Starting Plan A: Long-Horizon Training (50 Epochs) ---")
    
    # Latent GRN
    train(
        dataset=dataset,
        split=split,
        variant=variant,
        tc=tc,
        size="small",
        latent_grn=True,
        tag="long_latent"
    )
    
    # Static GRN
    train(
        dataset=dataset,
        split=split,
        variant=variant,
        tc=tc,
        size="small",
        latent_grn=False,
        tag="long_static"
    )

if __name__ == "__main__":
    run_long_study()
