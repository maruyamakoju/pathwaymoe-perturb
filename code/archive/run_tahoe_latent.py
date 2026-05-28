"""Real-World Scale-up: Latent GRN on Tahoe-100M subset.
"""
import torch
from pmoe.experiments.train import train, TrainConfig
from pmoe.config import SEED

def run_tahoe_latent():
    dataset = "tahoe_full"
    split = "unseen_drug"
    variant = "trrust" # Start with curated prior
    
    tc = TrainConfig(
        epochs=30,
        batch_size=128,
        lr=3e-4,
        warmup=500,
        grad_ckpt=True # Real data is large
    )
    
    print("\n--- Starting Plan B: Real-World Scale-up (Tahoe-22M) ---")
    
    train(
        dataset=dataset,
        split=split,
        variant=variant,
        tc=tc,
        size="base",
        latent_grn=True,
        tag="tahoe_latent"
    )

if __name__ == "__main__":
    run_tahoe_latent()
