"""Matched 50-epoch STATIC run, identical config to long_latent.

The original long_static was interrupted at epoch 4, making the headline
'Latent 0.70 vs Static 0.44' comparison invalid (50ep vs 4ep). This reruns
static for the full 50 epochs so the comparison is apples-to-apples.
"""
from pmoe.experiments.train import train, TrainConfig


def main():
    tc = TrainConfig(
        epochs=50,
        batch_size=64,
        lr=5e-4,
        warmup=200,
        patience=15,
        eval_every=2,
    )
    print("--- Matched STATIC 50ep (fair comparison vs long_latent) ---")
    res = train(
        dataset="synthetic_hard_v2",
        split="unseen_drug",
        variant="ground_truth",
        tc=tc,
        size="small",
        latent_grn=False,
        tag="long_static",
    )
    print(f"DONE long_static best_val_deg50={res['best_val_deg50']:.4f}")


if __name__ == "__main__":
    main()
