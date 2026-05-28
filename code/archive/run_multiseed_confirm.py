"""Multi-seed confirmation: is static >= latent robust across seeds?

The headline single-run comparison (static 0.774 vs latent 0.701 @50ep) needs error
bars before we assert "dynamic GRN does not help." This runs 3 seeds x {static, latent}
on the novel-target regime, 50 epochs each, and reports mean +/- std + the paired gap.

Note: make_splits uses the global SEED, so the train/val split is held fixed; varying
the train seed varies model init + optimization stochasticity (the relevant variance for
a same-data model comparison).

Outputs results/multiseed_confirm.json
"""
import json
from statistics import mean, pstdev

from pmoe.config import RESULTS_DIR
from pmoe.experiments.train import TrainConfig, train

DATASET, SPLIT, VARIANT = "synthetic_hard_v2", "unseen_drug", "ground_truth"
SEEDS = [0, 1, 2]


def tc():
    return TrainConfig(epochs=50, batch_size=64, lr=5e-4, warmup=200,
                       patience=15, eval_every=2)


def main():
    rows = {"static": [], "latent": []}
    for s in SEEDS:
        print(f"\n##### seed {s}: STATIC #####")
        r = train(DATASET, SPLIT, VARIANT, seed=s, tc=tc(), size="small",
                  latent_grn=False, tag=f"ms_static_s{s}")
        rows["static"].append(r["best_val_deg50"])

        print(f"\n##### seed {s}: LATENT #####")
        r = train(DATASET, SPLIT, VARIANT, seed=s, tc=tc(), size="small",
                  latent_grn=True, tag=f"ms_latent_s{s}")
        rows["latent"].append(r["best_val_deg50"])

    gaps = [st - la for st, la in zip(rows["static"], rows["latent"])]
    summary = {
        "seeds": SEEDS,
        "static": rows["static"],
        "latent": rows["latent"],
        "static_mean": mean(rows["static"]), "static_std": pstdev(rows["static"]),
        "latent_mean": mean(rows["latent"]), "latent_std": pstdev(rows["latent"]),
        "paired_gap_static_minus_latent": gaps,
        "gap_mean": mean(gaps), "gap_std": pstdev(gaps),
    }
    (RESULTS_DIR / "multiseed_confirm.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print("MULTISEED CONFIRMATION")
    print(f"  static: {summary['static_mean']:.4f} +/- {summary['static_std']:.4f}  {rows['static']}")
    print(f"  latent: {summary['latent_mean']:.4f} +/- {summary['latent_std']:.4f}  {rows['latent']}")
    print(f"  gap (static-latent): {summary['gap_mean']:+.4f} +/- {summary['gap_std']:.4f}  (all positive => static wins)")
    print("=" * 60)
    print("DONE multiseed")


if __name__ == "__main__":
    main()
