"""Figure for the GRN-quality study: DEG-Pearson@50 per GRN variant with 95% CIs, baselines as
reference lines, and significance annotations. Reads results/study_<dataset>.json."""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR

VARIANT_ORDER = ["none", "random", "trrust", "coexpr", "coexpr_lfc"]
VARIANT_LABEL = {"none": "no GRN", "random": "random\n(matched)", "trrust": "TRRUST\n(generic)",
                 "coexpr": "co-expr\n(observ.)", "coexpr_lfc": "co-response\n(LFC)"}


def make(dataset):
    study = json.loads((RESULTS_DIR / f"study_{dataset}.json").read_text())
    splits = list(study)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.6), squeeze=False)
    for ax, split in zip(axes[0], splits):
        s = study[split]
        vs = [v for v in VARIANT_ORDER if v in s["variants"]]
        means = [s["variants"][v]["deg50"] for v in vs]
        los = [s["variants"][v]["ci"][0] for v in vs]
        his = [s["variants"][v]["ci"][1] for v in vs]
        x = np.arange(len(vs))
        err = np.array([np.array(means) - np.array(los), np.array(his) - np.array(means)])
        ax.errorbar(x, means, yerr=err, fmt="o", capsize=4, ms=8, color="#2c7fb8", lw=2)
        ax.set_xticks(x); ax.set_xticklabels([VARIANT_LABEL.get(v, v) for v in vs], fontsize=8)
        # baselines as horizontal lines
        for name, c in [("B1_mean_effect", "#999"), ("B2_ridge", "#d95f02"), ("B3_ridge_bio", "#1b9e77")]:
            if name in s["baselines"]:
                ax.axhline(s["baselines"][name]["deg50"], ls="--", c=c, lw=1,
                           label=name.replace("_", " "))
        ax.set_title(f"{split}\n(n_test={s['n_test']})", fontsize=10)
        ax.set_ylabel("DEG-Pearson@50"); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"GRN-quality controlled study — {dataset}", fontsize=12)
    f = RESULTS_DIR / f"fig_grn_study_{dataset}.png"; fig.tight_layout(); fig.savefig(f, dpi=150)
    print(f"wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="tahoe_full")
    make(ap.parse_args().dataset)
