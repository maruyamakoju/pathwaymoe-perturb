"""Regenerate fig_comparison_tahoe.png from the leakage-free study_tahoe_full_v2.json.

The old fig_comparison_tahoe.png was built from the V1 comparison_tahoe.json with
HVG-leakage and corrupted-low baselines (it showed PathwayMoE 'winning' on the OOD
splits). This redraws B1/B2 vs PathwayMoE(no-GRN) from the clean V2 results, with
cluster-bootstrap CIs, so the figure matches REAL_TAHOE_OOD_AUDIT.md.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pmoe.config import RESULTS_DIR

SPLITS = ["unseen_drug", "unseen_cell_line", "unseen_both"]


def main():
    d = json.loads((RESULTS_DIR / "study_tahoe_full_v2.json").read_text())["splits"]
    series = [("B1 mean-effect", "baselines", "B1_mean_effect", "#bbbbbb"),
              ("B2 ridge", "baselines", "B2_ridge", "#7fae7f"),
              ("PathwayMoE (no GRN)", "model", "none", "#2c7fb8")]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SPLITS)); w = 0.26
    for i, (label, sect, key, color) in enumerate(series):
        means, los, his = [], [], []
        for s in SPLITS:
            o = d[s][sect].get(key, {})
            m = o.get("deg50", np.nan); ci = o.get("ci", [np.nan, np.nan])
            means.append(m); los.append(m - ci[0]); his.append(ci[1] - m)
        ax.bar(x + (i - 1) * w, means, w, label=label, color=color,
               yerr=[los, his], capsize=3, error_kw=dict(lw=1))
        for xi, m in zip(x + (i - 1) * w, means):
            if not np.isnan(m):
                ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(SPLITS)
    ax.set_ylabel("DEG-Pearson@50 (cluster-bootstrap 95% CI)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Real Tahoe-100M OOD: deep PathwayMoE does NOT beat linear baselines\n"
                 "(leakage-free V2 protocol; ridge wins unseen_cell_line, p<0.001)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    f = RESULTS_DIR / "fig_comparison_tahoe.png"
    fig.tight_layout(); fig.savefig(f, dpi=150)
    print(f"wrote {f}")
    # print the values plotted for the record
    for s in SPLITS:
        b1 = d[s]["baselines"]["B1_mean_effect"]["deg50"]
        b2 = d[s]["baselines"]["B2_ridge"]["deg50"]
        mo = d[s]["model"]["none"]["deg50"]
        print(f"  {s:18s} B1={b1:.3f} B2={b2:.3f} MoE={mo:.3f}")


if __name__ == "__main__":
    main()
