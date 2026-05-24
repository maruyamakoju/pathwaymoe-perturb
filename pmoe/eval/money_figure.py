"""The single 'money figure': GRN-prior effect (best/true GRN vs no-GRN) across the three regimes,
with cluster-bootstrap 95% CIs and the Δ + significance annotation. Reads the study/mechanism JSONs."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmoe.config import RESULTS_DIR


def _ci(d):  # (mean, lo, hi) from a {"deg50","ci"} dict
    return d["deg50"], d["ci"][0], d["ci"][1]


def build():
    tah = json.loads((RESULTS_DIR / "study_tahoe_full_v2.json").read_text())["splits"]["unseen_drug"]
    syn = json.loads((RESULTS_DIR / "study_synthetic_v2.json").read_text())
    syn = syn["splits"]["unseen_drug"] if "splits" in syn else syn["unseen_drug"]
    mech = json.loads((RESULTS_DIR / "mechanism_synthetic_hard_big.json").read_text())

    # (regime label, none dict, GRN dict, Δ, p, GRN-label)
    panels = [
        ("Real Tahoe-100M\n(shared targets, n=1707, 39 clusters)",
         tah["model"]["none"], tah["model"]["trrust"],
         tah["contrasts"]["trrust_vs_none"]["delta"], tah["contrasts"]["trrust_vs_none"].get("p_holm"),
         "best curated GRN"),
        ("Synthetic shared-target\n(n=270, 9 clusters)",
         syn["model"]["none"], syn["model"]["ground_truth"],
         syn["contrasts"]["ground_truth_vs_none"]["delta"], syn["contrasts"]["ground_truth_vs_none"].get("p_holm"),
         "TRUE GRN"),
        ("Synthetic novel-target\n(n=1230, 39 clusters)",
         mech["conditions"]["none"], mech["conditions"]["gt_mask"],
         mech["contrasts"]["gt_mask_vs_none"]["delta"], mech["contrasts"]["gt_mask_vs_none"].get("p_holm"),
         "TRUE GRN"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(panels)); w = 0.3
    for i, (lab, none_d, grn_d, delta, p, glab) in enumerate(panels):
        for j, (d, color, name) in enumerate([(none_d, "#888", "no GRN"), (grn_d, "#2c7fb8", glab)]):
            m, lo, hi = _ci(d)
            ax.errorbar(i + (j - 0.5) * w, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=10,
                        capsize=5, lw=2, color=color, label=name if i == 0 else None)
        sig = "ns" if (p is None or p >= 0.05) else "p<0.05"
        ax.annotate(f"Δ={delta:+.3f}\n({sig})", (i, max(_ci(none_d)[2], _ci(grn_d)[2]) + 0.02),
                    ha="center", fontsize=9, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in panels], fontsize=8)
    ax.set_ylabel("DEG-Pearson@50  (cluster-bootstrap 95% CI)")
    ax.set_title("Does a GRN prior help OOD perturbation prediction?\n"
                 "GRN (blue) vs no-GRN (grey) — fixed architecture, only the GRN varies", fontsize=11)
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    f = RESULTS_DIR / "fig_money_grn_effect.png"; fig.tight_layout(); fig.savefig(f, dpi=150)
    print(f"wrote {f}")


if __name__ == "__main__":
    build()
