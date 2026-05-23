"""Render the v2 study JSON into a markdown table + figure (DEG-Pearson@50 per GRN variant with
cluster-bootstrap 95% CIs, baselines as reference lines, GRN-propagation baseline overlaid)."""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmoe.config import RESULTS_DIR

VORDER = ["none", "random", "trrust", "trrust_weighted", "coexpr", "coexpr_lfc", "ground_truth"]
VLABEL = {"none": "no GRN", "random": "random", "trrust": "TRRUST", "trrust_weighted": "TRRUST\nweighted",
          "coexpr": "co-expr", "coexpr_lfc": "co-response", "ground_truth": "ground\ntruth"}


def table(study) -> str:
    lines = ["# GRN-quality study (v2, leakage-free, cluster-bootstrap 95% CI)\n",
             f"Data manifest: {json.dumps(study.get('manifest', {}))}\n"]
    for split, s in study["splits"].items():
        lines += [f"\n## {split}  (n_test={s['n_test']}, clusters={s['n_clusters']})\n",
                  "| Model | DEG-Pearson@50 | 95% CI (cluster) |", "|---|---|---|"]
        for k, v in s["baselines"].items():
            if "deg50" in v: lines.append(f"| baseline: {k} | {v['deg50']:.3f} | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] |")
        for k in VORDER:
            if k in s.get("model", {}):
                v = s["model"][k]; lines.append(f"| **MoE / GRN={k}** | **{v['deg50']:.3f}** | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] |")
        for k in VORDER:
            if k in s.get("grn_prop", {}) and "deg50" in s["grn_prop"][k]:
                v = s["grn_prop"][k]; lines.append(f"| GRN-prop / GRN={k} | {v['deg50']:.3f} | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] |")
        lines += ["\n**Contrasts (paired cluster bootstrap, Holm-corrected):**\n",
                  "| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |",
                  "|---|---|---|---|---|"]
        for k, v in s.get("contrasts", {}).items():
            lines.append(f"| {k} | {v['delta']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                         f"{v.get('p_holm', float('nan')):.3f} | {'YES' if v.get('significant_and_meaningful') else 'no'} |")
    return "\n".join(lines) + "\n"


def figure(study, dataset):
    splits = list(study["splits"])
    fig, axes = plt.subplots(1, len(splits), figsize=(5.4 * len(splits), 4.8), squeeze=False)
    for ax, split in zip(axes[0], splits):
        s = study["splits"][split]
        vs = [v for v in VORDER if v in s.get("model", {})]
        x = np.arange(len(vs))
        means = [s["model"][v]["deg50"] for v in vs]
        lo = [s["model"][v]["ci"][0] for v in vs]; hi = [s["model"][v]["ci"][1] for v in vs]
        err = np.array([np.array(means) - lo, np.array(hi) - means])
        ax.errorbar(x, means, yerr=err, fmt="o", ms=9, capsize=4, lw=2, color="#2c7fb8", label="PathwayMoE")
        # GRN-prop baseline overlaid
        gv = [v for v in vs if v in s.get("grn_prop", {}) and "deg50" in s["grn_prop"][v]]
        if gv:
            gx = [vs.index(v) for v in gv]; gm = [s["grn_prop"][v]["deg50"] for v in gv]
            ax.scatter(gx, gm, marker="s", color="#7570b3", s=40, label="GRN-prop (GRN-only)", zorder=3)
        for name, c in [("B1_mean_effect", "#999"), ("B2_ridge", "#d95f02"), ("B3_ridge_bio", "#1b9e77")]:
            if name in s["baselines"] and "deg50" in s["baselines"][name]:
                ax.axhline(s["baselines"][name]["deg50"], ls="--", c=c, lw=1, label=name.replace("_", " "))
        ax.set_xticks(x); ax.set_xticklabels([VLABEL.get(v, v) for v in vs], fontsize=8)
        ax.set_title(f"{split} (n_test={s['n_test']}, clusters={s['n_clusters']})", fontsize=10)
        ax.set_ylabel("DEG-Pearson@50"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"GRN-quality controlled study (leakage-free, cluster CIs) — {dataset}", fontsize=12)
    f = RESULTS_DIR / f"fig_grn_study_{dataset}_v2.png"; fig.tight_layout(); fig.savefig(f, dpi=150)
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="tahoe_full")
    a = ap.parse_args()
    study = json.loads((RESULTS_DIR / f"study_{a.dataset}_v2.json").read_text())
    md = table(study); (RESULTS_DIR / f"study_{a.dataset}_v2.md").write_text(md); print(md)
    print("wrote figure:", figure(study, a.dataset))


if __name__ == "__main__":
    main()
