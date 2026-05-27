# Audit: do PathwayMoE's real-Tahoe OOD "wins" survive a leakage-free protocol?

**Date:** 2026-05-27 · **Author:** Claude (Opus 4.7)
**TL;DR: No.** The headline wins in `comparison_tahoe.md` were an artifact. Under the
leakage-free V2 protocol, PathwayMoE does **not** beat strong linear baselines on either
hard OOD split — it *loses significantly* to ridge on unseen_cell_line and ties (slightly
below) the mean-effect baseline on unseen_both.

## Why we audited
`comparison_tahoe.md` claimed PathwayMoE beats baselines decisively on the hard OOD splits
(unseen_cell_line 0.551 vs 0.432; unseen_both 0.512 vs 0.324). But those numbers come from
the **V1 pipeline**, which `AUDIT.md` documents as having **HVG-selection leakage (bug B)**
— HVGs were chosen using test conditions. The leakage-free **V2 study only ever covered
`unseen_drug`**; the two impressive OOD splits were never re-run clean. Having just found a
prior "breakthrough" (latent-GRN) was illusory, we re-ran them under the clean protocol.

## Method
`pmoe.experiments.study` on `tahoe_full`, splits `unseen_cell_line` + `unseen_both`,
`variant=none` (no GRN — the paper already showed GRN doesn't matter), **3 seeds, 25 epochs**,
fixed split seed, deterministic fp32 eval, cluster-bootstrap 95% CIs. Added a **MoE-vs-baseline
paired cluster bootstrap** (kept in its own Holm family so published GRN p-values are unchanged).
*Caveat:* batch 96 exhausted VRAM (24 GB) and hung; we used **batch 48** (12.8 GB). Seed_std is
tiny (0.002–0.006) and the gaps are large, so the conclusion is robust to this.

## Result

| split | clusters | B1 mean-effect | B2 ridge | PathwayMoE (none) | MoE − best baseline |
|---|---|---|---|---|---|
| unseen_cell_line | 10 | 0.850 | **0.868** | 0.824 | **−0.044 vs ridge, p<0.001 (sig)** |
| unseen_both | 33 | **0.607** | 0.390 | 0.581 | −0.026 vs mean-effect, p=0.47 (tie) |

(unseen_drug, from the existing V2 study: MoE 0.630 [0.552,0.700] vs B1 0.608 [0.546,0.664],
B2 0.582 — heavily overlapping CIs, i.e. no significant MoE win there either.)

### The V1 numbers were corrupted on the *baseline* side
| split | baseline | V1 (comparison_tahoe) | V2 clean |
|---|---|---|---|
| unseen_cell_line | B1 mean | 0.432 | **0.850** |
| unseen_cell_line | B2 ridge | 0.396 | **0.868** |
| unseen_both | B1 mean | 0.324 | **0.607** |
| unseen_both | B2 ridge | 0.180 | **0.390** |

Under the clean protocol the baselines are 2× higher and **beat the deep model**. The V1
"win" was PathwayMoE (0.55) over an artificially weak baseline (0.43); fix the protocol and
the baseline (0.85–0.87) overtakes the model.

## Interpretation
This does not contradict the paper — it **confirms its own thesis** more strongly: deep,
biology-structured models do **not** beat linear baselines on OOD perturbation prediction.
For unseen_cell_line, a simple per-drug mean-effect / ridge already captures most signal
(drug effects are largely cell-line-independent for top DEGs), and the 45M-param model
slightly overfits. The honest headline for the project is therefore:
**neither the GRN prior, nor the dynamic-GRN, nor the deep architecture itself provides a
robust OOD advantage over linear baselines on real Tahoe data.**

## Action items
- `comparison_tahoe.md` is **superseded** (V1, leakage-tainted, baselines wrong) — flagged.
- Results merged into `results/study_tahoe_full_v2.json` (all 3 splits; `_audit_note` set).
- Any paper text claiming "PathwayMoE beats baselines on hard OOD" must be removed/reversed.
