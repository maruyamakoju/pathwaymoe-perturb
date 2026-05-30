# When does gene-regulatory structure help perturbation prediction? A controlled study from synthetic ground truth to 22M real cells

*Working draft. All numbers are populated from `results/study_*_v2.json`,
`mechanism_synthetic_hard_big.json`, and the OOD audit (`REAL_TAHOE_OOD_AUDIT.md`); figures in
`results/fig_*`. Reproduce with `reproduce.ps1` on the `pmoe/` package (44 tests).*

## Abstract

Network-structured inductive biases (GRN-masked attention, pathway Mixture-of-Experts) are widely
proposed to improve out-of-distribution (OOD) perturbation-response prediction, motivated by the
finding that large foundation models do not beat linear baselines (Nature Methods 2025). We ask a
sharper question than "does it help?": **when, and on which kind of GRN, does it help?** We hold the
architecture fixed and vary only the GRN prior across a quality spectrum — none, random (matched
density), generic curated (TRRUST), data-derived co-expression, and data-derived co-response (LFC
correlation) — measuring effect on OOD DEG-Pearson with bootstrap CIs and paired significance tests.
We run this on (i) a 22.6M-cell, 8,875-condition subset of Tahoe-100M with deep pseudobulk and
(ii) synthetic data whose generative process *is* GRN propagation, in two regimes (targets shared
with vs. held out from training). **Result:** with leakage-free GRNs and cluster-bootstrap statistics,
**no GRN prior — generic, weighted, or data-derived — significantly improves real-data OOD prediction**
(all |Δ DEG-Pearson|≤0.007, Holm-p>0.1). Moreover, under the same leakage-free protocol the deep
PathwayMoE does **not** robustly beat the linear baselines either: across three OOD splits it loses
to ridge on unseen_cell_line (0.824 vs 0.868, Holm-p=0.001), ties the mean-effect baseline on unseen_both
(0.581 vs 0.607, ns), and shows only a small non-significant edge on unseen_drug (0.629 vs 0.601,
overlapping CIs). On synthetic
data we localize why: when test perturbations **share targets** with training, even the *true* GRN is
redundant (the model learns the response directly, true GRN +0.014, ns); when targets are **novel**,
the true GRN gives a **meaningful but non-significant** gain (Δ=+0.079, Holm-p=0.57) in a regime that
is near-unpredictable, and the **injection mechanism** (hard attention mask vs soft message-passing)
does not matter. We conclude that a structured GRN prior is not an effective inductive bias for OOD
perturbation prediction where prediction is feasible (shared targets, incl. Tahoe) — it is redundant
there — and only *hints* at helping for novel targets, where prediction is anyway near-impossible. En
route we caught and fixed three result-invalidating bugs (test-set GRN leakage; a silent
"train-without-GRN" fallback; bf16 metric nondeterminism), underscoring the audit. We release a typed,
tested, leakage-audited benchmarking package (`pmoe/`).

## 1. Introduction

Predicting how single cells respond to unseen perturbations — new drugs, new cell lines, or both — is
a central goal of computational cell biology. Yet a 2025 *Nature Methods* benchmark reported that
billion-parameter single-cell foundation models do **not** beat simple linear baselines (ridge
regression, per-perturbation mean effects) on out-of-distribution (OOD) perturbation-response
prediction. A natural and widely pursued response is to inject **biology-structured inductive biases**:
restrict attention to gene-regulatory-network (GRN) edges, group computation by pathways (Mixture-of-
Experts), or propagate perturbation signals along a regulatory graph (GEARS-style message passing). The
implicit hypothesis is that encoding *which genes regulate which* gives a model the right scaffold to
generalize where a flexible black box cannot.

This hypothesis is rarely tested cleanly. Reported gains conflate three distinct factors: (i) the
**structural prior** (the GRN) versus the **architecture** that carries it; (ii) a **generic** curated
GRN versus one that is actually informative for the task; and (iii) genuine signal versus **leakage** —
data-derived networks and feature-selection steps that quietly see the test set. As a result it is
unclear whether a GRN prior helps, and if so, on *what kind* of GRN and in *what regime*.

We therefore ask a sharper, controlled question than "does biology help?": **holding the architecture
fixed, does GRN *quality* move OOD accuracy?** We sweep the prior across a quality spectrum — none,
random (density-matched), generic curated (TRRUST), data-derived co-expression and co-response, and, on
synthetic data, the *true* generative GRN — changing only *which* edges the model may attend to. We
evaluate on (i) a 22.6M-cell, 8,875-condition subset of Tahoe-100M with real priors and (ii) synthetic
data whose generative process *is* GRN propagation, in shared-target and novel-target regimes, with a
leakage-free, cluster-bootstrap, Holm-corrected protocol and a pre-registered minimum meaningful effect.

**Contributions.** (1) A controlled GRN-*quality* study, not just a presence/absence ablation, spanning
synthetic ground truth to 22M real cells. (2) A leakage-audited, well-powered statistical protocol
(train-only data-derived GRNs, cluster bootstrap over drugs/cell-lines with a +1/(B+1) continuity
correction on p-values, Holm correction, deterministic fp32 eval) that caught and fixed three
result-invalidating bugs. (3) The central finding: **no GRN
prior — generic, weighted, or data-derived — significantly improves real-data OOD prediction**, and on
synthetic data we localize *why* (redundant for shared targets, insufficient for novel ones). (4) Under
the same protocol, **the deep PathwayMoE architecture itself does not beat the linear baselines** on
real OOD (it loses to ridge on unseen_cell_line and ties the mean-effect baseline on unseen_both),
reinforcing the foundation-model result. (5) A typed, tested, leakage-audited benchmarking package
(`pmoe/`) for reproducible follow-up.

## 1.1 Related work

**Single-cell perturbation prediction and the linear-baseline gap.** Large single-cell foundation
models (e.g. transformer pretraining over millions of cells) were proposed to transfer to perturbation
response, but a 2025 benchmark found they do not outperform simple linear baselines on OOD splits
(Ahlmann-Eltze et al., *Nature Methods* 2025). This motivates the question of what inductive bias, if
any, closes the gap.

**Biology-structured inductive biases.** A prominent line injects gene-regulatory structure:
GEARS propagates perturbation signal over a GRN/GO graph (Roohani et al., *Nat. Biotechnol.* 2024);
scGPT and related models add pathway/GRN attention or tokens (Cui et al., *Nat. Methods* 2024; Chen &
Zou, *Nat. Biomed. Eng.* 2025); and pathway Mixture-of-Experts route computation by curated gene sets
(Lotfollahi et al., *Nat. Cell Biol.* 2023). Curated networks (TRRUST; Han et al., *NAR* 2018;
DoRothEA/CollecTRI; Garcia-Alonso et al., 2019; Müller-Dott et al., *NAR* 2023) and data-derived
co-expression graphs are the usual priors. Most reports show a gain from *adding* such structure but do
not isolate it from the surrounding architecture, nor vary GRN *quality* as a controlled variable.

**Evaluation pitfalls in single-cell ML.** Recent critiques highlight leakage and weak baselines:
test-aware feature/graph construction, condition-level dependence ignored by naive bootstrap, and
metrics dominated by easy genes (Peidli et al., *Nat. Methods* 2024). Our protocol (train-only
data-derived GRNs, cluster bootstrap over drugs/cell-lines, Holm correction, a pre-registered minimum
effect, deterministic eval) is designed against exactly these failure modes; we report three concrete
bugs that, uncaught, would each have produced a confident false positive.

**Our position.** Rather than ask "does biology help?", we hold the architecture fixed and ask whether
GRN *quality* moves OOD accuracy — and, separately, whether the deep architecture beats linear baselines
at all under a leakage-free protocol. Both answers are negative on real Tahoe data.

## 2. Methods

### 2.1 Model (held fixed)
PathwayMoE-Perturb (≈45M params): gene-token decoder with (a) self-attention where half the heads are
restricted to GRN edges (two-pass SDPA, no Triton), (b) perturbation cross-attention (ChemBERTa SMILES
embedding as query context), (c) top-2 Mixture-of-Experts FFN over Reactome pathways. Predicts
pseudobulk log-fold-change over N genes. Only the GRN mask is varied; everything else is identical.

### 2.2 GRN-quality spectrum (independent variable)
All masks share the gene space and are matched to TRRUST's edge count (only *which* edges differ):
- **none** — GRN masking disabled (dense attention).
- **random** — random edges, matched density (controls for "any sparse mask").
- **trrust** — TRRUST v2 curated human TF→target, binary (generic prior; symbol→Ensembl mapped).
- **trrust_weighted** — same edges, signed/weighted (Activation +, Repression −) → weighted attention bias (audit F).
- **coexpr** — top gene–gene corr of pseudobulk expression, **train conditions only** (data-derived, observational).
- **coexpr_lfc** — top gene–gene corr of LFC across **train conditions only** (data-derived, perturbation co-response).
- **ground_truth** (synthetic only) — the true generative GRN (upper bound).

**Leakage control (audit A/B):** data-derived GRNs are built from the split's TRAIN indices only and
are split-specific (`grn_<v>__<split>.npz`, recorded `leakage_safe=true`); a regression test asserts
that perturbing test-set rows does not change the GRN.

### 2.3 Data
- **Synthetic** (positive control): 2,000 genes, 20 pathways, known signed GRN with pathway-localized
  edges; perturbation = on-target effect (scaled by cell-line target baseline) propagated through the
  GRN; drug→target hubs shared within chemical class so chemistry is predictive. Ground-truth GRN
  available as an upper-bound variant.
- **Tahoe-100M subset** (real): 803/3,388 parquet shards → **22.66M cells**; memory-bounded streaming
  pseudobulk per (cell line, drug) with matched DMSO_TF controls; **8,875 conditions** (≥50 cells;
  mean 2,480 cells/condition) × **2,000 HVGs** selected by cross-condition LFC variance. Real priors:
  TRRUST v2 GRN + Reactome pathways (Ensembl-aligned).

### 2.4 Splits, metric, statistics
Leakage-controlled splits (split seed FIXED so the test set is identical across all variants and model
seeds): unseen_drug (Butina/Tanimoto<0.8 dedup), unseen_cell_line, unseen_both. Primary metric:
**DEG-Pearson@50** = Pearson on the 50 genes with largest |true LFC| per condition (dataset-agnostic).
Reference baselines: B1 mean-effect, B2/B3 ridge; plus a **GRN-propagation baseline** (audit E) — a
GRN-only linear message-passing model with no deep net, evaluated under each GRN variant to test
whether *any* model benefits from a given GRN.
Statistics (audit C/D/G): each config trained with **3 seeds**; point estimate = per-condition mean
over seeds; **cluster bootstrap** (resampling the grouping variable — drugs for unseen_drug/both,
cell-lines for unseen_cell_line) for 95% CIs and **paired cluster bootstrap** for between-variant
deltas; **Holm–Bonferroni** correction across the planned contrasts; effects called
"significant *and* meaningful" only when Holm-p<0.05 AND |Δ| ≥ 0.01 (pre-registered minimum effect).
We report the number of test clusters per split as the true effective sample size.

## 3. Results

### 3.1 Real Tahoe-100M — GRN-quality spectrum (PRIMARY RESULT)
22.66M cells, 8,875 conditions; unseen_drug test = 1,708 conditions across **39 drug clusters**.
DEG-Pearson@50 with **cluster-bootstrap** 95% CIs (resampling drugs). 3 seeds per variant.

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline B1 mean-effect | 0.601 | [0.537, 0.655] |
| baseline B2/B3 ridge | 0.582 | [0.507, 0.646] |
| MoE / GRN = none | 0.630 | [0.551, 0.697] |
| MoE / GRN = random (matched TRRUST density) | 0.623 | [0.538, 0.694] |
| MoE / GRN = TRRUST | 0.627 | [0.545, 0.697] |
| MoE / GRN = TRRUST weighted | 0.629 | [0.546, 0.698] |
| MoE / GRN = co-expr (train-only) | 0.626 | [0.542, 0.695] |
| MoE / GRN = co-response/LFC (train-only) | 0.625 | [0.544, 0.695] |
| MoE / GRN = **CollecTRI** (4× denser, signed, curated) | 0.626 | [0.544, 0.696] |
| MoE / GRN = CollecTRI weighted (signed bias) | 0.623 | [0.540, 0.694] |
| MoE / GRN = random @ CollecTRI density | 0.626 | [0.544, 0.697] |

Planned contrasts (paired cluster bootstrap, Holm-corrected): **every** GRN variant vs none is within
±0.007 with Holm-p > 0.1 — none significant, none meaningful (|Δ|≥0.01). Notably the data-derived
co-response GRN — the variant a leaky analysis would have favoured — is Δ = **−0.005 (p=0.52)** vs none
and −0.003 (p=0.83) vs TRRUST. **Dense, curated, signed CollecTRI closes the "TRRUST was too sparse"
objection**: at 2,193 in-space edges (4× TRRUST's 539) it is Δ = −0.004 (Holm-p=1.0) vs none and —
decisively — Δ = **−0.0001 (Holm-p=1.0)** vs a *random* graph at the same density, i.e. statistically
and numerically indistinguishable from random rewiring. Edge sign does not help either (CollecTRI
weighted vs CollecTRI Δ = −0.003, ns). Neither GRN density, curation, nor sign moves OOD accuracy. On unseen_drug the deep MoE (~0.63) shows a small numerical edge over
the linear baselines (0.58–0.60) but with heavily overlapping cluster CIs (MoE 0.629 [0.551,0.697] vs
B1 0.601 [0.537,0.655]) — not a significant win; and the **GRN attention mask contributes nothing
measurable, at any quality level**.
(GRN-propagation baseline: N/A here — Tahoe lacks per-row drug→target labels to seed it.)
This null is visualized per regime in **`fig_grn_study_tahoe_full_v2.png`**, and summarized across all
three regimes (real Tahoe, synthetic shared- and novel-target) in the headline
**`fig_money_grn_effect.png`**.

### 3.1b The deep model does not beat linear baselines on the harder OOD splits
Re-running the two harder OOD splits under the same leakage-free protocol (variant=none, 3 seeds,
deterministic fp32 eval, cluster bootstrap; `results/REAL_TAHOE_OOD_AUDIT.md`) overturns an earlier
(V1, HVG-leakage-tainted) comparison that had reported large PathwayMoE wins:

| split | clusters | B1 mean-effect | B2 ridge | PathwayMoE | MoE vs best baseline |
|---|---|---|---|---|---|
| unseen_cell_line | 10 | 0.850 | **0.868** | 0.824 | **−0.044 vs ridge, Holm-p=0.001 (sig & meaningful)** |
| unseen_both | 33 | **0.607** | 0.390 | 0.581 | −0.026 vs mean-effect, p=0.47 (ns) |

On unseen_cell_line the 45M-param model **loses significantly to ridge**; on unseen_both it ties (and
sits slightly below) the trivial mean-effect baseline. The earlier "wins" were an artifact of
corrupted-low V1 baselines (0.43/0.32 → 0.85/0.61 once HVG-selection leakage is removed). Thus, on
real Tahoe OOD, **neither the GRN prior nor the deep architecture itself confers a robust advantage
over linear baselines** — strengthening, not weakening, the paper's thesis. See
**`fig_comparison_tahoe.png`** (baselines vs PathwayMoE across the three OOD splits, cluster CIs).

### 3.2 Synthetic, shared-target regime — GRN is redundant
Synthetic data whose generative process *is* GRN propagation, where drug classes share target hubs
(training drugs hitting a hub teach the model that hub's response). unseen_drug test = 270 conditions,
9 drug clusters. DEG-Pearson@50, cluster CIs, 3 seeds:

| GRN variant | DEG-Pearson@50 | Δ vs none (Holm-p) |
|---|---|---|
| none | 0.596 | — |
| random | 0.595 | −0.001 (1.0) |
| co-expr (train-only) | 0.593 | −0.003 (1.0) |
| co-response/LFC (train-only) | 0.594 | −0.002 (1.0) |
| **ground_truth (true GRN)** | **0.610** | **+0.014 (raw p=0.07, Holm-p=0.35)** |

Even the **true** generative GRN gives only +0.014 (not significant). When test perturbations hit
targets already seen in training, a flexible model learns the response directly — the GRN mask is
**redundant**. (B3 ridge 0.631 is the strongest model here; the MoE neither needs nor benefits from
the GRN.)

### 3.3 Synthetic, novel-target regime — the boundary condition + mechanism test
Each drug hits a **distinct** target TF, so holding out a drug holds out its target's direct effect:
predicting it *requires* propagating along the GRN from a novel target. Well-powered (192 drugs, 39
clusters — same power as real data); deterministic fp32 eval; cluster CIs, 3 seeds. We also compare
GRN-injection *mechanisms*: hard attention mask vs soft GNN message-passing vs both.

| Model (all use the true GRN) | DEG-Pearson@50 | Δ vs none (Holm-p) |
|---|---|---|
| none | 0.317 [0.181, 0.445] | — |
| GRN as attention mask | 0.396 [0.270, 0.526] | **+0.079 (0.57)** |
| GRN as soft propagation | 0.370 [0.247, 0.497] | +0.053 (0.57) |
| GRN as mask + propagation | 0.391 [0.266, 0.523] | +0.075 (0.57) |

> CPU-deterministic eval (`--eval-device cpu`). The mechanism table is sensitive to CUDA
> atomic-add ordering in MoE `index_add_`; GPU re-evaluations vary by ~0.05 DEG50 across
> processes and bias the magnitude systematically low. See AUDIT.md item N.

Here the true GRN yields a **meaningful point-estimate gain (+0.05–0.08, all above the |Δ|≥0.01
threshold)** — and unlike the shared-target regime, the effect is positive and consistent — but it
is **not statistically significant** (Holm-p=0.57): held-out novel targets are intrinsically hard,
so between-drug variance is large and the 39-cluster CIs are wide ([0.18, 0.53]). The **injection
mechanism does not matter** (mask ≈ propagation ≈ both; mask−vs−prop Δ=−0.026), arguing against "we
just used the wrong mechanism." So the GRN *plausibly* helps exactly where it should (novel targets)
but only in a regime that is barely predictable; we cannot confirm it at p<0.05.

**Bottom line across regimes:** where OOD prediction is *feasible* — real Tahoe and synthetic
shared-target — a GRN prior confers **no** benefit (tight nulls; true GRN +0.014, ns). The GRN
*plausibly* helps for **novel targets**, a near-unpredictable regime where the +0.05–0.08 gain
is meaningful but not significant. Independent of injection mechanism.

## 4. Discussion

Across a real 22.6M-cell benchmark and two synthetic regimes with known ground-truth GRNs, a
GRN-structured attention mask provides **no robust, specific benefit** for OOD perturbation prediction:
- On **real Tahoe-100M**, no GRN variant — generic (TRRUST), weighted, or train-only data-derived —
  beats no-GRN (all |Δ|≤0.007, Holm-p>0.1); and under the same protocol the deep MoE does not beat
  the linear baselines either (loses to ridge on unseen_cell_line, ties mean-effect on unseen_both).
- On **synthetic shared-target** data, even the *true* GRN is redundant: with targets seen in training,
  the model learns the perturbation response directly.
- On **synthetic novel-target** data, where the GRN is in principle essential, the true GRN gives a
  meaningful but non-significant point-estimate gain (+0.05–0.08, Holm-p=0.57) — and the injection
  mechanism (mask vs soft message-passing) does not matter — but this regime is near-unpredictable.

A likely mechanism, beyond the redundancy/insufficiency dichotomy, is **mask sparsity**: with
500–4,500 edges over 2,000 gene tokens, GRN-masked attention heads see ~1–2 neighbours per gene and are
effectively starved, so the dense heads carry the model and the masked heads add little. The unifying
explanation is about **what generalization the GRN enables**: a hard mask could only help when a test
perturbation must propagate from a *novel* node to known genes — a regime where prediction is anyway
near-impossible. Whenever test perturbations share targets/pathways with training (the realistic case,
incl. Tahoe's drugs over shared pathways), a flexible model learns the response from data and the
structural prior is redundant. The leakage-free design matters: a
test-aware co-response GRN (the variant most likely to look good) is exactly Δ=−0.005 — naïvely
including test correlations would have manufactured a spurious positive.

This refines, rather than contradicts, the motivation: the issue is not that biology-structured priors
are wrong in principle, but that injecting a *static, binary, undirected* GRN as an attention mask is
not how to realize the benefit — the information it encodes is either redundant (shared-target) or
insufficient (novel-target) for this task at this scale.

## 5. Limitations
2,000 HVGs (not full transcriptome); one model family (PathwayMoE) — though we tested two GRN
*injection mechanisms* (hard mask and soft message-passing), both null; co-response GRN is
correlational, not causal; the synthetic generative process simplifies real regulation; the
GRN-propagation baseline needs drug→target labels (absent on the Tahoe subset). The novel-target
positive control, while suggestive (+0.02–0.03), is underpowered by the intrinsic difficulty of that
regime, so we cannot confirm a GRN benefit at p<0.05 even where one is plausible.

## 6. Reproducibility & availability
Code: the typed `pmoe/` package (`experiments/{study,mechanism,train}.py`, `priors/grn.py`,
`eval/{metrics,stats,loading,report}.py`) + `reproduce.ps1`; 44 tests incl. a leakage regression test.
Split seed fixed (identical test set across variants/seeds); 3 model seeds; deterministic fp32 eval;
cluster-bootstrap 95% CIs + Holm correction. Each checkpoint stores an env+data provenance manifest
(library versions, gene-list hash, shard count). Results: `results/study_*_v2.json`,
`mechanism_*.json`, the OOD-baseline audit `results/REAL_TAHOE_OOD_AUDIT.md`; figures
`results/fig_grn_study_*_v2.png`, `fig_money_grn_effect.png`, `fig_comparison_tahoe.png`. Real data =
803/3,388 Tahoe-100M shards (22.66M cells) on `E:\vc_project_data`; rebuild via
`python -m pmoe.data.preprocess_tahoe`. The OOD-split re-run used batch 48 (batch 96 exhausts 24 GB VRAM).
