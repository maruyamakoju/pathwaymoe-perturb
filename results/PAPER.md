# When does gene-regulatory structure help perturbation prediction? A controlled study from synthetic ground truth to 22M real cells

*Draft — results filled from `study_tahoe_full.json` after the grid completes.*

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
(all |Δ DEG-Pearson|≤0.007, Holm-p>0.1), although the deep model beats linear baselines. On synthetic
data we localize why: when test perturbations **share targets** with training, even the *true* GRN is
redundant (the model learns the response directly); when targets are **novel**, the task is
near-unpredictable and even the **ground-truth GRN at full statistical power** (39 clusters) gives no
significant benefit (Δ=−0.003, p=0.86). We conclude that injecting a static, binary GRN as an
attention mask is not an effective inductive bias for this task — the structure it encodes is either
redundant (shared targets) or insufficient (novel targets) for OOD perturbation prediction. We release
a typed, tested, leakage-audited benchmarking package (`pmoe/`).

## 1. Introduction

[Gap: foundation models tie ridge on OOD perturbation; community proposes biology-structured priors;
but it is unclear whether the *prior* helps or the *architecture*, and whether a *generic* GRN is
even the right structure. We isolate GRN quality as a controlled variable.]

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
22.66M cells, 8,875 conditions; unseen_drug test = 1,707 conditions across **39 drug clusters**.
DEG-Pearson@50 with **cluster-bootstrap** 95% CIs (resampling drugs). 3 seeds per variant.

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline B1 mean-effect | 0.608 | [0.546, 0.664] |
| baseline B2/B3 ridge | 0.582 | [0.507, 0.646] |
| MoE / GRN = none | 0.630 | [0.552, 0.701] |
| MoE / GRN = random | 0.623 | [0.539, 0.696] |
| MoE / GRN = TRRUST | 0.627 | [0.545, 0.699] |
| MoE / GRN = TRRUST weighted | 0.632 | [0.555, 0.701] |
| MoE / GRN = co-expr (train-only) | 0.626 | [0.543, 0.698] |
| MoE / GRN = co-response/LFC (train-only) | 0.625 | [0.544, 0.698] |

Planned contrasts (paired cluster bootstrap, Holm-corrected): **every** GRN variant vs none is within
±0.007 with Holm-p > 0.1 — none significant, none meaningful (|Δ|≥0.01). Notably the data-derived
co-response GRN — the variant a leaky analysis would have favoured — is Δ = **−0.005 (p=0.52)** vs none
and −0.003 (p=0.74) vs TRRUST. The deep MoE (~0.63) beats the linear baselines (0.58–0.61), but the
**GRN attention mask contributes nothing measurable, at any quality level**.
(GRN-propagation baseline: N/A here — Tahoe lacks per-row drug→target labels to seed it.)

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
| **ground_truth (true GRN)** | **0.610** | **+0.014 (0.34)** |

Even the **true** generative GRN gives only +0.014 (not significant). When test perturbations hit
targets already seen in training, a flexible model learns the response directly — the GRN mask is
**redundant**. (B3 ridge 0.631 is the strongest model here; the MoE neither needs nor benefits from
the GRN.)

### 3.3 Synthetic, novel-target regime — the boundary condition (positive control)
Each drug hits a **distinct** target TF, so holding out a drug holds out its target's direct effect:
predicting it *requires* propagating along the GRN from a novel target. This is where the GRN should
matter most, and the positive control that the measurement is *sensitive*.

- **Low-power (9 clusters):** ground_truth vs none Δ=+0.019 (**p<0.001, significant**) — the method
  *does* detect a GRN benefit when one exists — but random vs none Δ=+0.022 (p=0.10), i.e. of similar
  magnitude, and absolute DEG-Pearson is near zero for all models (novel-target extrapolation is
  near-impossible). The benefit is weak and **non-specific** (any sparse mask ≈ true GRN).
- **Well-powered (192 drugs, 39 clusters — same power as real data):** all models near zero
  (none 0.024, random 0.019, ground_truth 0.021); **ground_truth vs none Δ=−0.003 (Holm-p=0.86)**,
  random vs none Δ=−0.005 (p=0.86) — **no significant benefit**. The +0.019 "significance" at 9
  clusters was a low-power artifact that vanishes under proper powering. Held-out *unique* targets are
  essentially unpredictable (DEG-Pearson ≈ 0) and the true GRN does not change that.

**Bottom line across all regimes:** the GRN attention mask never confers a significant, specific
benefit — not on real data, not when the GRN is redundant (shared-target), and not when it should be
essential (novel-target), even with the ground-truth GRN at full statistical power.

## 4. Discussion

Across a real 22.6M-cell benchmark and two synthetic regimes with known ground-truth GRNs, a
GRN-structured attention mask provides **no robust, specific benefit** for OOD perturbation prediction:
- On **real Tahoe-100M**, no GRN variant — generic (TRRUST), weighted, or train-only data-derived —
  beats no-GRN (all |Δ|≤0.007, Holm-p>0.1), even though the deep MoE beats linear baselines.
- On **synthetic shared-target** data, even the *true* GRN is redundant: with targets seen in training,
  the model learns the perturbation response directly.
- On **synthetic novel-target** data, where the GRN is in principle essential, the task becomes
  near-unpredictable and even the ground-truth GRN at full power gives no significant benefit
  (Δ=−0.003, p=0.86); held-out unique targets are essentially unpredictable.

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
2,000 HVGs (not full transcriptome); GRN as a (mostly binary) attention mask — weighted GRN tested but
soft-/message-passing integration not exhausted; one model family (PathwayMoE); co-response GRN is
correlational; synthetic generative process is a simplification of real regulation; GRN-propagation
baseline is degenerate without drug→target labels (real data) and was not the focus.

## 6. Reproducibility
`code/preprocess_tahoe_stream.py`, `build_grn_variants.py`, `run_study.py`, `stats.py`. Seeds fixed;
splits fixed; 95% CIs + paired bootstrap. Data on E:, 803 shards listed in meta.json.
