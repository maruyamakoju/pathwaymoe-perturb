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
We run this on (i) synthetic data whose generative process *is* GRN propagation (positive control)
and (ii) a 22.6M-cell, 8,875-condition subset of Tahoe-100M with deep pseudobulk. **[Headline result
TBD from grid.]**

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

### 3.1 Positive control (synthetic)
[From earlier run: full model beats baselines on OOD; GRN-mask ablation −0.080 — the ground-truth GRN
is the dominant useful component. To re-run as the `ground_truth` end of the spectrum.]

### 3.2 Real Tahoe-100M — GRN-quality spectrum (PRIMARY RESULT)
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

## 4. Discussion

[Interpretation depends on the pattern:
 - If only coexpr_lfc > none (and > trrust/random): "GRN helps, but must be context/data-relevant; a
   generic curated GRN over HVGs does not transfer." Strong, novel, honest.
 - If no variant > none on real data: "GRN structure as a hard attention mask does not improve
   real-data OOD prediction at this scale/representation, despite helping on synthetic ground truth —
   the inductive bias is necessary in principle but not realizable with available GRNs." Also a result.
 - Contrast synthetic (GRN known → large gain) vs real (no/partial gain) to localize the bottleneck:
   it is GRN *quality*, not the architecture.]

## 5. Limitations
2,000 HVGs (not full transcriptome); GRN as a binary mask (not weighted/signed); no drug→target
features on real data; single architecture; co-response GRN is correlational, not causal.

## 6. Reproducibility
`code/preprocess_tahoe_stream.py`, `build_grn_variants.py`, `run_study.py`, `stats.py`. Seeds fixed;
splits fixed; 95% CIs + paired bootstrap. Data on E:, 803 shards listed in meta.json.
