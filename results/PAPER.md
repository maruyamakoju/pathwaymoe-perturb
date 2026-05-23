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
- **trrust** — TRRUST v2 curated human TF→target (generic prior; symbol→Ensembl mapped).
- **coexpr** — top gene–gene Pearson correlations of pseudobulk expression (data-derived, observational).
- **coexpr_lfc** — top gene–gene correlations of LFC across conditions (data-derived, perturbation co-response).

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
Leakage-controlled splits (seed fixed so the test set is identical across all runs): unseen_drug
(Butina/Tanimoto<0.8 dedup), unseen_cell_line, unseen_both. Primary metric: **DEG-Pearson@50** =
Pearson on the 50 genes with largest |true LFC| per condition (dataset-agnostic). Linear baselines:
B1 mean-effect, B2 ridge(ChemBERTa+cell line+baseline), B3 ridge+biology. Each model config trained
with **3 seeds**; we report mean and 95% bootstrap CI over test conditions, and **paired bootstrap**
significance for between-variant deltas (resampling conditions).

## 3. Results

### 3.1 Positive control (synthetic)
[From earlier run: full model beats baselines on OOD; GRN-mask ablation −0.080 — the ground-truth GRN
is the dominant useful component. To re-run as the `ground_truth` end of the spectrum.]

### 3.2 Real Tahoe-100M — GRN-quality spectrum
[Table + figure from `study_tahoe_full.json`: DEG-Pearson@50 per variant with CIs, paired p-values
trrust_vs_none, coexpr_lfc_vs_none, coexpr_lfc_vs_trrust, coexpr_lfc_vs_random.]

| GRN variant | DEG-Pearson@50 (95% CI) | Δ vs none | p |
|---|---|---|---|
| none | … | — | — |
| random | … | … | … |
| trrust (generic) | … | … | … |
| coexpr (observational) | … | … | … |
| coexpr_lfc (co-response) | … | … | … |

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
