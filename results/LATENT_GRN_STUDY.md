# Latent / Dynamic-GRN (LRI) Study — Honest Negative Result

**Date:** 2026-05-27 · **Author:** Claude (Opus 4.7), auditing a prior Gemini-CLI session
**Regime:** synthetic_hard_v2, `unseen_drug` (novel-target OOD), DEG50-Pearson on val.

## Question
A prior session added a *Variational Latent-GRN* ("dynamic regulatory inference", LRI)
that predicts a condition-specific regulatory gate `Gate_ij = σ(tf_act_i + tgt_rec_j)`
from the perturbation embedding, and claimed (a) a large performance gain over the static
GRN and (b) interpretable drug-specific mechanism (MoA). **Does the dynamic gate actually
help, and does it encode condition-specific regulatory structure?**

## Headline answer: No, on both counts.
The latent GRN does not beat the static GRN at any matched comparison, and the learned
gate is a degenerate no-op (a flat 0.5 matrix with no structure and no drug-specificity).
This is consistent with — and strengthens — the earlier V2 finding that structured GRN
priors give no significant benefit where prediction is feasible.

## Evidence

### 1. The claimed "breakthrough" was an unequal-epochs artifact
The prior snapshot reported "Latent 0.701 vs Static 0.44 (+0.26)". In fact the static
baseline run had been **interrupted at epoch 4** (`runs/...long_static.jsonl` had only
epochs 0/2/4) while latent ran the full 50. Matched comparisons:

| epochs | Static GRN | Latent GRN | Δ (static−latent) |
|--------|-----------|-----------|-------------------|
| 15     | 0.443     | 0.438     | +0.005            |
| 50     | **0.7738**| 0.7009    | **+0.073**        |

At equal epochs **static wins**, and the gap *grows* with training. The collapsed gate
slightly hurts relative to a clean static mask. *(Multi-seed error bars: see §4.)*

### 2. The gate is degenerate (VAE posterior collapse)
`code/analyze_gate_degeneracy.py` on the trained latent checkpoint(s):

| model (50/20ep) | gate mean | within-drug spatial std | across-drug std | sampling-noise std |
|---|---|---|---|---|
| long_latent (static+latent)  | 0.500 | 0.00098 | 0.00077 | 0.240 |
| steel_gateonly (gate-only)   | 0.500 | 0.00060 | 0.00072 | 0.240 |
| steel_gateonly_lowkl (kl=1e-5)| 0.500 | 0.00090 | 0.00094 | 0.240 |

- **Flat 0.5, no structure:** the deterministic (mean) gate is ≈0.5 on every edge
  (spatial std ~0.001). It applies a uniform `log(0.5)` bias — i.e. it does nothing.
- **Not condition-specific:** across-drug std ≈ 0.0008; trametinib vs nutlin-3 gates
  differ by ~0.001. The LRI essentially ignores the perturbation embedding.
- **The "0.99 edges" were sampling noise:** the LRI sampled `eps` even at eval, giving a
  per-edge std of 0.24 on top of the 0.5 mean. Signal/noise ≈ 0.003. The reported
  high-confidence "regulatory edges" were random draws thresholded at >0.8.
- **Rank-1 confound:** `σ(a_i+b_j)` forces "top edges" to be the Cartesian product of the
  few highest-receptivity columns — structurally guaranteed, not learned topology.

Diagnosis: textbook **posterior collapse** (μ→0, logvar→0 = the prior, KL→0). The decoder
learned to be robust to the noisy gate rather than use it.

### 3. Steelman: even forced to use the gate, it adds nothing
To give the idea its best shot, we (i) fixed eval to use μ (deterministic gates), (ii)
trained with the gate as the **only** regulatory pathway (`use_grn_mask=False`), and (iii)
weakened KL to 1e-5 to fight collapse. 20 epochs, novel-target:

| config | DEG50 |
|---|---|
| static GRN | **0.491** |
| no-GRN floor | 0.474 |
| gate-only, kl=1e-5 | 0.472 |
| gate-only, kl=0.001 | 0.454 |

**Gate-only ≤ the no-GRN floor**, and the gate stayed degenerate (above). When the model
is forced to rely on the dynamic gate, it does no better than having no regulatory prior
at all. The idea does not work here.

### 4. Multi-seed confirmation
`code/run_multiseed_confirm.py` — 3 seeds × {static, latent} × 50ep (fixed data split,
varied init/optimization), `results/multiseed_confirm.json`:

| seed | static | latent | gap (static−latent) |
|------|--------|--------|---------------------|
| 0    | 0.7650 | 0.7506 | +0.0144 |
| 1    | 0.7410 | 0.7125 | +0.0285 |
| 2    | 0.7976 | 0.7664 | +0.0312 |
| **mean** | **0.768 ± 0.023** | **0.743 ± 0.023** | **+0.025** |

**Static beats latent in all three seeds.** Paired gap +0.025 (sample SD 0.009; paired
t(2)≈4.7, p≈0.04 — significant, though n=3 is small so treat the p-value as indicative).
The single-run +0.073 gap was on the high side; the honest, seed-averaged gap is a smaller
but consistent **+0.025 in favor of static**. The dynamic gate never helps.

## "MoA discovery" — why it was void
The extracted trametinib/nutlin-3 "regulatory gates" came from the **synthetic** dataset
(gene IDs `G00000–G01999`, not real symbols), so cross-validation against MAPK/p53 was
never possible; the drug names are synthetic labels. The real-Tahoe latent model was never
checkpointed. Given the gate is non-specific noise (§2), there is no valid mechanism to map
to pathways. We did **not** pursue real-Tahoe training or enrichment testing — both would
map noise to biology.

## What was changed / added
- `pmoe/models/layers.py`: LRI uses μ at eval (deterministic, reproducible gates).
- `pmoe/experiments/train.py`: `latent_grn_kl_weight` override (for the KL sweep).
- `code/analyze_gate_degeneracy.py`, `run_long_static.py`, `run_steelman.py`,
  `run_multiseed_confirm.py`.
- Fixed 4 stale `tests/test_grn_propagation.py` (flat→nested ModelConfig); 44 pytest pass.
- `LATEST_STATUS.md` rewritten; full audit in `WORK_LOG.md`.

## Bottom line
The dynamic/latent-GRN does not improve OOD perturbation prediction and produces no
interpretable, condition-specific regulatory structure. The honest result is a clean
**negative**, reinforcing the project's V2 conclusion.
