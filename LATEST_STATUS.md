# Project Snapshot: Dynamic Regulatory Inference (LRI) — CORRECTED
Date: 2026-05-26 (rewritten after verification; supersedes the 2026-05-24 "breakthrough" snapshot)

> The previous version of this file (authored by Gemini CLI) claimed a "Performance
> Breakthrough" (Latent GRN 0.701 vs Static 0.44) and an "MoA Discovery". **Both claims
> did not survive verification.** This file records what is actually true. See
> `WORK_LOG.md` for the full audit trail and `code/archive/analyze_gate_degeneracy.py` for the
> reproducible evidence.

## 1. What the prior snapshot claimed vs. reality

| Claim (prior snapshot) | Reality (verified) |
|---|---|
| Latent GRN 0.701 vs Static 0.44 (+0.26 "breakthrough") | **Invalid comparison.** The static run was interrupted at epoch 4 (`long_static.jsonl` has only epochs 0/2/4); latent ran the full 50. The only *matched* (15-epoch) comparison, `results/latent_grn_comparison.json`, shows latent **0.438 vs static 0.443 — latent is slightly worse** (Δ=-0.0046). |
| "Reached 0.701 on Tahoe-22M in 2 epochs" | Suspicious: identical number to synthetic; run was interrupted, **no checkpoint saved**. Cannot be reproduced. |
| "MoA Discovery: trametinib/nutlin-3 regulatory gates" | **Void.** Gates were extracted from the SYNTHETIC dataset (genes `G00000–G01999`, not real symbols), so MAPK/p53 cross-validation was never possible. The drug names are synthetic labels. |
| "High-confidence gates (>0.99) = drug-specific logic" | **Sampling noise.** The LRI samples `eps` even at eval; the deterministic (mean) gate is a flat **0.5 everywhere**. Signal/noise across drugs = **0.003**. The >0.8 edges were noise. |

## 2. Root cause: the LRI collapsed to a no-op
`gate = sigmoid(tf_act_i + tgt_rec_j)` is a rank-1 outer sum. The trained model drove
μ→0, logvar→0 (the prior) — textbook **VAE posterior collapse**. The gate therefore adds
nothing, which is exactly why latent-GRN performance equals static-GRN performance. This
is consistent with the earlier rigorous V2 study finding: **structured GRN priors give no
significant benefit where prediction is feasible.**

## 3. What is actually being done now (2026-05-26 overnight session)
1. **Matched static 50ep rerun** — to confirm latent ≈ static on a fair, equal-epoch basis.
2. **Steelman of the dynamic-GRN idea** — give it its best shot before declaring it dead:
   deterministic eval (fixed), and train with the gate as the *only* regulatory pathway
   (`use_grn_mask=False, latent_grn=True`) so the model is forced to use it; plus a KL
   sweep to fight posterior collapse. Re-measure gate specificity + DEG50.
3. Honest write-up of the (likely negative) result.

## 4. Code added this session
- `code/archive/analyze_gate_degeneracy.py` — reproducible gate-degeneracy audit (saturation,
  specificity, signal/noise, rank-1 confound).
- `code/archive/run_long_static.py` — the missing matched static run.
- `code/archive/run_steelman.py` — the steelman experiments.
- `pmoe/models/layers.py` — LRI now uses μ at eval (deterministic gates).

---
*Corrected by Claude (Opus 4.7), verifying the prior Gemini CLI snapshot.*
