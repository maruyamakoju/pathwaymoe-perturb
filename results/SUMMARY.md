# Does a gene-regulatory-network prior help OOD perturbation prediction? A leakage-audited negative result

*Workshop-note summary. Full paper: `PAPER.md`. Code: github.com/maruyamakoju/pathwaymoe-perturb*

## Motivation
Foundation models do not beat linear baselines on out-of-distribution (OOD) single-cell perturbation
prediction (Nature Methods, 2025). A popular remedy is to add a **biology-structured inductive bias** —
e.g. constrain attention to gene-regulatory-network (GRN) edges, or propagate along them. We ask the
sharper question: **does a GRN prior actually help, and if so, in which regime and via which
mechanism?** We hold a fixed model (a pathway Mixture-of-Experts with gene-token attention) and vary
**only** the GRN.

## Method (what makes it trustworthy)
- **GRN-quality spectrum**, matched edge density: none / random / curated (TRRUST, binary &
  signed-weighted) / **dense curated (CollecTRI, 4× denser & signed)** / data-derived co-expression &
  co-response / synthetic ground-truth — plus **drug→target mechanistic grounding** (which gene each drug hits).
- **Two injection mechanisms**: hard attention mask vs soft GNN message-passing.
- **No leakage**: data-derived GRNs built from *training conditions only*, per split (regression-tested).
- **Cluster bootstrap** over drugs/cell-lines (the real independent units) + Holm correction +
  pre-registered minimum meaningful effect (0.01); deterministic CPU eval; headline re-verified by
  direct recompute. 67 tests. Data: 22.66M-cell Tahoe-100M subset (8,875 conditions × 2,000 HVGs).

## Result
![GRN effect across regimes](fig_money_grn_effect.png)

| Regime | no-GRN | + GRN | Δ (Holm-p) |
|---|---|---|---|
| Real Tahoe-100M (shared targets) | 0.629 | 0.619–0.628 (any variant, incl. CollecTRI) | within ±0.01, **ns** |
| Real Tahoe, **CollecTRI vs random @ same density** | 0.624 | 0.624 | +0.001, **ns** |
| Real Tahoe, **+ drug→target grounding** | 0.629 | 0.632 | +0.003, **ns** (p=0.93) |
| Synthetic, shared targets, **true GRN** | 0.596 | 0.610 | +0.014, **ns** |
| Synthetic, novel targets, **true GRN** | 0.317 | 0.396 | +0.079, **ns** |

Where OOD prediction is **feasible** (real data, shared targets), a GRN prior — generic, weighted,
**dense curated (CollecTRI)**, data-derived, or even the **ground-truth** GRN — confers **no significant
benefit**: a flexible model learns the perturbation response directly, so the structural prior is
**redundant**. CollecTRI (4× denser than TRRUST, signed) is indistinguishable from a *random* graph at
the same density (Δ=+0.001), closing the "the curated GRN was just too sparse" objection. A different
kind of prior — **drug→target mechanistic grounding** — is also null on OOD (Δ=+0.003, p=0.93), even
though a synthetic positive control confirms the target pathway is functional (target ablation Δ=+0.25).
The true GRN *plausibly* helps for **novel targets** (+0.05–0.08, meaningful but ns), a near-unpredictable
regime where the effect is not significant at the 39-cluster sample size. The **injection mechanism does not matter** (mask ≈ message-passing).

Under the same leakage-free protocol, the deep PathwayMoE **does not beat the linear baselines** on
real OOD either — it loses to ridge on unseen_cell_line (0.824 vs 0.868, Holm-p=0.001) and ties the
mean-effect baseline on unseen_both (a prior V1 comparison reporting large wins was an HVG-leakage
artifact; see `REAL_TAHOE_OOD_AUDIT.md`). So neither the GRN prior nor the architecture helps here.

## Why it matters / caveats
The contribution is (1) evidence that GRN-as-prior is redundant in the realistic regime, isolating
*when* it could help (novel targets) from *whether* it does (not significantly); (2) a leakage-audited,
cluster-bootstrapped benchmarking harness; (3) a cautionary tale — we caught **three result-invalidating
bugs** (test-set GRN leakage, a silent train-without-GRN fallback, bf16 metric nondeterminism), each of
which alone would have produced a confidently wrong claim. Scope: one model family, 2,000 HVGs, GRN as
mask/propagation; we do not claim GRNs are useless for all architectures or tasks — only that *this*
common way of injecting them does not improve OOD perturbation prediction where prediction is feasible.
