# Overnight Autonomous Work Log
Started 2026-05-26, autonomous 8h session. Author: Claude (Opus 4.7).
Read top-to-bottom; newest sections appended at the end.

## TL;DR for when you wake up
1. The Gemini "breakthrough" (LATEST_STATUS.md: Latent 0.701 vs Static 0.44) was an
   **invalid comparison** — the static run was killed at epoch 4 while latent ran 50
   epochs. The only matched (15ep) comparison shows latent is *slightly worse*
   (0.438 vs 0.443). A clean matched 50ep static rerun is running to confirm.
2. The "MoA discovery" (trametinib/nutlin-3 gates) is **scientifically void**:
   - gates were extracted from the SYNTHETIC dataset (genes G00000-G01999, fake), not
     real biology, so MAPK/p53 cross-validation was never possible;
   - the real-Tahoe latent checkpoint never existed (run interrupted at 2 epochs, no save);
   - the LRI gate is **degenerate** (see Step 0): mean gate = 0.5 flat, the "0.999 edges"
     were pure eval-time sampling noise.
3. Decision: gave the dynamic-GRN idea a genuine best-effort fix (steelman) before
   declaring it dead. Results below.

---

## Step 0 — LRI gate degeneracy (DECISION POINT) — DONE
Script: `code/analyze_gate_degeneracy.py` -> `results/interpretability/gate_degeneracy_report.json`
Checkpoint analyzed: `synthetic_hard_v2__...__long_latent.pt` (the 50ep model, val 0.701).

Findings (all damning):
- **Saturation**: mean gate 0.5000, std 0.0016, pctiles 1-99% = [0.496, 0.504].
  The deterministic (mean) gate is a flat 0.5 — structurally a no-op.
- **Specificity**: across-drug std per edge = 0.0008; trametinib vs nutlin-3 gate
  differ by 0.001. The LRI essentially ignores the perturbation embedding.
- **Noise vs signal**: eval samples `eps` even in eval() (bug). Sampling std = 0.24,
  i.e. signal/noise = 0.003. The saved >0.8 "edges" are ~100% sampling noise.
- **Rank-1 confound**: gate = sigmoid(a_i + b_j) is a rank-1 outer sum. Top-100 edges
  had 2 unique targets, all into top-10 receptivity genes — a Cartesian-product
  artifact, not learned topology.

Diagnosis: **VAE posterior collapse** + an eval-time sampling bug. The gate adds nothing,
which is exactly why latent-GRN performance == static-GRN performance.

---

## Session progress (appended live)
- [DONE] Fixed eval-time sampling bug: `pmoe/models/layers.py` LRI now uses mu at eval.
- [DONE] Threaded `latent_grn_kl_weight` override through train()/make_hierarchical_config.
- [DONE] Rewrote LATEST_STATUS.md to correct the invalid "breakthrough" + void "MoA".
- [DONE] Fixed 4 stale tests in tests/test_grn_propagation.py (flat->nested ModelConfig,
  model(batch)->model(**batch)). Full suite now 44 passed.
- [RUNNING] `code/run_long_static.py` -> matched static 50ep (runs/...long_static.jsonl).
- [RUNNING] `code/run_steelman.py` -> E1 nogrn / E2 gateonly / E3 gateonly_lowkl / E4 static,
  20ep each, then degeneracy re-analysis of the gate-bearing models (runs/steelman.log,
  results/steelman_results.json).
- [DONE] Updated memory (pathwaymoe-perturb.md) with the LRI episode.

### Steelman hypothesis & how to read it
If gate-only (E2/E3) reaches DEG50 >> nogrn floor (E1) AND the re-analysis shows
across-drug gate std rising well above ~0.01 with signal/noise > ~1.5, the dynamic-GRN
idea is revived and MoA becomes worth pursuing on a real-Tahoe model. If gate-only ≈ nogrn
and gates stay collapsed, the negative result stands: dynamic GRN adds nothing here.

## RESULT 1: matched static vs latent (50 epochs, novel-target) — the "breakthrough" was BACKWARDS
| model | DEG50 @50ep (val) |
|---|---|
| Static GRN | **0.7738** |
| Latent GRN | 0.7009 |
Static WINS by +0.073 at matched epochs. Combined with the 15ep result (static 0.443 >
latent 0.438), latent-GRN loses at every fair comparison. The Gemini "+0.26 breakthrough"
was purely 50-epochs-vs-4-epochs. The collapsed gate slightly HURTS vs clean static.

## RESULT 2: multi-seed confirmation (3 seeds x {static,latent} x 50ep) — DONE
static 0.768 +/- 0.023 | latent 0.743 +/- 0.023 | paired gap +0.025 (all 3 seeds positive:
+0.0144, +0.0285, +0.0312; paired t(2)~4.7, p~0.04). Static wins every seed. Full numbers
in results/multiseed_confirm.json; written up in results/LATENT_GRN_STUDY.md.

## FINAL VERDICT
The dynamic/latent-GRN (LRI) is a clean NEGATIVE on this task, triangulated 3 ways:
  1. Loses head-to-head at every matched comparison (15ep, 50ep single-run, AND 3-seed).
  2. Gate is a degenerate no-op (flat 0.5, no structure, not drug-specific) — posterior collapse.
  3. Steelman (gate-only, eval fixed, low KL) <= no-GRN floor; gate stays degenerate.
The "MoA discovery" was void (synthetic genes + sampling-noise edges). MoA on these gates
is not viable. Reinforces the V2 finding: structured GRN priors don't help where feasible.

## What I deliberately did NOT do (and why)
- Did NOT train the 22M-cell real-Tahoe latent model (Step 1): gates carry no signal, so it
  would map noise to biology and burn hours/VRAM for nothing.
- Did NOT run pathway-enrichment (Step 3): requires a meaningful gate; there isn't one.
- Did NOT commit anything: standing rule is commit only when asked. Everything is on disk.

## To commit when you're ready (suggested)
  git add code/ results/ pmoe/ tests/ LATEST_STATUS.md WORK_LOG.md
  git commit  # message e.g.: "Audit latent-GRN: invalid 'breakthrough' + degenerate gate; honest negative w/ multi-seed"
Note: pre-existing 4 failing tests were FIXED (now 44 pytest pass). New untracked files:
code/{analyze_gate_degeneracy,run_long_static,run_steelman,run_multiseed_confirm}.py,
results/{LATENT_GRN_STUDY.md,multiseed_confirm.json,steelman_results.json,
interpretability/gate_degeneracy_report*.json}.

## RESULT 3: real-Tahoe OOD "wins" do NOT survive the leakage-free protocol (2026-05-27)
comparison_tahoe.md claimed PathwayMoE beats baselines on hard OOD (cell_line 0.551>0.432,
both 0.512>0.324). Those are V1 (HVG-leakage bug B) and were never re-run clean. Re-ran
unseen_cell_line + unseen_both leakage-free (variant=none, 3 seeds, 25ep, batch48 — batch96
OOM-hung at 24GB):
  unseen_cell_line (10 clusters): ridge 0.868 / mean 0.850 / MoE 0.824 -> MoE LOSES to ridge
    by -0.044 (p<0.001, sig&meaningful).
  unseen_both (33 clusters): mean 0.607 / MoE 0.581 / ridge 0.390 -> MoE ties/slightly below
    mean-effect (-0.026, p=0.47); only beats the weak ridge.
The V1 baselines were corrupted-low (0.43/0.32 -> clean 0.85/0.61). The deep model does NOT
beat strong linear baselines on real OOD. Consistent with the paper's own thesis.
Writeup: results/REAL_TAHOE_OOD_AUDIT.md. comparison_tahoe.md flagged SUPERSEDED. Merged into
study_tahoe_full_v2.json (3 splits). Added MoE-vs-baseline paired bootstrap to study.py.

## OVERALL HONEST PICTURE (after this session)
On real Tahoe OOD perturbation prediction: (1) GRN prior doesn't help (V2, solid);
(2) dynamic/latent GRN doesn't help + gate is degenerate (this session); (3) the deep
PathwayMoE itself does NOT beat linear baselines (this session). The project's defensible
contribution is the rigorous, leakage-audited NEGATIVE: structured-bias deep models tie or
lose to ridge/mean-effect here. The earlier positive headlines were leakage/epoch artifacts.
