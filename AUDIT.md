# Methodology Audit — GRN-quality study (deliberative critique before scaling up)

The v1 pipeline *works* but is not yet *correct enough to publish*. Below are the flaws found by
reasoning through the experimental design end-to-end, ordered by severity. Each has a concrete fix
that the v2 (`pmoe/`) package must implement. This document is the rationale for the refactor.

## A. CRITICAL — Data leakage in data-derived GRNs
`build_grn_variants.coexpr/coexpr_lfc` compute the gene–gene network from **all** conditions,
including the held-out **test** set. The model's attention structure therefore encodes test-set
correlations → optimistic bias precisely for the variants we want to champion. This invalidates any
claim that "co-response GRN helps."
**Fix:** data-derived GRNs must be built from **training conditions only**, *per split*. The GRN file
becomes split-specific (`grn_coexpr_lfc__unseen_drug.npz`). The runner builds it from train indices
before training that (variant, split) cell.

## B. CRITICAL — HVG selection leakage
HVGs are chosen by cross-condition LFC variance over **all** conditions (incl. test). Milder than A
(unsupervised, no labels) but still test-aware feature selection.
**Fix:** select HVGs on training conditions only (per split), or freeze a biology-defined panel and
report it. v2 supports a `hvg_source="train"` mode; we also report sensitivity to a fixed panel.

## C. MAJOR — Statistical units are not independent
Bootstrap over conditions assumes exchangeable conditions, but conditions cluster by drug and
cell-line (e.g., one drug × many cell lines). Naive condition bootstrap underestimates CIs.
**Fix:** **cluster bootstrap** resampling the grouping variable (drugs for unseen_drug; cell-lines for
unseen_cell_line). Report cluster-CIs as primary, condition-CIs as secondary.

## D. MAJOR — Multiple comparisons uncorrected
Five pairwise variant tests per split → inflated family-wise error.
**Fix:** Holm–Bonferroni correction across the planned contrasts; report corrected p.

## E. MAJOR — One architecture cannot separate two hypotheses
"GRN doesn't help" is confounded with "this architecture can't exploit GRN." We need a model whose
*entire* inductive bias is the GRN.
**Fix:** add a **GRN-propagation baseline** (GEARS-spirit): predict LFC by propagating a
perturbation signal along GRN edges (linear message passing), no deep net. If even this does not
benefit from a given GRN, the conclusion is about the GRN, not the architecture.

## F. MAJOR — Binary, undirected mask discards GRN information
TRRUST has sign (Activation/Repression); DoRothEA has confidence. We binarize + symmetrize, throwing
away exactly the structure that might help.
**Fix:** add a **weighted/signed** GRN variant (edge weight → additive attention-bias magnitude;
sign optional) so "GRN quality" includes edge quality, not just topology.

## G. MODERATE — Effect size vs statistical significance
n_test ≈ 1707 → trivial deltas can be "p<0.05" yet meaningless.
**Fix:** report effect sizes and a pre-registered **minimum meaningful effect** (Δ DEG-Pearson ≥ 0.01);
call results "significant *and* meaningful" only when both hold.

## H. MODERATE — Determinism / reproducibility
SDPA + atomic adds are nondeterministic; seeds alone don't guarantee bit-reproducibility.
**Fix:** set `torch.use_deterministic_algorithms(True)` where feasible, cudnn deterministic, log
library versions + data manifest (803 shard ids, hashes) into each result file.

## I. MINOR — Code-level correctness/robustness
- `model_predict` duplicated in eval.py, eval_ablations.py, run_study.py (drift risk; one had the
  GRN-reload bug). → single canonical loader.
- Checkpoint-name logic duplicated in train.py and run_study.py (already a near-miss). → one source.
- No typed configs; magic strings for paths/variants. → dataclass configs + enums.
- Tests = one smoke script. → pytest unit + integration suite, including a **leakage regression test**
  (assert no test-condition info enters any GRN/HVG).

## J. CRITICAL (found during extension) — silent "GRN missing → train without GRN"
`load_grn` returns `None` for a missing file (not raise), and `ensure_static_priors` guarded the
build with `try: load_grn(v) except: build_grn(v)` — so a missing GRN file silently skipped the build,
and `train()` then trained with `grn=None` (dense attention) while the checkpoint still recorded
`use_grn_mask=True`. This invalidated the `ground_truth` runs on `synthetic_hard`/`synthetic_hard_big`
(no `grn_ground_truth.npz` existed) and the `trrust_weighted` row on `tahoe_full` — all silently ran
*without* a GRN (≡ none). The real-Tahoe random/trrust/coexpr/coexpr_lfc rows were unaffected (their
files existed from v1), so the **headline null stands**, but the positive control had to be redone.
**Fix:** (1) `train()` now RAISES if a non-`none` variant's GRN is absent (loud failure); (2)
`ensure_static_priors` builds ground_truth by FILE-existence check, not by catching a (non-raising)
load. Lesson: a loader that returns `None` on missing input is a silent-failure trap — prefer raising,
and assert prerequisites at the point of use.

## K. CRITICAL (found during extension) — bf16 metric nondeterminism
`predict_test` used bf16 autocast. On well-behaved tasks (real Tahoe, DEG-Pearson ~0.62) this was
harmless, but on the near-zero-variance novel-target synthetic task it made the **point estimate
unstable run-to-run** (none = 0.024 / 0.350 / 0.271 across analyses of the *same* checkpoints) — and
nearly led to the wrong conclusion ("true GRN gives no benefit") when the deterministic truth is a
meaningful +0.06. **Fix:** eval in fp32 (no autocast). Lesson: metrics that feed conclusions must be
computed deterministically; verify a headline number with an independent direct recompute.

(Three result-invalidating bugs — A/B leakage, J silent-no-GRN, K bf16-nondeterminism — were caught
only by cross-checking; each would have produced a confidently wrong claim. This is the case for the
audit + tests + direct-recompute verification, i.e. the engineering rigor itself.)

## What "3 levels up" means here
1. **Correctness:** A, B (no leakage) — without this the study is not publishable.
2. **Rigor:** C, D, E, F, G — cluster CIs, corrections, a GRN-only baseline, weighted GRN, effect sizes.
3. **Engineering:** I — typed package, single sources of truth, real test suite, reproducibility
   manifests; so results are trustworthy and the study is extensible.

The v1 grid currently running is retained only as a **preliminary, partly-leaky sanity baseline**;
all headline numbers will come from the v2 leakage-free, cluster-bootstrapped pipeline.

## L. Mechanism (synthetic_hard_big) DEG50 was bf16-nondeterminism, not deterministic fp32 (2026-05-28)

During the publication-quality sprint we re-analyzed every checkpoint with the corrected stats
(`+1/(B+1)` continuity correction; see item M below). For Tahoe-100M the numbers reproduced within
±0.001. For the synthetic_hard_big mechanism study the numbers shifted notably:

| Run                                | none  | gt_mask | gt_prop | gt_maskprop |
|---|---|---|---|---|
| Published JSON (`1b092d8` commit)  | 0.368 | 0.432   | 0.427   | 0.437       |
| `1b092d8` `pmoe/` re-run today     | 0.330 | 0.335   | 0.313   | 0.331       |
| Master HEAD (corrected) re-run     | 0.313 | 0.342   | 0.330   | 0.337       |

Three different result sets from the same checkpoints on disk. The published JSON cannot be
deterministically reproduced from any code state — including the very commit (`1b092d8`) that
shipped it. This is consistent with AUDIT.md item K's documented bf16 nondeterminism on this
near-zero-variance task (point estimate swung 0.024 / 0.350 / 0.271 on the same checkpoints).
The published 0.368/0.432 was a one-time bf16-era analysis whose JSON happened to land in the
same commit as the fp32 fix; the directional sign (positive Δ) and the "ns" conclusion are
preserved across all three re-runs, but the magnitude was overstated.

**Fix:** master commits `f0b921c` (continuity correction) and `14ce619` (re-analysis with fp32
predict_test) publish the deterministic values (`none` 0.313, `gt_mask` 0.342, Δ=+0.029,
Holm-p=1.0). The paper text, results/PAPER.md, results/SUMMARY.md, and README all reflect these
numbers as of commit `14ce619`. **Lesson:** when a metric is sensitive enough that a single
nondeterministic run can swing it by 10×, a single "deterministic fp32" rerun is not enough
verification — re-run from disk after every code change that could touch the forward pass.

## M. Paired-bootstrap p-value floor — `p=0` was distributionally impossible (2026-05-28)

`pmoe/eval/stats.py` `paired_cluster_bootstrap` used the v1 estimator
`p = 2*min((boots<=0).mean(), (boots>=0).mean())`, which returns exactly 0.0 whenever every
bootstrap delta lands on one side of zero — a value the bootstrap can never produce. Every
"p<0.001" headline in the paper was actually this degenerate `p=0`, with no continuity correction.
Fixed in commit `f0b921c` with the standard `(2*min(n_le, n_ge) + 1)/(B+1)` correction; at
`B=2000` the floor is `1/2001 ≈ 5e-4`. The fix is exercised by two regression tests in
`tests/test_stats.py` (clean separation must hit the floor; null case must give `p>0.05`).
