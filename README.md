# PathwayMoE-Perturb — When does gene-regulatory structure help perturbation prediction?

A leakage-audited, statistically rigorous study of whether a **gene-regulatory-network (GRN)
inductive bias** improves *out-of-distribution* single-cell perturbation prediction — motivated by
the 2025 *Nature Methods* finding that billion-parameter foundation models do not beat linear
baselines. We hold the architecture fixed and vary only the **GRN quality** (none → random → curated
TRRUST → weighted → dense curated CollecTRI → data-derived co-expression/co-response → ground-truth)
and the **injection mechanism** (hard attention mask vs soft GNN message-passing).

## TL;DR finding

> A structured GRN prior confers **no significant benefit** for OOD perturbation prediction **where
> prediction is feasible** — on a 22.6M-cell real Tahoe-100M subset and on synthetic data with shared
> drug targets it is *redundant* (a flexible model learns the response directly). This holds across the
> whole **quality spectrum**: generic (TRRUST), signed-weighted, **dense curated (CollecTRI, 4× denser
> — indistinguishable from random at the same density)**, and data-derived GRNs. A different kind of
> prior — **drug→target mechanistic grounding** — is *also* null on OOD (a synthetic positive control
> confirms the pathway works, so the null is real, not a plumbing bug). The true GRN only *plausibly*
> helps for **novel targets** (+0.05–0.08 DEG-Pearson, meaningful but not significant), a
> near-unpredictable regime, and the **injection mechanism does not matter**.

| Model (unseen-drug, 39 drug clusters, cluster-bootstrap 95% CI) | DEG-Pearson@50 |
|---|---|
| ridge (B2/B3) | 0.582 |
| mean-effect (B1) | 0.601 |
| PathwayMoE, **no GRN** | **0.629** |
| PathwayMoE, TRRUST / weighted / co-expr / co-response / random | 0.622–0.628 (all Δ vs none ≤0.007, ns) |
| PathwayMoE, **CollecTRI** (4× denser, signed, curated) | 0.624 (Δ −0.005 vs none, ns) |
| PathwayMoE, random @ CollecTRI density | 0.624 (Δ vs CollecTRI = **+0.001**, ns) |
| PathwayMoE, **no GRN + drug→target** | 0.632 (Δ +0.003, p=0.93, ns) |

> The "TRRUST was too sparse" objection is closed: a 4×-denser signed curated GRN (CollecTRI) is
> indistinguishable from random rewiring at the same density (Δ=+0.001, ns). And drug→target
> mechanistic grounding — the one prior we hadn't tried — is also null on OOD (Δ=+0.003, ns), though
> a synthetic positive control confirms the target pathway is functional. Neither regulatory-graph
> structure nor mechanistic target grounding moves OOD-to-novel-drug accuracy here.

Full numbers, synthetic positive control, and discussion: [`results/PAPER.md`](results/PAPER.md).
One-page summary + the money figure: [`results/SUMMARY.md`](results/SUMMARY.md).
Methodology audit trail — every result-invalidating bug we caught and fixed (items A–P, incl. GRN
test-set leakage, a silent train-without-GRN fallback, bf16/CUDA non-determinism, a p-value floor, and
a fabricated-number near-miss caught before it entered git): [`AUDIT.md`](AUDIT.md). This audit log is
a first-class artifact of the project — the negative result is only trustworthy because each of these
was caught.

### Scope & honest framing (please don't over-read this)
This is a rigorous **negative/limited result on one common design**: a GRN injected as an attention
mask or message-passing prior, one model family (PathwayMoE), 2,000 HVGs, the Tahoe-100M chemical-
perturbation setting. We do **not** claim GRNs are useless in general, nor that no architecture can use
them — only that *this* prior does not improve OOD perturbation prediction **where prediction is
feasible** (shared targets), and is at best suggestive (ns) where targets are novel. The original
"small biology-constrained model beats giant foundation models" claim was **not** tested here (we
pivoted to the controlled GRN-quality study); a head-to-head vs scGPT/STATE/Tahoe-x1 is future work.

## Why this is trustworthy (the rigor)

- **No leakage**: data-derived GRNs are built from *training conditions only*, per split
  (`leakage_safe=true`); a regression test asserts test rows can't change the GRN.
- **Cluster bootstrap** (resampling drugs/cell-lines, the real independent units) + **Holm**
  correction + pre-registered **minimum meaningful effect** (0.01); we report the # of test clusters.
- **Deterministic CPU eval** for near-zero-variance regimes; headline numbers independently
  re-verified by direct recompute from checkpoints.
- **67 unit/integration tests** incl. a leakage regression test and a checkpoint-roundtrip test.
- We caught and fixed **several result-invalidating bugs** — test-set GRN leakage (A), a silent
  train-without-GRN fallback (J), bf16/CUDA metric non-determinism (K, N), and a paired-bootstrap
  p-value floor that made every "p<0.001" a distributionally-impossible p=0 (M) — see `AUDIT.md`
  (items A–P). One was a fabricated number caught *before* it reached git (O).

## Package layout (`pmoe/`)

```
pmoe/
  config.py        typed configs, enums, RunSpec (single source for paths/seeds/names)
  io.py            checkpoint + env/data provenance manifests
  data/            loader, leakage-controlled splits (+cluster groups), dataset/collator,
                   synth (synthetic ground-truth dataset), preprocess_tahoe (Tahoe-100M streaming)
  priors/          grn.py (variant spectrum, TRAIN-ONLY data-derived), pathways, drugs (ChemBERTa)
  models/          pathway_moe (GRN-masked SDPA attn + pathway MoE + pert cross-attn),
                   layers (incl. GRNPropagation soft message-passing), baselines (+GRN-propagation)
  eval/            metrics (DEG-Pearson@K), stats (cluster bootstrap, Holm, effect size), loading,
                   report (tables+figures)
  experiments/     train (robust in-process), study (grid runner), mechanism (mask vs propagation)
tests/             55+ tests incl. test_grn_leakage.py
code/archive/      v1 scripts (superseded by pmoe/; kept for provenance only)
```

Data + checkpoints live on `E:\vc_project_data` (`VC_DATA_ROOT`); not in the repo.

## Reproduce

```powershell
.\.venv\Scripts\activate ; $env:VC_DATA_ROOT="E:\vc_project_data"
python -m pytest -q                                   # 55+ tests
# synthetic (controlled) + positive control:
python -m pmoe.data.synth --dataset synthetic
python -m pmoe.experiments.study  --dataset synthetic --splits unseen_drug --variants none random coexpr coexpr_lfc ground_truth
python -m pmoe.data.synth --dataset synthetic_hard_big --unique-targets --n-extra-drugs 150
python -m pmoe.experiments.mechanism --dataset synthetic_hard_big
# real Tahoe (needs the 22.6M-cell subset; see pmoe/data/preprocess_tahoe.py to build it):
python -m pmoe.experiments.study  --dataset tahoe_full --splits unseen_drug \
       --variants none random trrust trrust_weighted coexpr coexpr_lfc
python -m pmoe.eval.report --dataset tahoe_full        # table + figure
```

See `reproduce.ps1` for the end-to-end script.

## Status

Native Windows + RTX 4090. Real data = 803/3388 Tahoe-100M shards (22.66M cells → 8,875 conditions ×
2,000 HVGs, mean 2,480 cells/condition, matched DMSO controls). This is a research artifact, not a
package release; the conclusion (GRN-as-prior is redundant where prediction is feasible) is robust.

## License & citation

MIT-licensed (see `LICENSE`). If you use the code or the study, citation metadata is in
`CITATION.cff`; the preprint DOI will be added once the arXiv/bioRxiv upload is live.
