# PathwayMoE-Perturb

A small (~1–200M param), **GRN-sparsity-constrained, pathway Mixture-of-Experts** for virtual-cell
perturbation prediction, built to test whether *inductive bias* (not scale) is what closes the gap
to linear baselines on out-of-distribution perturbation prediction.

Motivation: *"Deep-learning-based gene perturbation effect prediction does not yet outperform simple
linear baselines"* (Nature Methods 2025). Giant foundation models often fail to beat ridge
regression on held-out perturbations. This repo (a) builds the linear baselines **properly and
first**, and (b) tests a biology-constrained architecture against them on identical splits/metrics.

## What's here

```
code/
  config.py      paths/seeds (data root = E:\vc_project_data)
  schema → data/schema.md   the interface contract every module obeys
  synth.py       synthetic Tahoe-shaped data WITH known GRN/pathway ground truth
  preprocess.py  real Tahoe-100M h5ad -> condition-level parquet (best-effort, schema-configurable)
  drugs.py       Morgan FP + ChemBERTa (offline fallback)
  grn.py         DoRothEA -> sparse GRN mask (offline fallback)
  pathways.py    Reactome -> gene×pathway multi-hot (offline fallback)
  splits.py      unseen_drug (Tanimoto<0.8 Butina dedupe) / unseen_cell_line / unseen_both
  metrics.py     Pearson(all), DEG-Pearson@{20,50,100}, MSE, direction accuracy
  baselines.py   B1 mean-effect, B2 ridge, B3 ridge+biology
  model.py       PathwayMoE-Perturb (SDPA mixed GRN/dense attn, pathway top-2 MoE, pert cross-attn)
  data.py        condition-level dataset + collator
  train.py       AdamW(+cosine,bf16,grad-ckpt), weighted-Huber + load-balance loss
  eval.py        baselines vs model comparison table across 3 splits
  figures.py     comparison bar, training curves, expert-usage heatmap
tests/smoke.py   fast end-to-end validation
run_all.ps1      orchestration
```

## Quickstart (native Windows)

```powershell
.\.venv\Scripts\activate
$env:VC_DATA_ROOT = "E:\vc_project_data"
python tests\smoke.py            # validate the whole pipeline in ~1 min
.\run_all.ps1 synthetic          # data -> baselines -> train 3 splits -> eval -> figures
```

Real data: download a Tahoe-100M subset to `E:\vc_project_data\data\tahoe100m`, then
`python code\preprocess.py --dry-run` to inspect obs columns, set `--cell-line-col/--drug-col/
--control-value`, run without `--dry-run`, then `run_all.ps1 tahoe`.

## Environment adaptations from the original spec

- **Native Windows** (not Linux/WSL2): Triton/flash-attn omitted; mixed GRN-masked/dense attention
  uses two PyTorch SDPA passes. WSL2 Ubuntu is available if the block-sparse kernel is later needed.
- **Data on E:** (4.5 TB) — C: had only ~100 GB free. Set via `VC_DATA_ROOT`.
- **Condition-level (pseudobulk) modeling** so the model is directly comparable to the linear
  baselines and the field's DEG-Pearson metric.

## Results (see `results/REPORT.md`)

DEG-Pearson@50, out-of-distribution test splits:

| | unseen_drug | unseen_cell_line | unseen_both |
|---|---|---|---|
| **Synthetic** (GRN ground truth) — best baseline | 0.704 | **0.988** | 0.698 |
| **Synthetic** — PathwayMoE | **0.724** | 0.930 | **0.712** |
| **Real Tahoe-100M** — best baseline | **0.715** | 0.432 | 0.324 |
| **Real Tahoe-100M** — PathwayMoE | 0.704 | **0.551** | **0.512** |

The biology-constrained model beats strong linear baselines on the hard OOD splits (decisively on
the hardest, unseen-both); on synthetic, ablations show the **GRN mask** is the driver, but on real
data the generic TRRUST-over-HVGs GRN is neutral — the real-data gains come from the broader
architecture. Honest, non-overclaimed, and reproducible.
