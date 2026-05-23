# Data Schema & Interface Contract — PathwayMoE-Perturb

All modules code against this contract. **Do not change without updating every consumer.**

## Design choice: condition-level (pseudobulk) modeling

The field's metric (Virtual Cell Challenge, the 2025 Nature Methods baseline paper) is computed on
**pseudobulk differential expression per perturbation condition**. The linear baselines operate on
log-fold-change (LFC) vectors. To make the fancy model *directly comparable* to those baselines and
that metric, the core training/eval unit is a **condition**, not a single cell.

A *condition* = a unique `(cell_line, treatment, dose)` tuple. For each condition we store the
pseudobulk mean log-normalized expression of perturbed cells, the matched DMSO control mean from the
same cell line, and their difference (LFC). Per-cell data is kept separately and is optional (used
only for the consistency loss / cell-encoder ablation).

## Paths

- Data root: `E:\vc_project_data` (env var `VC_DATA_ROOT`, default this path).
- Processed conditions: `${VC_DATA_ROOT}/data/<dataset>/processed/conditions.parquet`
- Per-cell (optional): `${VC_DATA_ROOT}/data/<dataset>/processed/cells.npz` (+ `cells_meta.parquet`)
- Gene list: `.../processed/genes.txt`        (one HGNC symbol per line, order == array index)
- Metadata: `.../processed/meta.json`          (cell_lines, drugs, dose units, N_GENES, dataset name)
- Priors:   `${VC_DATA_ROOT}/data/priors/`
- Results:  `<repo>/results/`, Checkpoints: `${VC_DATA_ROOT}/checkpoints/`

`<dataset>` ∈ {`synthetic`, `tahoe`, `replogle`}.

## `conditions.parquet` — the core table (one row per condition)

| column        | dtype                | notes |
|---------------|----------------------|-------|
| `condition_id`| string               | f"{cell_line}|{treatment}|{dose}" |
| `cell_line`   | string (categorical) | e.g. "A549" |
| `treatment`   | string               | drug name, or "DMSO" for controls (controls usually excluded from the table) |
| `smiles`      | string               | canonical SMILES; "" if unknown/genetic |
| `target_gene` | string               | primary target HGNC symbol; "" if unknown |
| `dose`        | float32              | raw dose (uM for chemical) |
| `dose_log`    | float32              | log1p(dose) |
| `n_cells`     | int32                | # perturbed cells aggregated |
| `n_ctrl_cells`| int32                | # matched DMSO cells aggregated |
| `ctrl_mean`   | list<float32>[N]     | pseudobulk mean log1p-normalized expr of matched DMSO |
| `pert_mean`   | list<float32>[N]     | pseudobulk mean log1p-normalized expr of perturbed cells |
| `lfc`         | list<float32>[N]     | `pert_mean - ctrl_mean` (this is the prediction TARGET) |
| `deg_mask`    | list<bool>[N]        | True where gene is a DEG (see metrics.py definition) |

N = number of genes == `len(genes.txt)` == `meta["N_GENES"]`.

## `meta.json`

```json
{ "dataset": "synthetic", "N_GENES": 2000, "cell_lines": ["A549", ...],
  "drugs": [{"name":"trametinib","smiles":"...","target_gene":"MAP2K1"}, ...],
  "dose_unit": "uM", "created": "ISO8601", "notes": "..." }
```

## Priors (produced by grn.py / pathways.py / drugs.py, consumed by model.py)

- `priors/grn_mask.npz`     : scipy sparse bool CSR, shape (N, N). `mask[i,j]=True` ⇒ gene i (TF) regulates gene j (target). Row/col order == genes.txt.
- `priors/pathways.npz`     : sparse bool CSR (N, P). `[g,p]=True` ⇒ gene g in pathway p. `pathway_names.txt` lists P names. P used for MoE expert grouping (top-level pathways collapsed to N_EXPERTS).
- `priors/drug_feats.parquet`: columns `treatment, smiles, target_gene, morgan(list<float32>[2048]), chemberta(list<float32>[D])`.

Every prior builder MUST: align to a passed-in `genes.txt`, and provide a deterministic offline
fallback (random-but-seeded mask of plausible density) when the source DB / network is unavailable,
recording `source="fallback"` in a sidecar `.meta.json` so results are never silently fake.

## Python API contract (function signatures other modules rely on)

```python
# loader.py
def load_conditions(dataset: str) -> pd.DataFrame      # parquet -> df with np arrays in cells
def load_genes(dataset: str) -> list[str]
def load_meta(dataset: str) -> dict
def stack_arrays(df, col) -> np.ndarray                 # (n_conditions, N) float32

# splits.py
def make_splits(df, mode: str, seed=1337, val_frac=0.1) -> dict[str, np.ndarray]
#   mode in {"unseen_drug","unseen_cell_line","unseen_both"}; returns index arrays
#   {"train","val","test"}. unseen_drug guarantees Tanimoto<=0.8 between train/test drugs.

# metrics.py
def compute_metrics(pred_lfc, true_lfc, deg_mask=None) -> dict   # per-condition then averaged
#   keys: pearson_all, pearson_deg{20,50,100}, mse, direction_acc{20,50,100}
def evaluate(pred, true, deg_mask, groups=None) -> dict          # adds per-group means + std

# baselines.py:  B1MeanEffect, B2Ridge, B3RidgeBio  -> .fit(df, train_idx) / .predict(df, idx)
# model.py:      PathwayMoEPerturb(cfg).forward(batch) -> pred_lfc (B,N)
```

All randomness seeded with **1337** unless stated. bf16 for model, fp32 for metrics.
