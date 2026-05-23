# `pmoe/` package design & interface contract (v2)

Goal: refactor the working v1 (`code/*.py`) into a typed, tested package that fixes the methodology
flaws in `AUDIT.md`. **Behavior must match v1 where v1 was correct** (model math, metrics ranking),
and must FIX leakage (A/B), add cluster bootstrap + corrections (C/D), a GRN-propagation baseline
(E), weighted GRN (F), effect sizes (G), determinism (H), and dedup (I).

All modules import config from `pmoe.config` (already written) and provenance from `pmoe.io`
(already written). **Do not redefine paths, seeds, enums, ModelConfig, TrainConfig, RunSpec,
grn_filename — import them.** Data schema is unchanged: `data/schema.md`.

The conditions DataFrame columns: `cell_line, treatment, smiles, target_gene, dose, dose_log,
n_cells, n_ctrl_cells, ctrl_mean[list], pert_mean[list], lfc[list], deg_mask[list]`.

## pmoe/data/  (Agent A)
```python
# loader.py
def load_conditions(dataset:str)->pd.DataFrame
def load_genes(dataset:str)->list[str]
def load_meta(dataset:str)->dict
def stack_arrays(df, col:str)->np.ndarray            # (n,N) float32, bool for deg_mask

# splits.py
def make_splits(df, mode:str, seed:int=SEED, val_frac=0.1, test_frac=0.2)->dict
#   returns {"train","val","test": np.ndarray, "groups": np.ndarray}
#   groups = per-condition cluster label for cluster bootstrap: drug for unseen_drug/unseen_both,
#            cell_line for unseen_cell_line. Used by eval.stats. unseen_drug must guarantee
#            Tanimoto<=0.8 (Butina) no-leak; fall back to drug-name clusters w/o rdkit.

# dataset.py
def build_shared(dataset:str, df)->dict              # arrays: ctrl,lfc,deg,dose_log,chemberta,
#                                                      target_idx,cb_dim,n_genes (uses priors.drugs)
class ConditionDataset(torch.utils.data.Dataset)     # __getitem__ -> dict incl cell_line,treatment
def collate(batch)->dict                              # tensors + cell_line[list]+treatment[list]
def make_loader(dataset, idx, df, shared, batch_size, shuffle, num_workers=0)->DataLoader
```
Tests: split sizes nonempty; **no drug leakage** in unseen_drug; groups aligned to df rows.

## pmoe/priors/  (Agent B)
```python
# drugs.py
def build_drug_feats(dataset:str)->pd.DataFrame      # treatment,smiles,target_gene,morgan,chemberta
def load_drug_feats(dataset:str)->pd.DataFrame       # build if missing; offline ChemBERTa fallback

# pathways.py
def build_pathways(dataset:str, max_pathways=40)->sp.csr_matrix   # (N,P) bool; Reactome else fallback
def load_pathways(dataset:str)->tuple[sp.csr_matrix, list[str]]

# grn.py  -- the study's independent variable; THIS is where leakage is fixed
def build_grn(dataset:str, variant:GRNVariant, *, train_idx:np.ndarray|None=None,
              df=None, match_edges:int|None=None, seed:int=SEED)->sp.csr_matrix
#   variant rules:
#     none         -> empty
#     random       -> random edges, count=match_edges (default: trrust count)
#     trrust       -> TRRUST v2 binary, symbol->Ensembl via gene_vocabulary.jsonl
#     trrust_weighted -> float weights in [−1,1] (Activation=+,Repression=−,Unknown=±small); CSR float
#     coexpr       -> top-|corr| of pert_mean over TRAIN-ONLY rows (df.iloc[train_idx]); REQUIRES train_idx
#     coexpr_lfc   -> top-|corr| of lfc over TRAIN-ONLY rows; REQUIRES train_idx
#     ground_truth -> load existing grn_mask.npz (synthetic)
#   saves to priors_dir(dataset)/grn_filename(variant, split-if-data-derived) + .meta.json(source,n_edges,leakage_safe:bool)
def load_grn(dataset:str, variant, split=None)->sp.csr_matrix|None   # float for weighted, bool else
```
**Leakage rule (CRITICAL):** coexpr/coexpr_lfc MUST raise if `train_idx is None`. meta records
`leakage_safe=True` only when built from train_idx. Tests: a `test_grn_leakage.py` asserting that the
coexpr GRN built from train_idx is unchanged when test rows are perturbed.

## pmoe/models/  (Agent C)
```python
# layers.py: MoEFFN (expert-dispatch, memory-safe), MixedAttention (two-SDPA: grn-masked + dense;
#            supports float weighted bias when cfg.weighted_grn), CrossAttention
# pathway_moe.py
class PathwayMoEPerturb(nn.Module):
    def __init__(self, cfg:ModelConfig, grn=None, gene_pathway=None)  # accept sparse OR dense; densify
    def forward(self, batch:dict)->Tensor   # (B,N) pred lfc ; batch keys: ctrl_mean,target_idx,
                                             #   chemberta,dose_log
    def aux_loss(self)->Tensor
#   weighted GRN: if cfg.weighted_grn and grn is float, attention bias = log-ish of |w| (allow) with
#   sign optional; else binary allow/deny (−inf). Non-persistent buffers (rebuild at load).
# baselines.py
class B1MeanEffect / B2Ridge / B3RidgeBio:  .fit(df,train_idx,val_idx,fz,deg)/.predict(df,idx)->(n,N)
class GRNPropagationBaseline:               # audit E: pure GRN message-passing, no deep net
#   fit: learn per-target initial effect from train (drug->target signal); predict: propagate the
#   perturbation onto GRN neighbours over H hops with a fitted decay; uses load_grn(variant).
#   .fit(df,train_idx,grn, val_idx)/.predict(df,idx)->(n,N)
def count_params(m)->int
```
Tests: tiny-config fwd/bwd finite grads (CPU); MoE load-balance >0; weighted vs binary both run;
GRNPropagationBaseline beats zero-pred on synthetic.

## pmoe/eval/  (Agent D)
```python
# metrics.py
def deg_pearson_per_condition(pred,true,k=50)->np.ndarray     # top-k by |true|
def compute_metrics(pred,true,deg_mask=None)->dict            # pearson_all, deg{20,50,100}, mse, dir{...}
# stats.py
def bootstrap_ci(values,n_boot=2000,seed=SEED,alpha=0.05)->(mean,lo,hi)
def cluster_bootstrap_ci(values,groups,n_boot=2000,...)->(mean,lo,hi)   # resample groups (audit C)
def paired_cluster_bootstrap(a,b,groups,n_boot=2000,...)->dict          # delta,ci,p (two-sided)
def holm_correction(pvalues:dict[str,float])->dict[str,float]           # audit D
def effect_summary(delta, ci, p_corrected, min_effect=MIN_MEANINGFUL_EFFECT)->dict  # significant&meaningful
# loading.py  -- single canonical loader (dedup, audit I); reloads the CORRECT grn variant
def load_model_for_eval(run:RunSpec, dataset, variant, split, device)->PathwayMoEPerturb
def predict_test(run, dataset, df, shared, test_idx, variant, split, device)->np.ndarray
```
Tests: good-pred DEG50 > zero-pred; cluster CI wider than naive when groups correlated; Holm monotone.

## pmoe/experiments/  (I integrate, after A–D)
train.py (function `train(run, train_cfg, ...)` + CLI), study.py (grid: build train-only GRN per
(variant,split), train seeds, analyze with cluster bootstrap + Holm + effect sizes -> study json).

## Conventions for ALL agents
- Use `.venv\Scripts\python.exe`. **Do NOT** run GPU training, kill processes, touch `E:\` data being
  written, reinstall packages, or run the full grid. Only fast CPU unit tests (tiny synthetic tensors).
- Type hints + concise docstrings. Match v1 numerics where v1 was correct (see `code/`).
- Put tests in `tests/test_<module>.py`, runnable via `python -m pytest tests/test_<x>.py -q`.
- Create only files in your assigned subpackage + your tests. Don't edit `pmoe/config.py` or `pmoe/io.py`.
