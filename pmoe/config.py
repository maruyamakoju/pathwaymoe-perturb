"""Typed configuration hub — single source of truth for paths, seeds, enums, and configs.

Imports only the stdlib so every module can import this cheaply. Heavy libs live in submodules.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

SEED = 1337

# ----- paths -----
DATA_ROOT = Path(os.environ.get("VC_DATA_ROOT", r"E:\vc_project_data"))
REPO_ROOT = Path(__file__).resolve().parent.parent
PRIORS_ROOT = DATA_ROOT / "data" / "priors"
CKPT_DIR = DATA_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"
for _d in (CKPT_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset: str) -> Path:
    return DATA_ROOT / "data" / dataset / "processed"


def conditions_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "conditions.parquet"


def genes_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "genes.txt"


def meta_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "meta.json"


def priors_dir(dataset: str) -> Path:
    p = PRIORS_ROOT / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----- enums -----
class Split(str, Enum):
    UNSEEN_DRUG = "unseen_drug"
    UNSEEN_CELL_LINE = "unseen_cell_line"
    UNSEEN_BOTH = "unseen_both"


class GRNVariant(str, Enum):
    NONE = "none"                 # no GRN mask
    RANDOM = "random"             # random edges, matched density
    TRRUST = "trrust"             # generic curated TF->target (binary)
    TRRUST_WEIGHTED = "trrust_weighted"   # signed/weighted edges
    COEXPR = "coexpr"             # data-derived: gene-gene corr of expression (TRAIN-ONLY)
    COEXPR_LFC = "coexpr_lfc"     # data-derived: gene-gene corr of LFC (TRAIN-ONLY)
    GROUND_TRUTH = "ground_truth" # synthetic only: the true GRN

DATA_DERIVED = {GRNVariant.COEXPR, GRNVariant.COEXPR_LFC}   # must be built train-only, per split


def grn_filename(variant: "GRNVariant | str", split: "Split | str | None" = None) -> str:
    """Canonical npz filename for a GRN variant. Data-derived variants are split-specific.
    Note: GRNVariant/Split subclass str, so we normalize via the enum to get the .value (the
    bare string), never the 'GRNVariant.X' repr."""
    v = GRNVariant(variant).value                      # works for enum or bare string
    if v == "none":
        return ""                                      # no file
    if GRNVariant(v) in DATA_DERIVED and split is not None:
        s = Split(split).value
        return f"grn_{v}__{s}.npz"
    return f"grn_{v}.npz"


# ----- configs -----
@dataclass
class ModelConfig:
    n_genes: int = 2000
    n_experts: int = 40
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_pert: int = 256
    chemberta_dim: int = 384
    dropout: float = 0.1
    top_k: int = 2
    grn_head_frac: float = 0.5
    use_grn_mask: bool = True
    use_moe: bool = True
    use_pathway_prior: bool = True
    pert_as_token: bool = False
    weighted_grn: bool = False     # if True, attention bias uses edge weights (not just allow/deny)
    grn_propagation: bool = False  # if True, apply SOFT GRN message-passing (orthogonal to mask)
    grn_prop_hops: int = 2         # number of propagation hops (P^1..P^k) when grn_propagation=True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 96
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup: int = 200
    lb_weight: float = 0.1
    patience: int = 6
    eval_every: int = 2
    grad_ckpt: bool = False
    seed: int = SEED
    deterministic: bool = True


@dataclass
class RunSpec:
    """Fully identifies one trained model (-> deterministic checkpoint name)."""
    dataset: str
    split: str
    variant: str = "none"
    seed: int = SEED
    size: str = "base"
    tag: str = ""              # optional mechanism tag (e.g. "proponly"); keeps names distinct

    @property
    def name(self) -> str:
        parts = [self.dataset, self.split, self.size, self.variant]
        if self.tag:
            parts.append(self.tag)
        if self.seed != SEED:
            parts.append(f"s{self.seed}")
        return "__".join(parts)

    @property
    def ckpt_path(self) -> Path:
        return CKPT_DIR / f"{self.name}.pt"


SIZE_PRESETS = {
    "tiny":  dict(d_model=64, n_heads=4, n_layers=2, d_pert=64),
    "small": dict(d_model=192, n_heads=6, n_layers=3, d_pert=192),
    "base":  dict(d_model=256, n_heads=8, n_layers=4, d_pert=256),
}

MIN_MEANINGFUL_EFFECT = 0.01   # pre-registered minimum meaningful Δ DEG-Pearson (audit item G)
