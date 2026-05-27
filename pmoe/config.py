"""Typed configuration hub — hierarchical source of truth for the PMoE framework.

This module provides a structured, type-safe configuration system using nested dataclasses.
It is designed to be lightweight, importing only standard libraries where possible.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Global constants
SEED = 1337
MIN_MEANINGFUL_EFFECT = 0.01  # Minimum meaningful Δ DEG-Pearson

# ----- Paths -----
DATA_ROOT = Path(os.environ.get("VC_DATA_ROOT", r"E:\vc_project_data"))
REPO_ROOT = Path(__file__).resolve().parent.parent
PRIORS_ROOT = DATA_ROOT / "data" / "priors"
CKPT_DIR = DATA_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"

# Ensure essential directories exist
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


# ----- Enums -----
class Split(str, Enum):
    UNSEEN_DRUG = "unseen_drug"
    UNSEEN_CELL_LINE = "unseen_cell_line"
    UNSEEN_BOTH = "unseen_both"


class GRNVariant(str, Enum):
    NONE = "none"
    RANDOM = "random"
    TRRUST = "trrust"
    TRRUST_WEIGHTED = "trrust_weighted"
    COEXPR = "coexpr"
    COEXPR_LFC = "coexpr_lfc"
    GROUND_TRUTH = "ground_truth"


DATA_DERIVED = {GRNVariant.COEXPR, GRNVariant.COEXPR_LFC}


def grn_filename(variant: Union[GRNVariant, str], split: Optional[Union[Split, str]] = None) -> str:
    """Returns the canonical filename for a GRN variant."""
    v = GRNVariant(variant).value
    if v == "none":
        return ""
    if GRNVariant(v) in DATA_DERIVED and split is not None:
        s = Split(split).value
        return f"grn_{v}__{s}.npz"
    return f"grn_{v}.npz"


# ----- Hierarchical Configuration -----

@dataclass(frozen=True)
class ConfigBase:
    """Base class for all configurations providing serialization utilities."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class ArchitectureConfig(ConfigBase):
    """Core Transformer architecture parameters."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    n_genes: int = 2000


@dataclass(frozen=True)
class PerturbationConfig(ConfigBase):
    """Perturbation embedding parameters."""
    d_pert: int = 256
    chemberta_dim: int = 384
    pert_as_token: bool = False  # If True, add to gene tokens; else use cross-attention


@dataclass(frozen=True)
class GRNConfig(ConfigBase):
    """Gene Regulatory Network (GRN) integration settings."""
    use_grn_mask: bool = True
    grn_head_frac: float = 0.5
    weighted_grn: bool = False
    grn_propagation: bool = False
    grn_prop_hops: int = 2
    latent_grn: bool = False
    latent_grn_dim: int = 128
    latent_grn_kl_weight: float = 0.001


@dataclass(frozen=True)
class MoEConfig(ConfigBase):
    """Mixture-of-Experts (MoE) settings."""
    use_moe: bool = True
    use_pathway_prior: bool = True
    n_experts: int = 40
    top_k: int = 2


@dataclass(frozen=True)
class ModelConfig(ConfigBase):
    """Root model configuration, nesting sub-configs."""
    arch: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    pert: PerturbationConfig = field(default_factory=PerturbationConfig)
    grn: GRNConfig = field(default_factory=GRNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)

    # Shorthand properties for flat access (maintaining backward compatibility where useful)
    @property
    def n_genes(self) -> int: return self.arch.n_genes
    @property
    def d_model(self) -> int: return self.arch.d_model
    @property
    def n_heads(self) -> int: return self.arch.n_heads
    @property
    def n_layers(self) -> int: return self.arch.n_layers
    @property
    def dropout(self) -> float: return self.arch.dropout
    @property
    def d_pert(self) -> int: return self.pert.d_pert
    @property
    def chemberta_dim(self) -> int: return self.pert.chemberta_dim
    @property
    def pert_as_token(self) -> bool: return self.pert.pert_as_token
    @property
    def use_grn_mask(self) -> bool: return self.grn.use_grn_mask
    @property
    def grn_head_frac(self) -> float: return self.grn.grn_head_frac
    @property
    def weighted_grn(self) -> bool: return self.grn.weighted_grn
    @property
    def grn_propagation(self) -> bool: return self.grn.grn_propagation
    @property
    def grn_prop_hops(self) -> int: return self.grn.grn_prop_hops
    @property
    def latent_grn(self) -> bool: return self.grn.latent_grn
    @property
    def latent_grn_dim(self) -> int: return self.grn.latent_grn_dim
    @property
    def latent_grn_kl_weight(self) -> float: return self.grn.latent_grn_kl_weight
    @property
    def use_moe(self) -> bool: return self.moe.use_moe
    @property
    def use_pathway_prior(self) -> bool: return self.moe.use_pathway_prior
    @property
    def n_experts(self) -> int: return self.moe.n_experts
    @property
    def top_k(self) -> int: return self.moe.top_k


@dataclass(frozen=True)
class TrainConfig(ConfigBase):
    """Training hyperparameters and environment settings."""
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


@dataclass(frozen=True)
class RunSpec(ConfigBase):
    """Identifies a specific experiment run."""
    dataset: str
    split: str
    variant: str = "none"
    seed: int = SEED
    size: str = "base"
    tag: str = ""

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


# Presets for model sizes
SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":  dict(d_model=64, n_heads=4, n_layers=2, d_pert=64),
    "small": dict(d_model=192, n_heads=6, n_layers=3, d_pert=192),
    "base":  dict(d_model=256, n_heads=8, n_layers=4, d_pert=256),
}
