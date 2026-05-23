"""Central config + paths. Single source of truth for the data root and seeds."""
from __future__ import annotations
import os
from pathlib import Path

SEED = 1337

DATA_ROOT = Path(os.environ.get("VC_DATA_ROOT", r"E:\vc_project_data"))
REPO_ROOT = Path(__file__).resolve().parent.parent

PRIORS_DIR = DATA_ROOT / "data" / "priors"
CKPT_DIR = DATA_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"

for _d in (PRIORS_DIR, CKPT_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def dataset_dir(dataset: str) -> Path:
    return DATA_ROOT / "data" / dataset / "processed"


def conditions_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "conditions.parquet"


def genes_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "genes.txt"


def meta_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "meta.json"
