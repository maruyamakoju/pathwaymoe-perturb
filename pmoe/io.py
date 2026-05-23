"""Checkpoint I/O, environment + data provenance manifests (reproducibility, audit item H)."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path

import torch

from pmoe.config import ModelConfig, RunSpec, meta_path, genes_path


def save_checkpoint(run: RunSpec, model_state: dict, model_cfg: ModelConfig, extra: dict | None = None):
    run.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_state, "cfg": model_cfg.to_dict(), "run": run.__dict__,
               "extra": extra or {}, "env": env_manifest()}
    torch.save(payload, run.ckpt_path)
    return run.ckpt_path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def env_manifest() -> dict:
    mods = {}
    for name in ("torch", "numpy", "scipy", "sklearn", "pandas", "scanpy", "transformers", "rdkit"):
        try:
            m = __import__(name)
            mods[name] = getattr(m, "__version__", "?")
        except Exception:
            mods[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": mods}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def data_manifest(dataset: str) -> dict:
    mp, gp = meta_path(dataset), genes_path(dataset)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    genes_hash = _sha1(gp.read_text()) if gp.exists() else None
    return {"dataset": dataset, "n_genes": meta.get("N_GENES"),
            "n_conditions": meta.get("n_conditions"), "n_cells_total": meta.get("n_cells_total"),
            "n_shards": meta.get("n_shards"), "genes_sha1": genes_hash,
            "cell_lines": len(meta.get("cell_lines", []))}
