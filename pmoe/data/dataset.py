"""Condition-level dataset + collator for PathwayMoE-Perturb.

Everything fits in RAM (conditions are pseudobulk: a few thousand rows of length-N vectors), so we
pre-stack into numpy arrays once and index cheaply. The model consumes::

    ctrl_mean (N), target_idx (int), chemberta (Dc), dose_log (float)  -> predicts lfc (N).

Ported from v1 ``code/data.py``; imports are routed through the ``pmoe`` package. ``build_shared``
depends on :func:`pmoe.priors.drugs.load_drug_feats`, which is built in parallel (Agent B) — the
import is intentionally local so this module imports cleanly even before that exists.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pmoe.data.loader import load_conditions, load_genes, stack_arrays


def build_shared(dataset: str, df) -> dict:
    """Pre-stack the per-condition arrays the model/dataset reuse across indices.

    Returns a dict of dense arrays: ``ctrl, lfc, deg, dose_log, chemberta, target_idx`` plus the
    scalars ``cb_dim`` and ``n_genes``. ChemBERTa features and target indices are resolved via the
    drug-feature table and the gene vocabulary.
    """
    from pmoe.priors.drugs import load_drug_feats  # local import: built in parallel (Agent B)

    genes = load_genes(dataset)
    gidx = {g: i for i, g in enumerate(genes)}
    dfeats = load_drug_feats(dataset).set_index("treatment")
    cb_map = {t: np.asarray(r, np.float32) for t, r in dfeats["chemberta"].items()}
    cb_dim = len(next(iter(cb_map.values())))
    chemberta = np.stack([cb_map.get(t, np.zeros(cb_dim, np.float32)) for t in df["treatment"]])
    target_idx = np.array([gidx.get(t, -1) for t in df["target_gene"].fillna("")], np.int64)
    return {
        "ctrl": stack_arrays(df, "ctrl_mean"),
        "lfc": stack_arrays(df, "lfc"),
        "deg": stack_arrays(df, "deg_mask"),
        "dose_log": df["dose_log"].to_numpy(np.float32),
        "chemberta": chemberta.astype(np.float32),
        "target_idx": target_idx,
        "cb_dim": cb_dim,
        "n_genes": len(genes),
    }


class ConditionDataset(Dataset):
    """Index-into-shared-arrays dataset; one item == one condition (pseudobulk row)."""

    def __init__(self, dataset: str, indices: np.ndarray, df=None, shared: dict | None = None):
        self.dataset = dataset
        if df is None:
            df = load_conditions(dataset)
        self.idx = np.asarray(indices)
        if shared is None:
            shared = build_shared(dataset, df)
        self.s = shared
        self.cell_line = df["cell_line"].to_numpy()
        self.treatment = df["treatment"].to_numpy()

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        j = self.idx[i]
        s = self.s
        return {
            "ctrl_mean": s["ctrl"][j],
            "lfc": s["lfc"][j],
            "deg_mask": s["deg"][j],
            "target_idx": s["target_idx"][j],
            "chemberta": s["chemberta"][j],
            "dose_log": s["dose_log"][j],
            "cell_line": self.cell_line[j],
            "treatment": self.treatment[j],
        }


def collate(batch: list[dict]) -> dict:
    """Collate a list of condition dicts into batched tensors + cell_line/treatment lists."""
    out: dict = {}
    out["ctrl_mean"] = torch.from_numpy(np.stack([b["ctrl_mean"] for b in batch])).float()
    out["lfc"] = torch.from_numpy(np.stack([b["lfc"] for b in batch])).float()
    out["deg_mask"] = torch.from_numpy(np.stack([b["deg_mask"] for b in batch])).bool()
    out["chemberta"] = torch.from_numpy(np.stack([b["chemberta"] for b in batch])).float()
    out["target_idx"] = torch.tensor([int(b["target_idx"]) for b in batch], dtype=torch.long)
    out["dose_log"] = torch.tensor([float(b["dose_log"]) for b in batch], dtype=torch.float)
    out["cell_line"] = [b["cell_line"] for b in batch]
    out["treatment"] = [b["treatment"] for b in batch]
    return out


def make_loader(dataset: str, indices: np.ndarray, df, shared: dict, batch_size: int = 16,
                shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Build a ``DataLoader`` over the given condition indices using the shared arrays."""
    ds = ConditionDataset(dataset, indices, df=df, shared=shared)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, drop_last=False, pin_memory=torch.cuda.is_available())
