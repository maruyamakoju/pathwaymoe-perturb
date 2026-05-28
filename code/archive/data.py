"""Condition-level dataset + collator for PathwayMoE-Perturb.

Everything fits in RAM (conditions are pseudobulk: a few thousand rows of length-N vectors), so we
pre-stack into numpy arrays once and index cheaply. The model consumes:
  ctrl_mean (N), target_idx (int), chemberta (Dc), dose_log (float)  -> predicts lfc (N).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from loader import load_conditions, load_genes, stack_arrays
from drugs import load_drug_feats


class ConditionDataset(Dataset):
    def __init__(self, dataset: str, indices: np.ndarray, df=None, shared=None):
        self.dataset = dataset
        if df is None:
            df = load_conditions(dataset)
        self.idx = np.asarray(indices)
        if shared is None:
            shared = build_shared(dataset, df)
        self.s = shared
        self.cell_line = df["cell_line"].to_numpy()
        self.treatment = df["treatment"].to_numpy()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
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


def build_shared(dataset: str, df):
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


def collate(batch):
    out = {}
    out["ctrl_mean"] = torch.from_numpy(np.stack([b["ctrl_mean"] for b in batch])).float()
    out["lfc"] = torch.from_numpy(np.stack([b["lfc"] for b in batch])).float()
    out["deg_mask"] = torch.from_numpy(np.stack([b["deg_mask"] for b in batch])).bool()
    out["chemberta"] = torch.from_numpy(np.stack([b["chemberta"] for b in batch])).float()
    out["target_idx"] = torch.tensor([int(b["target_idx"]) for b in batch], dtype=torch.long)
    out["dose_log"] = torch.tensor([float(b["dose_log"]) for b in batch], dtype=torch.float)
    out["cell_line"] = [b["cell_line"] for b in batch]
    out["treatment"] = [b["treatment"] for b in batch]
    return out


def make_loader(dataset, indices, df, shared, batch_size=16, shuffle=True, num_workers=0):
    ds = ConditionDataset(dataset, indices, df=df, shared=shared)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, drop_last=False, pin_memory=torch.cuda.is_available())
