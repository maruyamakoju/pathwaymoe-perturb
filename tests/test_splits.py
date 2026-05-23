"""Self-contained tests for pmoe.data.splits — no real data, no other pmoe subpackages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmoe.config import SEED, Split
from pmoe.data.splits import _HAVE_RDKIT, make_splits

# A few real, distinct SMILES so the Butina/Tanimoto path actually runs when rdkit is present.
_SMILES = [
    "CCO",                       # ethanol
    "CC(=O)O",                   # acetic acid
    "c1ccccc1",                  # benzene
    "CC(=O)Oc1ccccc1C(=O)O",     # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "C1CCCCC1",                  # cyclohexane
    "CCN(CC)CC",                 # triethylamine
    "OCC(O)CO",                  # glycerol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "Cn1cnc2c1c(=O)n(C)c(=O)n2",   # theophylline-ish
]


def _fake_conditions(n_drugs: int = 10, with_smiles: bool = True) -> pd.DataFrame:
    """Build a tiny conditions DataFrame matching data/schema.md columns."""
    cell_lines = ["A549", "MCF7", "K562", "A375"]
    n_genes = 6
    rng = np.random.default_rng(0)
    rows = []
    for ci, cl in enumerate(cell_lines):
        for d in range(n_drugs):
            smiles = _SMILES[d % len(_SMILES)] if with_smiles else ""
            ctrl = rng.normal(size=n_genes).astype(np.float32)
            pert = (ctrl + rng.normal(scale=0.5, size=n_genes)).astype(np.float32)
            lfc = (pert - ctrl).astype(np.float32)
            rows.append({
                "condition_id": f"{cl}|drug{d}|1.0",
                "cell_line": cl,
                "treatment": f"drug{d}",
                "smiles": smiles,
                "target_gene": "",
                "dose": 1.0,
                "dose_log": float(np.log1p(1.0)),
                "n_cells": 100,
                "n_ctrl_cells": 100,
                "ctrl_mean": ctrl.tolist(),
                "pert_mean": pert.tolist(),
                "lfc": lfc.tolist(),
                "deg_mask": (np.abs(lfc) > 0.3).tolist(),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["unseen_drug", "unseen_cell_line", "unseen_both"])
def test_splits_nonempty_and_disjoint(mode):
    df = _fake_conditions()
    s = make_splits(df, mode, seed=SEED)
    for key in ("train", "val", "test", "groups"):
        assert key in s, f"missing key {key}"
    for key in ("train", "val", "test"):
        assert len(s[key]) > 0, f"{mode}: empty split {key}"
    # index splits must be disjoint
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


@pytest.mark.parametrize("mode", ["unseen_drug", "unseen_cell_line", "unseen_both"])
def test_groups_aligned(mode):
    df = _fake_conditions()
    s = make_splits(df, mode, seed=SEED)
    groups = s["groups"]
    assert isinstance(groups, np.ndarray)
    assert len(groups) == len(df), "groups must be aligned to every df row"


def test_groups_semantics_cell_line():
    """For unseen_cell_line, groups must encode the cell line (one label per line)."""
    df = _fake_conditions()
    s = make_splits(df, "unseen_cell_line", seed=SEED)
    groups = s["groups"]
    # rows sharing a cell line share a group; rows of different lines differ
    by_line = df.groupby("cell_line").apply(lambda g: set(groups[g.index]), include_groups=False)
    seen: set = set()
    for line, gset in by_line.items():
        assert len(gset) == 1, f"cell line {line} spans multiple group labels"
        lbl = next(iter(gset))
        assert lbl not in seen, "distinct cell lines must get distinct group labels"
        seen.add(lbl)


def test_unseen_drug_no_drug_overlap():
    """Core leakage guarantee: no drug appears in both train and test for unseen_drug."""
    df = _fake_conditions()
    s = make_splits(df, "unseen_drug", seed=SEED)
    train_drugs = set(df["treatment"].to_numpy()[s["train"]])
    test_drugs = set(df["treatment"].to_numpy()[s["test"]])
    val_drugs = set(df["treatment"].to_numpy()[s["val"]])
    assert train_drugs.isdisjoint(test_drugs), "drug leakage between train and test!"
    assert val_drugs.isdisjoint(test_drugs), "drug leakage between val and test!"


def test_unseen_drug_cluster_no_leak():
    """Butina/Tanimoto clustering: a held-out cluster must not straddle train/test."""
    df = _fake_conditions(with_smiles=True)
    s = make_splits(df, "unseen_drug", seed=SEED)
    groups = s["groups"]
    train_clusters = set(groups[s["train"]])
    test_clusters = set(groups[s["test"]])
    assert train_clusters.isdisjoint(test_clusters), "drug-cluster leakage between train/test!"


def test_rdkit_optional_fallback():
    """Without smiles the name-based fallback path still yields valid, leak-free splits."""
    df = _fake_conditions(with_smiles=False)
    s = make_splits(df, "unseen_drug", seed=SEED)
    assert len(s["test"]) > 0 and len(s["train"]) > 0
    train_drugs = set(df["treatment"].to_numpy()[s["train"]])
    test_drugs = set(df["treatment"].to_numpy()[s["test"]])
    assert train_drugs.isdisjoint(test_drugs)


def test_split_enum_accepted():
    """make_splits accepts a Split enum as well as a raw string."""
    df = _fake_conditions()
    s_enum = make_splits(df, Split.UNSEEN_DRUG, seed=SEED)
    s_str = make_splits(df, "unseen_drug", seed=SEED)
    assert np.array_equal(s_enum["test"], s_str["test"])


def test_determinism():
    df = _fake_conditions()
    a = make_splits(df, "unseen_drug", seed=SEED)
    b = make_splits(df, "unseen_drug", seed=SEED)
    assert np.array_equal(a["train"], b["train"])
    assert np.array_equal(a["test"], b["test"])


def test_bad_mode_raises():
    df = _fake_conditions()
    with pytest.raises(ValueError):
        make_splits(df, "nonsense_mode")
