"""Tests for pmoe.priors.drugs fallback semantics.

The ChemBERTa fallback used to swallow every exception and silently substitute a random
projection. Now it warns loudly, and only ImportError / OSError trigger the fallback.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pmoe.priors import drugs as drugs_mod


def test_random_projection_warns_on_chemberta_oserror(monkeypatch, tmp_path):
    """If ChemBERTa weights aren't reachable (OSError), the fallback fires AND warns."""
    # Force _chemberta to look like the offline / model-not-found case.
    monkeypatch.setattr(drugs_mod, "_chemberta",
                        lambda smiles: (None, "fallback:OSError:test"))

    # Wire build_drug_feats to a tiny inline parquet so we don't hit E:\.
    pq = tmp_path / "conditions.parquet"
    pd.DataFrame({"treatment": ["d1", "d2"], "smiles": ["CCO", "c1ccccc1"],
                  "target_gene": ["G0", "G1"]}).to_parquet(pq)
    monkeypatch.setattr(drugs_mod, "conditions_path", lambda _ds: pq)
    monkeypatch.setattr(drugs_mod, "priors_dir", lambda _ds: tmp_path)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        feats = drugs_mod.build_drug_feats("fake_ds_for_drugs_test")
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("ChemBERTa unavailable" in m for m in msgs), \
        f"expected ChemBERTa fallback warning, got: {msgs}"
    assert "chemberta" in feats.columns
    # Embedding dim is CHEMBERTA_DIM (random projection).
    assert len(feats.iloc[0]["chemberta"]) == drugs_mod.CHEMBERTA_DIM


def test_morgan_fp_returns_zero_vector_for_empty_smiles():
    """An empty SMILES with rdkit present must return the zero vector (not the fallback)."""
    v = drugs_mod.morgan_fp("")
    assert v.shape == (drugs_mod.MORGAN_BITS,)
    assert v.sum() == 0.0
    assert v.dtype == np.float32
