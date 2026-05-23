r"""Self-contained tests for pmoe.data.loader.stack_arrays — no real data dependency.

The file-reading helpers (load_conditions/load_genes/load_meta) need real on-disk artifacts
under E:\, which are off-limits here, so those are not exercised. stack_arrays operates purely
on an in-memory DataFrame and is fully testable offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pmoe.data.loader import stack_arrays


def _fake_conditions(n: int = 5, n_genes: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        ctrl = rng.normal(size=n_genes).astype(np.float32)
        pert = (ctrl + rng.normal(scale=0.5, size=n_genes)).astype(np.float32)
        lfc = (pert - ctrl).astype(np.float32)
        rows.append({
            "cell_line": "A549",
            "treatment": f"drug{i}",
            "ctrl_mean": ctrl.tolist(),
            "pert_mean": pert.tolist(),
            "lfc": lfc.tolist(),
            "deg_mask": (np.abs(lfc) > 0.3).tolist(),
        })
    return pd.DataFrame(rows)


def test_stack_float_shape_and_dtype():
    n, n_genes = 5, 7
    df = _fake_conditions(n, n_genes)
    arr = stack_arrays(df, "lfc")
    assert arr.shape == (n, n_genes)
    assert arr.dtype == np.float32


def test_stack_ctrl_mean():
    n, n_genes = 4, 9
    df = _fake_conditions(n, n_genes)
    arr = stack_arrays(df, "ctrl_mean")
    assert arr.shape == (n, n_genes)
    assert arr.dtype == np.float32


def test_stack_deg_mask_is_bool():
    n, n_genes = 5, 7
    df = _fake_conditions(n, n_genes)
    arr = stack_arrays(df, "deg_mask")
    assert arr.shape == (n, n_genes)
    assert arr.dtype == np.bool_


def test_stack_values_preserved():
    df = _fake_conditions(3, 4)
    arr = stack_arrays(df, "lfc")
    expected = np.stack([np.asarray(x, np.float32) for x in df["lfc"]])
    assert np.allclose(arr, expected)


def test_stack_lfc_equals_pert_minus_ctrl():
    df = _fake_conditions(3, 6)
    ctrl = stack_arrays(df, "ctrl_mean")
    pert = stack_arrays(df, "pert_mean")
    lfc = stack_arrays(df, "lfc")
    assert np.allclose(lfc, pert - ctrl, atol=1e-5)
