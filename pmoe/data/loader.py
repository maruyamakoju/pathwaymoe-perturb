"""Load processed conditions / genes / meta and stack list-columns into dense arrays.

Ported from v1 ``code/loader.py``; only the import paths change (``pmoe.config`` instead
of the flat ``config`` module). Numerics are unchanged.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pmoe.config import conditions_path, genes_path, meta_path


def load_meta(dataset: str) -> dict:
    """Read ``meta.json`` for a dataset (cell lines, drugs, N_GENES, ...)."""
    return json.loads(meta_path(dataset).read_text())


def load_genes(dataset: str) -> list[str]:
    """Read ``genes.txt`` -> list of HGNC symbols; list index == array column index."""
    return genes_path(dataset).read_text().splitlines()


def load_conditions(dataset: str) -> pd.DataFrame:
    """Read ``conditions.parquet``.

    List columns come back as numpy object arrays of lists; leave them as-is and use
    :func:`stack_arrays` to densify a given column on demand.
    """
    return pd.read_parquet(conditions_path(dataset))


def stack_arrays(df: pd.DataFrame, col: str) -> np.ndarray:
    """Stack a list-column into a dense ``(n_conditions, N)`` array.

    Returns ``bool`` for ``deg_mask``, ``float32`` otherwise.
    """
    arr = np.stack([np.asarray(x) for x in df[col].to_numpy()])
    if col == "deg_mask":
        return arr.astype(bool)
    return arr.astype(np.float32)
