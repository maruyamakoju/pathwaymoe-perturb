"""Load processed conditions / genes / meta and stack list-columns into dense arrays."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from config import conditions_path, genes_path, meta_path


def load_meta(dataset: str) -> dict:
    return json.loads(meta_path(dataset).read_text())


def load_genes(dataset: str) -> list[str]:
    return genes_path(dataset).read_text().splitlines()


def load_conditions(dataset: str) -> pd.DataFrame:
    """Read conditions.parquet. List columns come back as numpy object arrays of lists;
    leave them as-is and use stack_arrays() to densify a given column."""
    return pd.read_parquet(conditions_path(dataset))


def stack_arrays(df: pd.DataFrame, col: str) -> np.ndarray:
    """(n_conditions, N) float32 (or bool for deg_mask)."""
    arr = np.stack([np.asarray(x) for x in df[col].to_numpy()])
    if col == "deg_mask":
        return arr.astype(bool)
    return arr.astype(np.float32)
