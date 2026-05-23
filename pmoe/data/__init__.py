"""Data subpackage: loaders, leakage-controlled splits, and the condition dataset/collator."""
from __future__ import annotations

from pmoe.data.loader import load_conditions, load_genes, load_meta, stack_arrays
from pmoe.data.splits import make_splits

__all__ = [
    "load_conditions",
    "load_genes",
    "load_meta",
    "stack_arrays",
    "make_splits",
]
