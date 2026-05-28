"""Data subpackage: loaders, leakage-controlled splits, the condition dataset/collator,
and dataset builders (synthetic ground-truth + streaming Tahoe-100M preprocessor)."""
from __future__ import annotations

from pmoe.data.loader import load_conditions, load_genes, load_meta, stack_arrays
from pmoe.data.splits import make_splits
from pmoe.data import synth, preprocess_tahoe

__all__ = [
    "load_conditions",
    "load_genes",
    "load_meta",
    "stack_arrays",
    "make_splits",
    "synth",
    "preprocess_tahoe",
]
