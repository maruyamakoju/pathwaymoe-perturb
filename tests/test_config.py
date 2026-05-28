"""Tests for pmoe.config naming conventions.

The headline audit-A guarantee is that data-derived GRNs are split-specific (so the v1
all-conditions leakage can't recur). The file-naming convention encodes that on disk;
this test locks the convention in.
"""
from __future__ import annotations

import pytest

from pmoe.config import GRNVariant, Split, grn_filename


def test_data_derived_grn_filename_includes_split():
    assert grn_filename(GRNVariant.COEXPR, Split.UNSEEN_DRUG) == "grn_coexpr__unseen_drug.npz"
    assert grn_filename(GRNVariant.COEXPR_LFC, "unseen_cell_line") == "grn_coexpr_lfc__unseen_cell_line.npz"
    assert grn_filename("coexpr", "unseen_both") == "grn_coexpr__unseen_both.npz"


def test_non_data_derived_grn_filename_omits_split():
    """trrust, trrust_weighted, random, ground_truth share the gene space across splits,
    so they live at a single per-dataset path -- not per-split."""
    assert grn_filename(GRNVariant.TRRUST, "unseen_drug") == "grn_trrust.npz"
    assert grn_filename(GRNVariant.TRRUST_WEIGHTED, None) == "grn_trrust_weighted.npz"
    assert grn_filename(GRNVariant.RANDOM, "unseen_drug") == "grn_random.npz"
    assert grn_filename(GRNVariant.GROUND_TRUTH, "unseen_both") == "grn_ground_truth.npz"


def test_none_grn_filename_is_empty():
    assert grn_filename(GRNVariant.NONE) == ""
    assert grn_filename("none", "unseen_drug") == ""


def test_grn_filename_accepts_str_and_enum():
    """Both bare-str and enum inputs should resolve to the same canonical filename."""
    assert grn_filename("trrust") == grn_filename(GRNVariant.TRRUST)
    assert grn_filename("coexpr_lfc", "unseen_drug") == \
        grn_filename(GRNVariant.COEXPR_LFC, Split.UNSEEN_DRUG)
