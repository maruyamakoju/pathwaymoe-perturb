"""Tests for pmoe.priors.pathways fallback semantics.

The headline regression here is the silent ``except Exception`` that used to swallow every
error from the Reactome parser and substitute a deterministic-random pathway assignment
without telling the caller. Now the missing-file case warns; other errors propagate.
"""
from __future__ import annotations

import warnings

import pytest

from pmoe.priors import pathways as pw


def test_fallback_warns_when_reactome_absent(monkeypatch, tmp_path):
    """If the Reactome dump is absent, build_pathways must emit a UserWarning before falling
    back to the random assignment -- so a downstream MoE pathway prior is never silently
    based on fake biology.
    """
    # Make PRIORS_ROOT point at an empty tmp dir so Ensembl2Reactome_All_Levels.txt is absent.
    monkeypatch.setattr(pw, "PRIORS_ROOT", tmp_path)
    # Patch load_genes so we don't need a real dataset on E:\.
    monkeypatch.setattr(pw, "load_genes", lambda _ds: [f"G{i:05d}" for i in range(30)])
    # priors_dir builds under PRIORS_ROOT; redirect to a writable tmp area.
    out_dir = tmp_path / "priors_out"
    out_dir.mkdir()
    monkeypatch.setattr(pw, "priors_dir", lambda _ds: out_dir)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = pw.build_pathways("fake_ds_for_fallback_test", max_pathways=5)
    user_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert user_warns, "expected a UserWarning when falling back to random pathways"
    assert "Reactome" in str(user_warns[0].message)
    # Fallback still produces a usable sparse matrix.
    assert m.shape[0] == 30 and m.shape[1] > 0


def test_reactome_parse_error_propagates(monkeypatch, tmp_path):
    """A malformed Reactome file should error -- not silently downgrade to random."""
    bogus = tmp_path / "Ensembl2Reactome_All_Levels.txt"
    bogus.write_text("not\ttab\tseparated\nat\tall\n")  # too few columns

    monkeypatch.setattr(pw, "PRIORS_ROOT", tmp_path)
    monkeypatch.setattr(pw, "load_genes", lambda _ds: [f"G{i:05d}" for i in range(30)])
    out_dir = tmp_path / "priors_out"
    out_dir.mkdir()
    monkeypatch.setattr(pw, "priors_dir", lambda _ds: out_dir)

    with pytest.raises(RuntimeError, match="no Reactome pathways"):
        pw.build_pathways("fake_ds_for_parse_test", max_pathways=5)
