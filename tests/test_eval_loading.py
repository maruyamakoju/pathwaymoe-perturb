"""Tests for pmoe.eval.loading: the single canonical checkpoint loader.

The headline regression class here (audit-I) is that wrong-shape or wrong-key state_dicts
should NOT silently load against a model with hardcoded defaults. We exercise the raised
error path; an end-to-end checkpoint roundtrip lives in tests/test_io.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pmoe.config import RunSpec, SEED
from pmoe.eval.loading import _remap_state_dict, load_model_for_eval


def test_remap_state_dict_renames_cross_keys():
    """v2-window checkpoints stored q_proj/kv_proj/out_proj; current model uses v1 names."""
    src = {
        "layers.0.cross.q_proj.weight": torch.zeros(2, 2),
        "layers.0.cross.kv_proj.bias": torch.zeros(4),
        "layers.0.cross.out_proj.weight": torch.zeros(2, 2),
        "layers.0.attn.qkv.weight": torch.zeros(6, 2),
        "gene_emb.weight": torch.zeros(3, 2),
    }
    out = _remap_state_dict(src)
    assert "layers.0.cross.q.weight" in out
    assert "layers.0.cross.kv.bias" in out
    assert "layers.0.cross.proj.weight" in out
    # non-cross keys must be untouched
    assert "layers.0.attn.qkv.weight" in out
    assert "gene_emb.weight" in out
    assert len(out) == len(src)


def test_remap_state_dict_is_identity_for_v1_names():
    src = {"layers.0.cross.q.weight": torch.zeros(2, 2),
           "layers.0.cross.kv.weight": torch.zeros(4, 2),
           "layers.0.cross.proj.weight": torch.zeros(2, 2)}
    out = _remap_state_dict(src)
    assert set(out.keys()) == set(src.keys())


def test_load_model_for_eval_raises_on_missing_cfg_fields(tmp_path: Path, monkeypatch):
    """Audit-I regression: a v1 flat cfg without shape keys must RAISE, not silently default.

    Build a fake checkpoint with a flat-cfg dict that lacks n_genes/n_experts/chemberta_dim,
    point RunSpec.ckpt_path at it, and assert load_model_for_eval raises KeyError naming the
    missing keys.
    """
    bogus_path = tmp_path / "bogus.pt"
    torch.save({"model": {}, "cfg": {"use_grn_mask": True},  # flat, no n_genes etc.
                "run": {}, "extra": {}, "env": {}}, bogus_path)

    run = RunSpec(dataset="fake_ds", split="unseen_drug", variant="none", seed=SEED, size="tiny")
    # monkey-patch ckpt_path to point at the bogus file (RunSpec.ckpt_path is a property)
    monkeypatch.setattr(type(run), "ckpt_path", property(lambda self: bogus_path))

    with pytest.raises(KeyError, match="missing required shape keys"):
        load_model_for_eval(run, "fake_ds", "none", "unseen_drug", "cpu")
