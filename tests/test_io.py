"""Tests for pmoe.io: checkpoint roundtrip + manifest.

The audit-I regression class is the non-persistent ``grn_bias`` buffer: a model trained
with GRN variant X stores its weights in the state_dict but NOT its grn_bias (rebuilt at
load from ``load_grn(dataset, variant, split)``). If the loader reloads the wrong GRN, the
reloaded model silently has different attention structure -- the published metric on disk
no longer matches what a re-eval would produce. This module is where we lock that down.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from pmoe.config import ModelConfig, RunSpec
from pmoe.experiments.train import make_hierarchical_config
from pmoe.io import data_manifest, env_manifest, load_checkpoint, save_checkpoint
from pmoe.models.pathway_moe import PathwayMoEPerturb


def _tiny_cfg() -> ModelConfig:
    return make_hierarchical_config(
        dataset="synthetic", size="tiny", variant="trrust",
        n_genes=24, n_experts=4, cb_dim=8,
    )


def _tiny_grn(n: int, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < 0.1
    return sp.csr_matrix(mask, dtype=bool)


def _tiny_batch(cfg: ModelConfig, b: int = 3) -> dict:
    return dict(
        ctrl_mean=torch.rand(b, cfg.n_genes),
        target_idx=torch.randint(0, cfg.n_genes, (b,)),
        chemberta=torch.randn(b, cfg.chemberta_dim),
        dose_log=torch.rand(b),
    )


def test_checkpoint_roundtrip_preserves_predictions(tmp_path, monkeypatch):
    """Train one step on a tiny model, save, reload, assert identical predictions.

    Closes audit-I: a future regression that makes ``grn_bias`` accidentally persistent
    (and thus loaded twice or with stale shapes), or that rebuilds the model with a
    different non-persistent buffer, would change the reloaded prediction and trip this.
    """
    torch.manual_seed(7)
    cfg = _tiny_cfg()
    grn = _tiny_grn(cfg.n_genes, seed=1)

    # The model we save: train one step so weights are non-trivial.
    model = PathwayMoEPerturb(cfg, grn).train()
    batch = _tiny_batch(cfg)
    pred = model(**batch)
    loss = pred.pow(2).mean() + 0.1 * model.aux_loss()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.data -= 0.01 * p.grad

    # Reference predictions in eval mode.
    model.eval()
    with torch.no_grad():
        ref_pred = model(**batch)

    # Save -> reload via the same code path the study uses.
    run = RunSpec(dataset="ckpt_roundtrip", split="unseen_drug",
                  variant="trrust", seed=42, size="tiny")
    monkeypatch.setattr(type(run), "ckpt_path",
                        property(lambda self: tmp_path / f"{self.name}.pt"))
    save_checkpoint(run, model.state_dict(), cfg, extra={"test": True})

    ck = load_checkpoint(run.ckpt_path)
    reloaded = PathwayMoEPerturb(cfg, grn).eval()
    reloaded.load_state_dict(ck["model"])
    with torch.no_grad():
        rel_pred = reloaded(**batch)

    assert torch.allclose(ref_pred, rel_pred, atol=1e-6), \
        "reloaded checkpoint must produce bit-identical predictions"


def test_env_manifest_records_versions():
    m = env_manifest()
    assert "python" in m and "platform" in m
    assert "packages" in m
    # Torch is a hard dependency; its version string must be present.
    assert m["packages"].get("torch")


def test_data_manifest_handles_missing_dataset(tmp_path, monkeypatch):
    """data_manifest is best-effort: a non-existent dataset shouldn't crash."""
    # Force conditions_path / meta_path to point at a missing tmp dir.
    import pmoe.io as io_mod
    monkeypatch.setattr(io_mod, "meta_path", lambda _ds: tmp_path / "missing_meta.json")
    monkeypatch.setattr(io_mod, "genes_path", lambda _ds: tmp_path / "missing_genes.txt")
    m = data_manifest("no_such_dataset")
    assert m["dataset"] == "no_such_dataset"
    assert m["genes_sha1"] is None
