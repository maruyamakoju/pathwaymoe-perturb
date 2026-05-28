"""Shared pytest fixtures.

The audit found that several tests under tests/test_grn_{variants,leakage}.py write
``grn_*.npz`` to ``E:\\vc_project_data\\data\\priors\\<dataset>\\``. That works on the
developer machine but errors on a CI/clean checkout where E:\\ does not exist, and it
risks racing parallel test runs. This conftest redirects ``pmoe.config.PRIORS_ROOT``
(and the derived ``priors_dir`` helper) to a session-scoped tmp directory whenever a
test touches the priors path, so the suite is self-contained.

Tests that explicitly need the real E:\\ data can opt in by marking themselves with
``@pytest.mark.needs_e_drive`` and the fixture autouse=False machinery below.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _sandbox_priors_root(tmp_path_factory, monkeypatch, request):
    """Redirect PRIORS_ROOT to a tmp dir for every test except those marked needs_e_drive.

    pmoe.config.priors_dir(dataset) is built from PRIORS_ROOT, so monkeypatching just
    the module-level constant and the helper function is enough to sandbox writes from
    build_grn / build_pathways / build_drug_feats.
    """
    if "needs_e_drive" in request.keywords:
        # The test explicitly wants the real PRIORS_ROOT (rare; we don't have any yet).
        return
    sandbox = tmp_path_factory.mktemp("vc_priors_sandbox")

    import pmoe.config as cfg
    import pmoe.priors.grn as grn_mod
    import pmoe.priors.pathways as pw_mod
    import pmoe.priors.drugs as drugs_mod

    monkeypatch.setattr(cfg, "PRIORS_ROOT", sandbox)

    def _priors_dir(dataset: str) -> Path:
        p = sandbox / dataset
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(cfg, "priors_dir", _priors_dir)
    # Modules that imported `priors_dir` / `PRIORS_ROOT` at import time use their own
    # binding; rebind those too.
    monkeypatch.setattr(grn_mod, "priors_dir", _priors_dir, raising=False)
    monkeypatch.setattr(grn_mod, "PRIORS_ROOT", sandbox, raising=False)
    monkeypatch.setattr(pw_mod, "priors_dir", _priors_dir, raising=False)
    monkeypatch.setattr(pw_mod, "PRIORS_ROOT", sandbox, raising=False)
    monkeypatch.setattr(drugs_mod, "priors_dir", _priors_dir, raising=False)
