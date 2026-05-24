"""Single canonical model loader for evaluation (audit I — dedup of ``model_predict``).

v1 duplicated ``model_predict`` across ``eval.py``, ``eval_ablations.py`` and ``run_study.py``;
one copy unconditionally reloaded ``grn_mask.npz`` instead of the variant/split the model was
trained with. Because ``grn_bias`` is a NON-PERSISTENT buffer (rebuilt at load, not stored in the
state dict), loading the wrong GRN silently gives the wrong attention structure at eval time. This
module is the one place that reconstructs the model and reloads the CORRECT GRN.

``pmoe.models`` / ``pmoe.priors`` are built in parallel, so they are imported *inside* the
functions to keep this module importable on its own.
"""
from __future__ import annotations

import numpy as np
import torch

from pmoe.config import ModelConfig, RunSpec
from pmoe.io import load_checkpoint


def load_model_for_eval(run: RunSpec, dataset: str, variant, split, device):
    """Reconstruct the trained model from its checkpoint with the CORRECT GRN reloaded.

    Reconstructs ``ModelConfig`` from the checkpoint, loads the matching pathway prior, reloads
    the ``variant``/``split``-specific GRN via :func:`pmoe.priors.grn.load_grn` (the fix for the
    non-persistent ``grn_bias`` buffer bug), then loads the state dict in eval mode.
    """
    from pmoe.models import PathwayMoEPerturb
    from pmoe.priors.grn import load_grn
    from pmoe.priors.pathways import load_pathways

    ck = load_checkpoint(run.ckpt_path, map_location=device)
    cfg = ModelConfig(**ck["cfg"])

    # Reload the CORRECT GRN variant for this split (None for the 'none' variant / missing file).
    grn = load_grn(dataset, variant, split)

    # Pathway prior is variant-independent; tolerate absence.
    try:
        gene_pathway, _names = load_pathways(dataset)
    except Exception:
        gene_pathway = None

    model = PathwayMoEPerturb(cfg, grn, gene_pathway).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def predict_test(run: RunSpec, dataset: str, df, shared: dict, test_idx: np.ndarray,
                 variant, split, device) -> np.ndarray:
    """Run the trained model over the test conditions and return predicted LFC (n_test, N)."""
    from pmoe.data.dataset import make_loader

    model = load_model_for_eval(run, dataset, variant, split, device)
    loader = make_loader(dataset, test_idx, df, shared, batch_size=16, shuffle=False)
    # fp32 (no bf16 autocast) for DETERMINISTIC, reproducible eval metrics. bf16 autocast made the
    # point estimate run-to-run unstable on near-zero-variance tasks (per-condition DEG-Pearson ~0).
    preds = []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            preds.append(model(b).float().cpu().numpy())
    return np.concatenate(preds)
