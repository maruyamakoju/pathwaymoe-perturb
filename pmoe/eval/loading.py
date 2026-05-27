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
    """Reconstruct the trained model from its checkpoint with the CORRECT GRN reloaded."""
    from pmoe.models import PathwayMoEPerturb
    from pmoe.priors.grn import load_grn
    from pmoe.priors.pathways import load_pathways
    from pmoe.experiments.train import make_hierarchical_config

    ck = load_checkpoint(run.ckpt_path, map_location=device)
    raw_cfg = ck["cfg"]
    
    # Handle both old flat configs and new hierarchical configs
    if "arch" in raw_cfg:
        # New hierarchical config
        from pmoe.config import ArchitectureConfig, PerturbationConfig, GRNConfig, MoEConfig
        cfg = ModelConfig(
            arch=ArchitectureConfig(**raw_cfg["arch"]),
            pert=PerturbationConfig(**raw_cfg["pert"]),
            grn=GRNConfig(**raw_cfg["grn"]),
            moe=MoEConfig(**raw_cfg["moe"])
        )
    else:
        # Old flat config - reconstruct using the helper
        # We need cb_dim which might be in shared or extra. 
        # For eval loading, we just need to reconstruct the ModelConfig object.
        cfg = make_hierarchical_config(
            dataset=dataset,
            size=run.size,
            variant=variant,
            n_genes=raw_cfg.get("n_genes", 2000),
            n_experts=raw_cfg.get("n_experts", 40),
            cb_dim=raw_cfg.get("chemberta_dim", 384),
            grn_propagation=raw_cfg.get("grn_propagation", False),
            use_grn_mask=raw_cfg.get("use_grn_mask", True)
        )

    grn = load_grn(dataset, variant, split)

    try:
        gene_pathway, _ = load_pathways(dataset)
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
    
    preds = []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # Use **b to match the new forward signature
            preds.append(model(**b).float().cpu().numpy())
    return np.concatenate(preds)
