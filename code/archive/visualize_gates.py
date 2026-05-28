"""Interpretability Audit: Visualizing Learned Regulatory Gates.

This script extracts the condition-specific GRN gates learned by the 
Latent Regulatory Inference (LRI) model for a given drug.
"""
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from pmoe.config import CKPT_DIR, DATA_ROOT, ModelConfig, ArchitectureConfig, PerturbationConfig, GRNConfig, MoEConfig
from pmoe.data.loader import load_genes, load_conditions
from pmoe.data.dataset import build_shared
from pmoe.models.pathway_moe import PathwayMoEPerturb

def dict_to_cfg(d: dict) -> ModelConfig:
    """Reconstructs ModelConfig from nested dict."""
    return ModelConfig(
        arch=ArchitectureConfig(**d["arch"]),
        pert=PerturbationConfig(**d["pert"]),
        grn=GRNConfig(**d["grn"]),
        moe=MoEConfig(**d["moe"])
    )

def visualize_gates(run_name: str, drug_name: str):
    print(f"Visualizing gates for {drug_name} in run {run_name}...")
    
    # 1. Load Checkpoint
    ckpt_path = CKPT_DIR / f"{run_name}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return
        
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict_to_cfg(ckpt["cfg"])
    
    # 2. Reconstruct Model
    dataset = ckpt.get("extra", {}).get("dataset", "synthetic_hard_v2")
    genes = load_genes(dataset)
    model = PathwayMoEPerturb(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # 3. Get Drug Features via build_shared
    df = load_conditions(dataset)
    shared = build_shared(dataset, df)
    
    # Find the first index for this drug
    drug_indices = df[df["treatment"] == drug_name].index
    if len(drug_indices) == 0:
        print(f"Drug {drug_name} not found in dataset {dataset}.")
        return
    idx = drug_indices[0]
    
    chemberta = torch.from_numpy(shared["chemberta"][idx]).unsqueeze(0)
    target_idx = torch.tensor([shared["target_idx"][idx]])
    dose_log = torch.tensor([shared["dose_log"][idx]])
    
    # 4. Infer Gates
    with torch.no_grad():
        target_emb = model.gene_emb(target_idx.clamp(min=0))
        pert_features = torch.cat([chemberta, target_emb, dose_log.unsqueeze(-1)], dim=-1)
        pert_emb = model.pert_mlp(pert_features)
        
        gate, _ = model.lri(pert_emb)
        # gate is (B, N, N)
        gate = gate.squeeze(0).numpy() # (N, N)
        
    # 5. Extract Top Regulatory Paths
    n = len(genes)
    edges = []
    for i in range(n):
        for j in range(n):
            if gate[i, j] > 0.8: 
                edges.append((genes[i], genes[j], gate[i, j]))
                
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    
    print(f"\nTop Learned Edges for {drug_name}:")
    for src, tgt, score in edges[:20]:
        print(f"  {src} -> {tgt}: {score:.4f}")
        
    # 6. Save
    out_dir = Path("results/interpretability")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "drug": drug_name,
        "edges": [{"source": s, "target": t, "score": float(sc)} for s, t, sc in edges[:100]]
    }
    
    with open(out_dir / f"{drug_name}_gates.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / f'{drug_name}_gates.json'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        visualize_gates(sys.argv[1], sys.argv[2])
