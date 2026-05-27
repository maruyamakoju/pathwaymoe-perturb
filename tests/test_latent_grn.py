import torch
from pmoe.config import ModelConfig, ArchitectureConfig, GRNConfig, TrainConfig
from pmoe.models.pathway_moe import PathwayMoEPerturb
import numpy as np

def test_latent_grn_forward():
    print("Testing Latent GRN forward pass...")
    cfg = ModelConfig(
        arch=ArchitectureConfig(n_genes=100, d_model=64, n_heads=4, n_layers=2),
        grn=GRNConfig(latent_grn=True, latent_grn_dim=32),
    )
    
    # Mock GRN (100x100)
    grn = np.random.randint(0, 2, (100, 100))
    # Mock Pathways (100x5)
    gp = np.random.randint(0, 2, (100, 5))
    
    model = PathwayMoEPerturb(cfg, grn=grn, gene_pathway=gp)
    model.eval()
    
    batch = {
        "ctrl_mean": torch.randn(4, 100),
        "target_idx": torch.randint(0, 100, (4,)),
        "chemberta": torch.randn(4, 384),
        "dose_log": torch.randn(4,)
    }
    
    with torch.no_grad():
        out = model(**batch)
        kl = model.kl_loss()
    
    print(f"Output shape: {out.shape}")
    print(f"KL Loss: {kl.item():.4f}")
    assert out.shape == (4, 100)
    assert kl.item() >= 0
    print("Latent GRN forward pass SUCCESSFUL.")

if __name__ == "__main__":
    test_latent_grn_forward()
