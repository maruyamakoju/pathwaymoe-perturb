"""CPU-only unit tests for pmoe.models (tiny tensors)."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from pmoe.config import ModelConfig
from pmoe.models import MixedAttention, MoEFFN, PathwayMoEPerturb, count_params
from pmoe.experiments.train import make_hierarchical_config


def _tiny_cfg(**kw) -> ModelConfig:
    # Use the helper to build a hierarchical config
    base = dict(
        dataset="synthetic",
        size="tiny",
        variant="none",
        n_genes=60,
        n_experts=6,
        cb_dim=16,
    )
    base.update(kw)
    return make_hierarchical_config(**base)


def _rand_batch(cfg: ModelConfig, B: int = 4) -> dict:
    N = cfg.n_genes
    return dict(
        ctrl_mean=torch.rand(B, N),
        target_idx=torch.randint(-1, N, (B,)),
        chemberta=torch.randn(B, cfg.chemberta_dim),
        dose_log=torch.rand(B),
    )


def _bool_grn(N, seed=0, density=0.05):
    rng = np.random.default_rng(seed)
    return rng.random((N, N)) < density


def _float_grn(N, seed=0, density=0.05):
    rng = np.random.default_rng(seed)
    mask = rng.random((N, N)) < density
    w = rng.uniform(-1.0, 1.0, size=(N, N)).astype(np.float32) * mask
    return sp.csr_matrix(w)


def test_forward_shape_and_finite_grads():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    N = cfg.n_genes
    gp = np.zeros((N, cfg.n_experts + 2), bool)
    rng = np.random.default_rng(1)
    for g in range(N):
        gp[g, rng.integers(0, cfg.n_experts + 2)] = True
    model = PathwayMoEPerturb(cfg, sp.csr_matrix(_bool_grn(N)), gp).train()
    batch = _rand_batch(cfg, B=5)
    pred = model(**batch)
    assert pred.shape == (5, 60)
    assert torch.isfinite(pred).all()

    loss = F.mse_loss(pred, torch.randn_like(pred)) + 0.1 * model.aux_loss()
    assert torch.isfinite(loss)
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert np.isfinite(gnorm) and gnorm > 0
    assert count_params(model) > 0


def test_aux_loss_positive_in_train_mode():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = PathwayMoEPerturb(cfg, _bool_grn(cfg.n_genes)).train()
    _ = model(**_rand_batch(cfg))
    assert float(model.aux_loss()) > 0.0


def test_weighted_and_binary_grn_both_run():
    torch.manual_seed(0)
    N = 60
    # binary (bool) GRN, variant="trrust" will trigger use_grn_mask=True in helper
    cfg_b = _tiny_cfg(variant="trrust")
    m_b = PathwayMoEPerturb(cfg_b, _bool_grn(N)).eval()
    out_b = m_b(**_rand_batch(cfg_b))
    assert out_b.shape == (4, 60) and torch.isfinite(out_b).all()

    # weighted (float scipy sparse) GRN
    cfg_w = _tiny_cfg(variant="trrust_weighted")
    m_w = PathwayMoEPerturb(cfg_w, _float_grn(N)).eval()
    # the weighted bias must be finite on allowed edges, -inf only on absent ones
    gb = m_w.grn_bias.numpy()
    finite = np.isfinite(gb)
    assert finite.any() and (gb[finite] >= 0).all()           # non-negative finite bias
    assert np.isneginf(gb).any()                              # absent edges masked
    out_w = m_w(**_rand_batch(cfg_w))
    assert out_w.shape == (4, 60) and torch.isfinite(out_w).all()


def test_moeffn_routes_changes_input():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    ffn = MoEFFN(cfg).eval()
    x = torch.randn(2, cfg.n_genes, cfg.d_model)
    out, aux = ffn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, x)                          # MoE actually transforms tokens
    assert float(aux) > 0


def test_lri_eval_determinism():
    """LatentGRNInference must be deterministic in eval mode.

    Regression for the code/archive/analyze_gate_degeneracy.py finding: the v1 LRI sampled eps from
    its variational posterior even with model.eval(), so two forward passes on the same
    batch produced different gates -- which made the "high-confidence regulatory edges"
    in the original LRI episode pure sampling noise. The fix in pmoe/models/layers.py uses
    mu (not mu + eps*std) at eval; this test would fail if that fix is ever reverted.
    """
    torch.manual_seed(7)
    cfg = _tiny_cfg(latent_grn=True)
    model = PathwayMoEPerturb(cfg, _bool_grn(cfg.n_genes)).eval()
    batch = _rand_batch(cfg, B=3)
    with torch.no_grad():
        o1 = model(**batch)
        o2 = model(**batch)
    assert torch.equal(o1, o2), "LRI eval forward must be deterministic"


def test_mixed_attention_weighted_bias_runs():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    attn = MixedAttention(cfg).eval()
    N = cfg.n_genes
    x = torch.randn(3, N, cfg.d_model)
    # a weighted-style finite/-inf bias
    bias = torch.full((N, N), float("-inf"))
    bias.fill_diagonal_(0.0)
    bias[0, 1] = 1.5
    out = attn(x, bias)
    assert out.shape == x.shape and torch.isfinite(out).all()
