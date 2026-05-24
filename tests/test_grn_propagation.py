"""CPU-only unit tests for the SOFT GRN message-passing mechanism (GRNPropagation).

Covers: forward shape + finiteness, finite grads after backward, that turning propagation
on actually changes the prediction (same weights/seed), and that the propagation operator P
is row-normalized (rows sum to ~1, or 0 for genes with no edges).
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from pmoe.config import ModelConfig
from pmoe.models import PathwayMoEPerturb
from pmoe.models.layers import GRNPropagation


def _cfg(**kw) -> ModelConfig:
    base = dict(n_genes=40, d_model=32, n_layers=2, n_experts=4, n_heads=4, chemberta_dim=16,
                d_pert=32)
    base.update(kw)
    return ModelConfig(**base)


def _rand_batch(cfg: ModelConfig, B: int = 4) -> dict:
    N = cfg.n_genes
    return dict(
        ctrl_mean=torch.rand(B, N),
        target_idx=torch.randint(-1, N, (B,)),
        chemberta=torch.randn(B, cfg.chemberta_dim),
        dose_log=torch.rand(B),
    )


def _sparse_grn(N=40, seed=3, density=0.08):
    rng = np.random.default_rng(seed)
    mask = rng.random((N, N)) < density
    w = rng.uniform(-1.0, 1.0, size=(N, N)).astype(np.float32) * mask
    return sp.csr_matrix(w)


def test_propagation_forward_shape_and_finite_grads():
    torch.manual_seed(0)
    cfg = _cfg(grn_propagation=True)
    model = PathwayMoEPerturb(cfg, _sparse_grn(cfg.n_genes)).train()
    assert model.grn_prop is not None
    batch = _rand_batch(cfg, B=5)
    pred = model(batch)
    assert pred.shape == (5, 40)
    assert torch.isfinite(pred).all()

    loss = F.mse_loss(pred, torch.randn_like(pred)) + 0.1 * model.aux_loss()
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert np.isfinite(gnorm) and gnorm > 0
    # the learnable per-hop scalars must receive gradient
    assert model.grn_prop.alpha.grad is not None
    assert torch.isfinite(model.grn_prop.alpha.grad).all()


def test_propagation_changes_prediction():
    """With identical weights/seed/input, grn_propagation=True must differ from =False,
    proving the propagation actually changes the prediction."""
    grn = _sparse_grn(40)

    torch.manual_seed(123)
    m_off = PathwayMoEPerturb(_cfg(grn_propagation=False), grn).eval()
    torch.manual_seed(123)
    m_on = PathwayMoEPerturb(_cfg(grn_propagation=True), grn).eval()

    # copy over all shared parameters so the only difference is the propagation layer
    off_state = m_off.state_dict()
    m_on.load_state_dict(off_state, strict=False)
    assert m_on.grn_prop is not None and m_off.grn_prop is None

    batch = _rand_batch(_cfg(), B=4)
    with torch.no_grad():
        out_off = m_off(batch)
        out_on = m_on(batch)
    assert out_on.shape == out_off.shape
    assert torch.isfinite(out_on).all()
    assert not torch.allclose(out_on, out_off)


def test_propagation_operator_row_normalized():
    grn = _sparse_grn(40)
    prop = GRNPropagation(_cfg(grn_propagation=True), grn)
    P = prop.P.numpy()
    assert P.shape == (40, 40)
    # P built from |A| symmetrized with zero diagonal -> non-negative, zero diagonal
    assert (P >= 0).all()
    assert np.allclose(np.diag(P), 0.0)
    rowsum = P.sum(axis=1)
    # each row sums to ~1 (has edges) or ~0 (no edges)
    near_one = np.isclose(rowsum, 1.0, atol=1e-5)
    near_zero = np.isclose(rowsum, 0.0, atol=1e-5)
    assert (near_one | near_zero).all()
    assert near_one.any()   # the random GRN should produce at least some connected rows


def test_propagation_buffer_non_persistent():
    """P (and grn_bias) must be non-persistent so they are rebuilt at load, not stored."""
    cfg = _cfg(grn_propagation=True)
    model = PathwayMoEPerturb(cfg, _sparse_grn(cfg.n_genes))
    sd = model.state_dict()
    assert not any(k.endswith("grn_prop.P") for k in sd)
