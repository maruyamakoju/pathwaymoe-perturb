"""PathwayMoE-Perturb: GRN-sparsity-constrained, pathway Mixture-of-Experts model.

Condition-level model: predicts pseudobulk LFC over N genes given the cell-line baseline
expression, cell-line id, drug chemistry, target gene, and dose.

Causal factorization (each ablatable):
  - GRN-masked self-attention : half the heads may only attend along DoRothEA TF->target edges,
                                 injecting directed regulatory structure. (use_grn_mask)
  - perturbation cross-attention : the perturbation enters as a query against the cell state,
                                 never mixed into self-attention. (pert_as_token flips this)
  - Pathway top-2 MoE FFN     : experts == Reactome top-level pathways; a gene's pathway membership
                                 biases the router toward its pathway's expert. (use_moe / use_pathway_prior)

Two SDPA calls (masked heads + dense heads) implement mixed attention without Triton, so this runs
on native-Windows PyTorch on a 4090. A tiny config is used for unit tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_genes: int = 2000
    n_experts: int = 20            # == #pathways (1:1 mapping when possible)
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_pert: int = 256
    chemberta_dim: int = 384
    dropout: float = 0.1
    top_k: int = 2
    grn_head_frac: float = 0.5     # fraction of heads that are GRN-masked
    # ablation switches
    use_grn_mask: bool = True
    use_moe: bool = True
    use_pathway_prior: bool = True
    pert_as_token: bool = False    # if True, perturbation is concatenated as a token (ablation)


class MoEFFN(nn.Module):
    """Top-k pathway MoE. Each expert is a small MLP. Router logits get an additive bias from
    the per-token pathway prior so genes lean toward their own pathway's expert."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, E = cfg.d_model, cfg.n_experts
        h = d * 2
        self.w_in = nn.Parameter(torch.empty(E, d, h))
        self.w_out = nn.Parameter(torch.empty(E, h, d))
        self.b_in = nn.Parameter(torch.zeros(E, h))
        self.b_out = nn.Parameter(torch.zeros(E, d))
        for e in range(E):
            nn.init.kaiming_uniform_(self.w_in[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_out[e], a=math.sqrt(5))
        self.router = nn.Linear(d, E)

    def forward(self, x, pathway_prior=None):
        # x: (B, N, d); pathway_prior: (N, E) or None
        B, N, d = x.shape
        E, k = self.cfg.n_experts, self.cfg.top_k
        logits = self.router(x)                                   # (B,N,E)
        if pathway_prior is not None and self.cfg.use_pathway_prior:
            logits = logits + pathway_prior.unsqueeze(0)          # broadcast over batch
        if self.training:                                         # noisy top-k for load balance
            logits = logits + torch.randn_like(logits) * 1.0
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(k, dim=-1)                        # (B,N,k)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)

        # Expert dispatch: each token-slot routed to one expert; process tokens per expert with
        # a batched matmul. Avoids materializing per-token weight tensors (memory-safe at N=2000).
        xf = x.reshape(-1, d)                                     # (T, d), T=B*N
        T = xf.shape[0]
        tok_idx = torch.arange(T, device=x.device).repeat_interleave(k)   # (T*k,)
        exp_idx = topi.reshape(-1)                                # (T*k,)
        gate = topv.reshape(-1)                                   # (T*k,)
        out = torch.zeros_like(xf)
        for e in range(E):
            sel = (exp_idx == e).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            ti = tok_idx[sel]
            xe = xf[ti]                                           # (n_e, d)
            h = F.gelu(xe @ self.w_in[e] + self.b_in[e])         # (n_e, h)
            o = (h @ self.w_out[e] + self.b_out[e]) * gate[sel].unsqueeze(-1)
            out.index_add_(0, ti, o.to(out.dtype))
        out = out.reshape(B, N, d)
        importance = probs.mean(dim=(0, 1))                      # (E,)
        load = torch.zeros(E, device=x.device, dtype=x.dtype).scatter_add_(
            0, topi.reshape(-1), torch.ones(topi.numel(), device=x.device, dtype=x.dtype))
        load = load / (load.sum() + 1e-9)
        aux = E * (importance * load).sum()
        return out, aux


class MixedAttention(nn.Module):
    """Self-attention where the first `grn_head_frac` heads are restricted to GRN edges."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.n_grn = int(round(self.h * cfg.grn_head_frac)) if cfg.use_grn_mask else 0
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = cfg.dropout

    def forward(self, x, grn_bias=None):
        B, N, d = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # (B,h,N,hd)
        p = self.drop if self.training else 0.0
        if self.n_grn > 0 and grn_bias is not None:
            o_grn = F.scaled_dot_product_attention(
                q[:, :self.n_grn], k[:, :self.n_grn], v[:, :self.n_grn],
                attn_mask=grn_bias, dropout_p=p)
            o_dense = F.scaled_dot_product_attention(
                q[:, self.n_grn:], k[:, self.n_grn:], v[:, self.n_grn:], dropout_p=p)
            o = torch.cat([o_grn, o_dense], dim=1)
        else:
            o = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        o = o.transpose(1, 2).reshape(B, N, d)
        return self.proj(o)


class CrossAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = cfg.dropout

    def forward(self, x, ctx):
        B, N, d = x.shape; M = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.h, self.hd).transpose(1, 2)
        kv = self.kv(ctx).reshape(B, M, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, N, d)
        return self.proj(o)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MixedAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.cross = CrossAttention(cfg) if not cfg.pert_as_token else None
        self.ln3 = nn.LayerNorm(cfg.d_model)
        if cfg.use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.ffn = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model * 2), nn.GELU(),
                                     nn.Linear(cfg.d_model * 2, cfg.d_model))

    def forward(self, x, ctx, grn_bias, pathway_prior):
        x = x + self.attn(self.ln1(x), grn_bias)
        if self.cross is not None:
            x = x + self.cross(self.ln2(x), ctx)
        h = self.ln3(x)
        if self.cfg.use_moe:
            o, aux = self.ffn(h, pathway_prior)
            x = x + o
            return x, aux
        else:
            return x + self.ffn(h), x.new_zeros(())


class PathwayMoEPerturb(nn.Module):
    def __init__(self, cfg: ModelConfig, grn_mask=None, gene_pathway=None):
        super().__init__()
        self.cfg = cfg
        N, d = cfg.n_genes, cfg.d_model
        # accept scipy sparse or dense; work internally with dense numpy bool arrays
        if grn_mask is not None and sp.issparse(grn_mask):
            grn_mask = grn_mask.toarray()
        if grn_mask is not None:
            grn_mask = np.asarray(grn_mask).astype(bool)
        if gene_pathway is not None and sp.issparse(gene_pathway):
            gene_pathway = gene_pathway.toarray()
        if gene_pathway is not None:
            gene_pathway = np.asarray(gene_pathway).astype(bool)
        self.gene_emb = nn.Embedding(N, d)
        self.expr_proj = nn.Linear(1, d)
        self.dose_proj = nn.Linear(1, d)
        self.pert_mlp = nn.Sequential(
            nn.Linear(cfg.chemberta_dim + d + 1, cfg.d_pert), nn.GELU(),
            nn.Linear(cfg.d_pert, d))
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.head_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.register_buffer("gene_ids", torch.arange(N), persistent=False)

        # GRN attention bias (additive float mask): 0 allowed, -inf disallowed
        if grn_mask is not None and cfg.use_grn_mask:
            allow = (grn_mask | grn_mask.T)
            allow[np.diag_indices(N)] = True
            bias = np.where(allow, 0.0, float("-inf")).astype(np.float32)
            self.register_buffer("grn_bias", torch.from_numpy(bias), persistent=False)
        else:
            self.grn_bias = None

        # pathway prior over experts (N, E): map pathway membership to experts (1:1 truncated)
        if gene_pathway is not None and cfg.use_pathway_prior:
            P = gene_pathway.shape[1]
            prior = np.zeros((N, cfg.n_experts), np.float32)
            e_of_p = np.arange(P) % cfg.n_experts
            for p in range(P):
                prior[gene_pathway[:, p].astype(bool), e_of_p[p]] += 2.0
            self.register_buffer("pathway_prior", torch.from_numpy(prior), persistent=False)
        else:
            self.pathway_prior = None
        self.grad_checkpoint = False

    def forward(self, batch):
        ctrl = batch["ctrl_mean"]                                 # (B,N)
        B, N = ctrl.shape
        d = self.cfg.d_model
        tok = self.gene_emb(self.gene_ids).unsqueeze(0).expand(B, N, d)
        tok = tok + self.expr_proj(ctrl.unsqueeze(-1))
        # perturbation embedding
        tgt_emb = self.gene_emb(batch["target_idx"].clamp(min=0))
        tgt_emb = torch.where((batch["target_idx"] >= 0).unsqueeze(-1), tgt_emb, torch.zeros_like(tgt_emb))
        pert = self.pert_mlp(torch.cat([batch["chemberta"], tgt_emb,
                                        batch["dose_log"].unsqueeze(-1)], dim=-1))  # (B,d)
        if self.cfg.pert_as_token:
            tok = tok + pert.unsqueeze(1)                         # inject as additive token bias
            ctx = None
        else:
            ctx = pert.unsqueeze(1)                               # (B,1,d) for cross-attn

        aux_total = tok.new_zeros(())
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                tok, aux = torch.utils.checkpoint.checkpoint(
                    layer, tok, ctx, self.grn_bias, self.pathway_prior, use_reentrant=False)
            else:
                tok, aux = layer(tok, ctx, self.grn_bias, self.pathway_prior)
            aux_total = aux_total + aux
        pred = self.head(self.head_ln(tok)).squeeze(-1)           # (B,N)
        self._last_aux = aux_total / max(1, self.cfg.n_layers)
        return pred

    def aux_loss(self):
        return getattr(self, "_last_aux", torch.zeros((), device=self.head.weight.device))


def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    return tot


if __name__ == "__main__":
    torch.manual_seed(1337)
    N, E = 200, 8
    grn = np.random.rand(N, N) < 0.02
    gp = np.zeros((N, E + 2), bool)
    for g in range(N):
        gp[g, np.random.randint(0, E + 2)] = True
    cfg = ModelConfig(n_genes=N, n_experts=E, d_model=64, n_heads=4, n_layers=2,
                      chemberta_dim=16)
    m = PathwayMoEPerturb(cfg, grn, gp)
    B = 4
    batch = dict(ctrl_mean=torch.rand(B, N), target_idx=torch.randint(-1, N, (B,)),
                 chemberta=torch.randn(B, 16), dose_log=torch.rand(B))
    pred = m(batch)
    loss = F.mse_loss(pred, torch.randn(B, N)) + 0.1 * m.aux_loss()
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(pred).all() and np.isfinite(gnorm)
    print(f"OK forward {tuple(pred.shape)} params={count_params(m):,} "
          f"loss={loss.item():.4f} grad_norm={gnorm:.3f} aux={m.aux_loss().item():.4f}")
