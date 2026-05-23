"""Core layers for PathwayMoE-Perturb.

Ported from v1 ``code/model.py`` (the working, memory-safe versions), with one extension:
:class:`MixedAttention` now accepts a **weighted** GRN attention bias (float edge weights), in
addition to the binary 0/-inf mask, so the model can exploit signed/weighted GRNs (audit item F).

Two SDPA calls (GRN-masked heads + dense heads) implement mixed attention without Triton, so this
runs on native-Windows PyTorch. Tiny configs are used for unit tests.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pmoe.config import ModelConfig


class MoEFFN(nn.Module):
    """Top-k pathway MoE. Each expert is a small MLP. Router logits get an additive bias from
    the per-token pathway prior so genes lean toward their own pathway's expert.

    This is the **efficient expert-dispatch** implementation: tokens are gathered per expert and
    processed with a single batched matmul per expert, then scattered back with ``index_add_``.
    This avoids materializing per-token weight tensors (memory-safe at N=2000).
    """

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

    def forward(self, x: torch.Tensor, pathway_prior: torch.Tensor | None = None):
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
            h = F.gelu(xe @ self.w_in[e] + self.b_in[e])          # (n_e, h)
            o = (h @ self.w_out[e] + self.b_out[e]) * gate[sel].unsqueeze(-1)
            out.index_add_(0, ti, o.to(out.dtype))
        out = out.reshape(B, N, d)
        importance = probs.mean(dim=(0, 1))                       # (E,)
        load = torch.zeros(E, device=x.device, dtype=x.dtype).scatter_add_(
            0, topi.reshape(-1), torch.ones(topi.numel(), device=x.device, dtype=x.dtype))
        load = load / (load.sum() + 1e-9)
        aux = E * (importance * load).sum()
        return out, aux


class MixedAttention(nn.Module):
    """Self-attention where the first ``grn_head_frac`` heads are restricted to GRN edges.

    The GRN-restricted heads use an additive attention bias ``grn_bias`` of shape (N, N) that is
    broadcast across batch/heads. Two flavors are supported:

    * **Binary mask** (default): ``grn_bias`` is 0 for allowed edges and ``-inf`` for absent edges.
    * **Weighted bias** (audit item F): ``grn_bias`` is a finite, non-negative bias proportional to
      edge weight on allowed edges and ``-inf`` on absent edges. The bias is built upstream in
      :class:`~pmoe.models.pathway_moe.PathwayMoEPerturb`; this layer is agnostic to which flavor it
      receives -- it simply adds it inside SDPA.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.n_grn = int(round(self.h * cfg.grn_head_frac)) if cfg.use_grn_mask else 0
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = cfg.dropout

    def forward(self, x: torch.Tensor, grn_bias: torch.Tensor | None = None):
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
    """Cross-attention: gene tokens query the perturbation context (drug/target/dose)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = cfg.dropout

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        B, N, d = x.shape
        M = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.h, self.hd).transpose(1, 2)
        kv = self.kv(ctx).reshape(B, M, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, N, d)
        return self.proj(o)
