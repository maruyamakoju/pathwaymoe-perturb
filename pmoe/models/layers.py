"""Core architectural components for PathwayMoE-Perturb.

This module implements the fundamental neural layers of the PMoE framework, including
GRN-constrained attention, pathway-informed Mixture-of-Experts, and soft GRN propagation.
All layers are designed for high-performance and memory efficiency at genomic scales.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from pmoe.config import ModelConfig


class GRNPropagation(nn.Module):
    r"""Soft Gene Regulatory Network (GRN) message-passing layer.

    Implements feature propagation along GRN edges using a row-normalized operator:
    $$H_{out} = \text{LayerNorm}\left( H + \sum_{k=1}^{K} \alpha_k (P^k H) \right)$$
    where $P$ is the symmetrized, row-normalized adjacency matrix of the GRN,
    and $\alpha_k$ are learnable hop-specific scalar weights.

    Attributes:
        alpha (nn.Parameter): Learnable strengths for each propagation hop.
        P (torch.Tensor): Non-persistent buffer for the propagation operator.
    """

    def __init__(self, cfg: ModelConfig, grn: Union[np.ndarray, sp.spmatrix]):
        super().__init__()
        self.cfg = cfg
        self.hops = int(cfg.grn_prop_hops)

        p_matrix = self._build_prop_operator(grn)
        self.register_buffer("P", torch.from_numpy(p_matrix), persistent=False)
        self.alpha = nn.Parameter(torch.full((self.hops,), 0.1))
        self.norm = nn.LayerNorm(cfg.d_model)

    @staticmethod
    def _build_prop_operator(grn: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        """Constructs the row-normalized propagation operator P."""
        if sp.issparse(grn):
            grn = grn.toarray()
        # M = |A|, symmetrized, zero diagonal
        adj = np.abs(np.asarray(grn, dtype=np.float32))
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0.0)
        
        # Row normalization: P[i, j] = M[i, j] / sum_j M[i, j]
        row_sums = adj.sum(axis=1, keepdims=True)
        p_operator = np.divide(adj, row_sums, out=np.zeros_like(adj), where=row_sums > 0)
        return p_operator.astype(np.float32)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Input features of shape (B, N, D).
        Returns:
            Propagated features of shape (B, N, D).
        """
        out = h
        h_k = h
        for k in range(self.hops):
            h_k = torch.einsum("ij,bjd->bid", self.P, h_k)
            out = out + self.alpha[k] * h_k
        return self.norm(out)


class MoEFFN(nn.Module):
    r"""Pathway-informed Mixture-of-Experts (MoE) Feed-Forward Network.

    Uses a top-k routing mechanism where router logits are biased by a genomic pathway prior:
    $$\text{logits} = H W_{router} + \text{pathway\_prior}$$
    Experts are implemented as small MLPs. Token dispatch is performed efficiently
    by grouping tokens per expert.

    Mathematical Intent:
    Incorporate biological domain knowledge (pathways) into the sparse expert routing
    process to improve interpretability and generalization.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, e_count = cfg.d_model, cfg.n_experts
        h_dim = d * 2
        
        # Expert parameters: (E, D, H) and (E, H, D)
        self.w_in = nn.Parameter(torch.empty(e_count, d, h_dim))
        self.w_out = nn.Parameter(torch.empty(e_count, h_dim, d))
        self.b_in = nn.Parameter(torch.zeros(e_count, h_dim))
        self.b_out = nn.Parameter(torch.zeros(e_count, d))
        
        self._reset_parameters()
        self.router = nn.Linear(d, e_count)

    def _reset_parameters(self) -> None:
        for e in range(self.cfg.n_experts):
            nn.init.kaiming_uniform_(self.w_in[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_out[e], a=math.sqrt(5))

    def forward(self, x: torch.Tensor, pathway_prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, N, D).
            pathway_prior: Optional prior bias of shape (N, E).
        Returns:
            Tuple of (output tensor, load balancing loss).
        """
        b, n, d = x.shape
        e_count, k = self.cfg.n_experts, self.cfg.top_k
        
        # Routing
        logits = self.router(x)
        if pathway_prior is not None and self.cfg.use_pathway_prior:
            logits = logits + pathway_prior.unsqueeze(0)
            
        if self.training:
            # Switch-Transformer-style routing noise; std=0.1 matches Fedus et al. 2021.
            # The pre-2026 std=1.0 was an order of magnitude larger than the literature.
            logits = logits + 0.1 * torch.randn_like(logits)
            
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(k, dim=-1)
        top_probs = top_probs / (top_probs.sum(-1, keepdim=True) + 1e-9)

        # Dispatch and Expert computation
        x_flat = x.reshape(-1, d)
        num_tokens = x_flat.shape[0]
        
        # Flattened indices for dispatch
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(k)
        expert_indices = top_indices.reshape(-1)
        gates = top_probs.reshape(-1)
        
        combined_output = torch.zeros_like(x_flat)
        
        for e in range(e_count):
            mask = (expert_indices == e)
            if not mask.any():
                continue
                
            selected_token_indices = token_indices[mask]
            expert_input = x_flat[selected_token_indices]
            
            # Expert MLP: GELU(x W_in + b_in) W_out + b_out
            hidden = F.gelu(torch.matmul(expert_input, self.w_in[e]) + self.b_in[e])
            expert_output = torch.matmul(hidden, self.w_out[e]) + self.b_out[e]
            
            # Weighted scatter
            combined_output.index_add_(0, selected_token_indices, expert_output * gates[mask].unsqueeze(-1))

        # Auxiliary load balancing loss (Switch Transformer style)
        importance = probs.mean(dim=(0, 1))
        load = torch.zeros(e_count, device=x.device, dtype=x.dtype).scatter_add_(
            0, top_indices.reshape(-1), torch.ones(top_indices.numel(), device=x.device, dtype=x.dtype)
        )
        load = load / (load.sum() + 1e-9)
        aux_loss = e_count * (importance * load).sum()
        
        return combined_output.reshape(b, n, d), aux_loss


class LatentGRNInference(nn.Module):
    r"""Variational Latent Gene Regulatory Network (LRI) Inference.

    Predicts condition-specific regulatory gates by factorizing the edge activation
    into TF-activity and Gene-receptivity latent variables:
    $$\text{Gate}_{ij} = \sigma(\text{TF\_Activity}_i + \text{Target\_Receptivity}_j)$$
    The latent variables are sampled from a Variational distribution $q(z|P)$ conditioned
    on the perturbation embedding $P$.

    Mathematical Intent:
    Moving from a static biological prior to a dynamic, state-dependent regulatory graph.
    The KL-divergence regularizes the "regulatory complexity," forcing the model to
    select the most parsimonious set of active edges for a given perturbation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, n, l_dim = cfg.d_model, cfg.n_genes, cfg.latent_grn_dim
        
        # Inference network: Perturbation -> Latent Distribution Params
        self.infer = nn.Sequential(
            nn.Linear(d, l_dim),
            nn.GELU(),
            nn.Linear(l_dim, n * 2 * 2)  # (TF_mu, TF_std, Tgt_mu, Tgt_std)
        )
        self.register_buffer("gene_ids", torch.arange(n))

    def forward(self, pert_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pert_emb: Perturbation embedding of shape (B, D).
        Returns:
            Tuple of (Gate matrix (B, N, N), KL divergence).
        """
        b, n = pert_emb.shape[0], self.cfg.n_genes
        params = self.infer(pert_emb).reshape(b, n, 2, 2)  # (B, N, TF/Tgt, Mu/Std)
        
        mu = params[..., 0]
        logvar = params[..., 1]

        # Reparameterization trick (sample only while training; use the mean at eval so
        # extracted gates are deterministic/reproducible. Sampling eps at eval made the
        # gate ~100% noise — see code/analyze_gate_degeneracy.py.)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # (B, N, 2)
        else:
            z = mu
        
        tf_act = z[..., 0].unsqueeze(-1)  # (B, N, 1)
        tgt_rec = z[..., 1].unsqueeze(1)   # (B, 1, N)
        
        # Edge Gating: σ(TF + Tgt)
        # Broadcasting produces (B, N, N)
        gate = torch.sigmoid(tf_act + tgt_rec)
        
        # KL Divergence vs N(0, 1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl / b  # Normalized by batch
        
        return gate, kl


class MixedAttention(nn.Module):
    r"""Mixed Dense/GRN-Sparsity Attention Layer.

    Applies standard self-attention to a subset of heads, while constraining the remaining
    heads to a Gene Regulatory Network (GRN) structure using an additive bias:
    $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$
    where $B$ is the `grn_bias` containing $0$ or $-\infty$ (or weighted values).

    Mathematical Intent:
    Inject inductive bias from known regulatory architectures into the Transformer's
    attention mechanism.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.num_grn_heads = int(round(self.num_heads * cfg.grn_head_frac)) if cfg.use_grn_mask else 0
        
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, grn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).
            grn_bias: Additive attention bias of shape (N, N).
        """
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        p = self.dropout_p if self.training else 0.0
        
        if self.num_grn_heads > 0 and grn_bias is not None:
            # Split heads between GRN-constrained and dense
            o_grn = F.scaled_dot_product_attention(
                q[:, :self.num_grn_heads], k[:, :self.num_grn_heads], v[:, :self.num_grn_heads],
                attn_mask=grn_bias, dropout_p=p
            )
            o_dense = F.scaled_dot_product_attention(
                q[:, self.num_grn_heads:], k[:, self.num_grn_heads:], v[:, self.num_grn_heads:], 
                dropout_p=p
            )
            output = torch.cat([o_grn, o_dense], dim=1)
        else:
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            
        output = output.transpose(1, 2).reshape(b, n, d)
        return self.proj(output)


class CrossAttention(nn.Module):
    """Perturbation Cross-Attention Layer.

    Enables gene tokens to attend to the perturbation embedding context.

    Submodule names (``q``, ``kv``, ``proj``) match the v1 checkpoint layout so the
    existing E:\\vc_project_data\\checkpoints can be loaded into this v2 model without
    a key-remap shim. Renaming these would silently break checkpoint compatibility.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Gene tokens of shape (B, N, D).
            context: Perturbation context of shape (B, M, D).
        """
        b, n, d = x.shape
        m = context.shape[1]

        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(b, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        p = self.dropout_p if self.training else 0.0
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=p)

        output = output.transpose(1, 2).reshape(b, n, d)
        return self.proj(output)
