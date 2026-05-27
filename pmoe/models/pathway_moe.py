"""PathwayMoE-Perturb: A Sparsity-Constrained Regulatory Model.

This module implements the core PathwayMoEPerturb model, which integrates Gene Regulatory 
Networks (GRNs) and pathway Mixture-of-Experts (MoE) to predict cellular responses 
to perturbations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pmoe.config import ModelConfig
from pmoe.models.layers import CrossAttention, GRNPropagation, MixedAttention, MoEFFN


def _ensure_dense(arr: Optional[Union[np.ndarray, sp.spmatrix]]) -> Optional[np.ndarray]:
    """Ensures the input is a dense numpy array."""
    if arr is None:
        return None
    if sp.issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


class DecoderLayer(nn.Module):
    """A single Transformer decoder layer with GRN-masked and Cross-attention.

    Attributes:
        cfg: Model configuration.
        ln1, ln2, ln3: Layer normalization modules.
        attn: Mixed Dense/GRN attention.
        cross: Optional cross-attention for perturbation context.
        ffn: MoE or standard FFN.
    """

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
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 2),
                nn.GELU(),
                nn.Linear(cfg.d_model * 2, cfg.d_model)
            )

    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor], 
        grn_bias: Optional[torch.Tensor], 
        pathway_prior: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, N, D).
            context: Perturbation context (B, M, D).
            grn_bias: GRN attention bias (N, N).
            pathway_prior: MoE router prior (N, E).
        """
        # Self-attention (Mixed)
        x = x + self.attn(self.ln1(x), grn_bias)
        
        # Cross-attention (Perturbation)
        if self.cross is not None and context is not None:
            x = x + self.cross(self.ln2(x), context)
        
        # FFN (MoE or Dense)
        h = self.ln3(x)
        if self.cfg.use_moe:
            out, aux_loss = self.ffn(h, pathway_prior)
            return x + out, aux_loss
            
        return x + self.ffn(h), x.new_zeros(())


class PathwayMoEPerturb(nn.Module):
    r"""The PMoE Model for Perturbation Prediction.

    Integrates multiple inductive biases:
    1.  **GRN-Sparsity**: Restricts attention heads to known regulatory edges.
    2.  **Pathway-MoE**: Biases expert routing using gene pathway membership.
    3.  **Cross-Attention**: Decouples perturbation injection from gene-gene interaction.
    4.  **Latent-GRN**: Dynamically predicts the active regulatory graph (Condition-Specific).

    Mathematical Intent:
    Model the change in gene expression $Y$ as a function of baseline state $X_0$ and 
    perturbation $P$: $Y = f(X_0, P; \mathcal{G}, \mathcal{M})$ where $\mathcal{G}$ is 
    the GRN and $\mathcal{M}$ is the pathway membership.
    """

    def __init__(
        self, 
        cfg: ModelConfig, 
        grn: Optional[Union[np.ndarray, sp.spmatrix]] = None, 
        gene_pathway: Optional[Union[np.ndarray, sp.spmatrix]] = None
    ):
        super().__init__()
        self.cfg = cfg
        n_genes, d_model = cfg.n_genes, cfg.d_model

        # Densify inputs
        grn = _ensure_dense(grn)
        gene_pathway = _ensure_dense(gene_pathway)

        # Embeddings
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.expr_proj = nn.Linear(1, d_model)
        self.dose_proj = nn.Linear(1, d_model)
        
        # Perturbation MLP: [Chemberta, Target_Emb, Dose] -> D_pert
        pert_input_dim = cfg.chemberta_dim + d_model + 1
        self.pert_mlp = nn.Sequential(
            nn.Linear(pert_input_dim, cfg.d_pert),
            nn.GELU(),
            nn.Linear(cfg.d_pert, d_model)
        )

        # Layers
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.head_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        
        self.register_buffer("gene_ids", torch.arange(n_genes), persistent=False)

        # Inductive Biases
        self._init_grn_bias(grn)
        self._init_grn_propagation(grn)
        self._init_pathway_prior(gene_pathway)
        
        if cfg.latent_grn:
            from pmoe.models.layers import LatentGRNInference
            self.lri = LatentGRNInference(cfg)
        else:
            self.lri = None

        self.grad_checkpointing = False
        self._last_aux_loss = torch.tensor(0.0)
        self._last_kl_loss = torch.tensor(0.0)

    def _init_grn_bias(self, grn: Optional[np.ndarray]) -> None:
        """Initializes the additive GRN attention bias."""
        if grn is not None and self.cfg.use_grn_mask:
            bias = self._build_grn_bias(grn)
            self.register_buffer("grn_bias", torch.from_numpy(bias), persistent=False)
        else:
            self.grn_bias = None

    def _init_grn_propagation(self, grn: Optional[np.ndarray]) -> None:
        """Initializes soft GRN message-passing."""
        if self.cfg.grn_propagation and grn is not None:
            self.grn_prop = GRNPropagation(self.cfg, grn)
        else:
            self.grn_prop = None

    def _init_pathway_prior(self, gene_pathway: Optional[np.ndarray]) -> None:
        """Initializes the MoE router prior based on pathway membership."""
        if gene_pathway is not None and self.cfg.use_pathway_prior:
            gp = gene_pathway.astype(bool)
            n_genes, n_pathways = gp.shape
            n_experts = self.cfg.n_experts
            
            prior = np.zeros((n_genes, n_experts), dtype=np.float32)
            expert_mapping = np.arange(n_pathways) % n_experts
            
            for p in range(n_pathways):
                prior[gp[:, p], expert_mapping[p]] += 2.0
            self.register_buffer("pathway_prior", torch.from_numpy(prior), persistent=False)
        else:
            self.pathway_prior = None

    def _build_grn_bias(self, grn: np.ndarray) -> np.ndarray:
        """Builds the (N, N) additive bias matrix for GRN-constrained attention."""
        n = grn.shape[0]
        diag = np.arange(n)
        
        is_weighted = self.cfg.weighted_grn and np.issubdtype(grn.dtype, np.floating)
        
        if is_weighted:
            weights = np.abs(grn.astype(np.float32))
            weights = np.maximum(weights, weights.T)  # Symmetrize
            mask = weights != 0.0
            mask[diag, diag] = True
            
            max_w = float(weights.max()) if weights.size and weights.max() > 0 else 1.0
            scale = 2.0
            bias = np.where(mask, scale * (weights / max_w), float("-inf")).astype(np.float32)
            bias[diag, diag] = 0.0
        else:
            mask = (grn != 0)
            mask = mask | mask.T
            mask[diag, diag] = True
            bias = np.where(mask, 0.0, float("-inf")).astype(np.float32)
            
        return bias

    def forward(
        self, 
        ctrl_mean: torch.Tensor,
        target_idx: torch.Tensor,
        chemberta: torch.Tensor,
        dose_log: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Args:
            ctrl_mean: Baseline expression (B, N).
            target_idx: Indices of perturbed genes (B,).
            chemberta: Drug chemical embeddings (B, D_chem).
            dose_log: Log-scaled dosage (B,).
        Returns:
            Predicted LFC (B, N).
        """
        batch_size, n_genes = ctrl_mean.shape
        d_model = self.cfg.d_model
        
        # 1. Base Gene Tokens
        # Shape: (B, N, D)
        tokens = self.gene_emb(self.gene_ids).unsqueeze(0).expand(batch_size, n_genes, d_model)
        tokens = tokens + self.expr_proj(ctrl_mean.unsqueeze(-1))
        
        # 2. Perturbation Context
        # target_idx can be -1 for controls
        valid_targets = target_idx.clamp(min=0)
        target_emb = self.gene_emb(valid_targets)
        target_emb = torch.where((target_idx >= 0).unsqueeze(-1), target_emb, torch.zeros_like(target_emb))
        
        pert_features = torch.cat([chemberta, target_emb, dose_log.unsqueeze(-1)], dim=-1)
        pert_emb = self.pert_mlp(pert_features)  # (B, D)
        
        if self.cfg.pert_as_token:
            tokens = tokens + pert_emb.unsqueeze(1)
            context = None
        else:
            context = pert_emb.unsqueeze(1)  # (B, 1, D) for cross-attention
            
        # 2b. Latent Regulatory Inference
        current_grn_bias = self.grn_bias
        kl_loss = tokens.new_zeros(())
        if self.lri is not None:
            # gate: (B, N, N), kl: scalar
            gate, kl_loss = self.lri(pert_emb)
            if self.grn_bias is not None:
                # Apply gate to existing bias.
                current_grn_bias = self.grn_bias.unsqueeze(0) + torch.log(gate + 1e-9)
            else:
                current_grn_bias = torch.log(gate + 1e-9)
            
            # Ensure broadcastability over heads: (B, 1, N, N)
            current_grn_bias = current_grn_bias.unsqueeze(1)

        # 3. Backbone Layers
        total_aux_loss = tokens.new_zeros(())
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                tokens, layer_aux = checkpoint(
                    layer, tokens, context, current_grn_bias, self.pathway_prior, use_reentrant=False
                )
            else:
                tokens, layer_aux = layer(tokens, context, current_grn_bias, self.pathway_prior)
            total_aux_loss = total_aux_loss + layer_aux
            
        # 4. Global Refinement & Prediction
        if self.grn_prop is not None:
            tokens = self.grn_prop(tokens)
            
        predictions = self.head(self.head_ln(tokens)).squeeze(-1)
        self._last_aux_loss = total_aux_loss / max(1, len(self.layers))
        self._last_kl_loss = kl_loss
        
        return predictions

    def aux_loss(self) -> torch.Tensor:
        """Returns the accumulated MoE load balancing loss from the last forward pass."""
        return self._last_aux_loss

    def kl_loss(self) -> torch.Tensor:
        """Returns the LRI KL divergence from the last forward pass."""
        return self._last_kl_loss


def count_params(model: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
