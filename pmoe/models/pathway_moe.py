"""PathwayMoE-Perturb: GRN-sparsity-constrained, pathway Mixture-of-Experts model.

Condition-level model: predicts pseudobulk LFC over N genes given the cell-line baseline
expression, cell-line id, drug chemistry, target gene, and dose.

Causal factorization (each ablatable):
  - GRN-masked self-attention   : half the heads may only attend along GRN TF->target edges,
                                  injecting directed regulatory structure. (use_grn_mask)
  - perturbation cross-attention: the perturbation enters as a query against the cell state,
                                  never mixed into self-attention. (pert_as_token flips this)
  - Pathway top-2 MoE FFN       : experts == pathways; a gene's pathway membership biases the
                                  router toward its pathway's expert. (use_moe / use_pathway_prior)

Ported from v1 ``code/model.py``; the ``ModelConfig`` and layers are imported from the typed
package (``pmoe.config`` / ``pmoe.models.layers``). The only numerical extension is the **weighted
GRN attention bias** (audit item F): see :meth:`PathwayMoEPerturb._build_grn_bias`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from pmoe.config import ModelConfig
from pmoe.models.layers import CrossAttention, GRNPropagation, MixedAttention, MoEFFN


def _densify(a):
    """Accept scipy sparse OR dense; return a dense numpy array (or None)."""
    if a is None:
        return None
    if sp.issparse(a):
        a = a.toarray()
    return np.asarray(a)


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
        return x + self.ffn(h), x.new_zeros(())


class PathwayMoEPerturb(nn.Module):
    """GRN + pathway MoE perturbation model.

    Parameters
    ----------
    cfg : ModelConfig
        Typed config (imported from ``pmoe.config``). ``cfg.weighted_grn`` selects the weighted
        attention bias when ``grn`` carries float edge weights.
    grn : scipy.sparse | np.ndarray | None
        (N, N) gene-gene network. ``bool``/integer -> binary allow/deny mask. ``float`` and
        ``cfg.weighted_grn`` -> weighted additive bias (see :meth:`_build_grn_bias`).
    gene_pathway : scipy.sparse | np.ndarray | None
        (N, P) gene->pathway membership (bool); maps to MoE expert router prior.
    """

    def __init__(self, cfg: ModelConfig, grn=None, gene_pathway=None):
        super().__init__()
        self.cfg = cfg
        N, d = cfg.n_genes, cfg.d_model

        # accept scipy sparse OR dense; densify internally (port the densify guard from v1)
        grn = _densify(grn)
        gene_pathway = _densify(gene_pathway)

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

        # GRN attention bias (additive float (N,N)); non-persistent buffer (rebuilt at load).
        bias = self._build_grn_bias(grn) if (grn is not None and cfg.use_grn_mask) else None
        if bias is not None:
            self.register_buffer("grn_bias", torch.from_numpy(bias), persistent=False)
        else:
            self.grn_bias = None

        # SOFT GRN message-passing (orthogonal to the mask above). The GRNPropagation module
        # owns its own non-persistent P buffer (rebuilt at load). Same grn as the mask.
        if cfg.grn_propagation and grn is not None:
            self.grn_prop = GRNPropagation(cfg, grn)
        else:
            self.grn_prop = None

        # pathway prior over experts (N, E): map pathway membership to experts (1:1 truncated)
        if gene_pathway is not None and cfg.use_pathway_prior:
            gp = gene_pathway.astype(bool)
            P = gp.shape[1]
            prior = np.zeros((N, cfg.n_experts), np.float32)
            e_of_p = np.arange(P) % cfg.n_experts
            for p in range(P):
                prior[gp[:, p], e_of_p[p]] += 2.0
            self.register_buffer("pathway_prior", torch.from_numpy(prior), persistent=False)
        else:
            self.pathway_prior = None
        self.grad_checkpoint = False

    def _build_grn_bias(self, grn: np.ndarray) -> np.ndarray:
        """Build the (N, N) additive attention bias for the GRN-masked heads.

        Two flavors, selected by ``cfg.weighted_grn`` and the dtype of ``grn``:

        * **Binary** (``cfg.weighted_grn`` False, or ``grn`` is bool/int):
              allow = (grn != 0) symmetrized, plus the diagonal (self-edges always allowed).
              bias = 0 for allowed edges, -inf for absent edges.

        * **Weighted** (``cfg.weighted_grn`` True AND ``grn`` is float):
              Edge magnitude |w| in (0, 1] is mapped to a finite, non-negative additive bias so
              stronger edges receive more attention; the sign of w is intentionally discarded
              (an inhibitory edge is still an *edge to attend to*). The exact formula is

                  bias[i, j] = GRN_BIAS_SCALE * |w_ij| / max(|w|)        for an allowed edge
                  bias[i, j] = 0                                          for the diagonal
                  bias[i, j] = -inf                                       for an absent edge

              We normalize by ``max(|w|)`` so the strongest edge gets exactly ``GRN_BIAS_SCALE`` of
              additive logit and all biases are finite & non-negative; absent edges are masked out
              with -inf exactly as in the binary case. ``GRN_BIAS_SCALE = 2.0`` (a couple of nats,
              comparable to the pathway-prior magnitude) so weights meaningfully shift attention
              without saturating it. The matrix is symmetrized by taking the max |w| of (i,j)/(j,i).
        """
        N = grn.shape[0]
        diag = np.arange(N)
        weighted = bool(self.cfg.weighted_grn) and np.issubdtype(np.asarray(grn).dtype, np.floating)
        if weighted:
            w = np.abs(np.asarray(grn, dtype=np.float32))
            w = np.maximum(w, w.T)                     # symmetrize by strongest direction
            allow = w != 0.0
            allow[diag, diag] = True
            mx = float(w.max()) if w.size and w.max() > 0 else 1.0
            GRN_BIAS_SCALE = 2.0
            bias = np.where(allow, GRN_BIAS_SCALE * (w / mx), float("-inf")).astype(np.float32)
            bias[diag, diag] = 0.0                     # self-edge: neutral, never -inf
        else:
            allow = (np.asarray(grn) != 0)
            allow = allow | allow.T
            allow[diag, diag] = True
            bias = np.where(allow, 0.0, float("-inf")).astype(np.float32)
        return bias

    def forward(self, batch: dict) -> torch.Tensor:
        ctrl = batch["ctrl_mean"]                                 # (B,N)
        B, N = ctrl.shape
        d = self.cfg.d_model
        tok = self.gene_emb(self.gene_ids).unsqueeze(0).expand(B, N, d)
        tok = tok + self.expr_proj(ctrl.unsqueeze(-1))
        # perturbation embedding
        tgt_idx = batch["target_idx"]
        tgt_emb = self.gene_emb(tgt_idx.clamp(min=0))
        tgt_emb = torch.where((tgt_idx >= 0).unsqueeze(-1), tgt_emb, torch.zeros_like(tgt_emb))
        pert = self.pert_mlp(torch.cat(
            [batch["chemberta"], tgt_emb, batch["dose_log"].unsqueeze(-1)], dim=-1))  # (B,d)
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
        if self.grn_prop is not None:                             # soft GRN message-passing (residual)
            tok = self.grn_prop(tok)
        pred = self.head(self.head_ln(tok)).squeeze(-1)           # (B,N)
        self._last_aux = aux_total / max(1, self.cfg.n_layers)
        return pred

    def aux_loss(self) -> torch.Tensor:
        return getattr(self, "_last_aux", torch.zeros((), device=self.head.weight.device))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
