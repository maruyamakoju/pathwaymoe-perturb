"""PMoE Training Framework: Orchestration, Optimization, and Evaluation.

This module provides a high-level Trainer API for PMoE framework, featuring a modular 
callback system, hierarchical configuration support, and robust evaluation.
"""
from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pmoe.config import (ArchitectureConfig, GRNConfig, GRNVariant, ModelConfig,
                         MoEConfig, PerturbationConfig, RunSpec, SEED,
                         SIZE_PRESETS, TrainConfig, RUNS_DIR)
from pmoe.data.loader import load_conditions, load_genes
from pmoe.data.splits import make_splits
from pmoe.data.dataset import build_shared, make_loader
from pmoe.priors.grn import load_grn
from pmoe.priors.pathways import load_pathways
from pmoe.models.pathway_moe import PathwayMoEPerturb, count_params
from pmoe.eval.metrics import compute_metrics
from pmoe.io import save_checkpoint


# ----- Callbacks & Inversion of Control -----

class Callback(ABC):
    """Abstract base class for training callbacks."""
    def on_train_begin(self, trainer: Trainer) -> None: pass
    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None: pass
    def on_batch_end(self, trainer: Trainer, batch: Dict[str, torch.Tensor], loss: torch.Tensor) -> None: pass
    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Dict[str, float]) -> None: pass
    def on_train_end(self, trainer: Trainer) -> None: pass


class LoggingCallback(Callback):
    """Logs training progress to console and JSONL."""
    def __init__(self, log_path: Path, verbose: bool = True):
        self.log_path = log_path
        self.verbose = verbose
        self.log_file = log_path.open("w")

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Dict[str, float]) -> None:
        log_entry = {"epoch": epoch, **metrics}
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()
        if self.verbose:
            msg = f"  ep{epoch:3d} " + " ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
            print(msg, flush=True)

    def on_train_end(self, trainer: Trainer) -> None:
        self.log_file.close()


# ----- Loss Functions -----

def weighted_huber_loss(
    pred: torch.Tensor, 
    true: torch.Tensor, 
    w_mover: float = 3.0, 
    delta: float = 1.0
) -> torch.Tensor:
    r"""Computes a weighted Huber loss, prioritizing high-magnitude responses.
    
    $L = w \cdot \text{Huber}(pred - true, \delta)$ where $w = 1 + w_{mover} \cdot |true|$.
    """
    weights = 1.0 + w_mover * true.abs()
    err = (pred - true).abs()
    quad_mask = err <= delta
    
    loss = torch.where(
        quad_mask,
        0.5 * err ** 2,
        delta * (err - 0.5 * delta)
    )
    return (weights * loss).mean()


# ----- Trainer -----

class Trainer:
    """Orchestrates the training lifecycle of a PMoE model."""

    def __init__(
        self,
        model: PathwayMoEPerturb,
        train_config: TrainConfig,
        device: Union[str, torch.device],
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model.to(device)
        self.cfg = train_config
        self.device = device
        self.callbacks = callbacks or []
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.95)
        )
        # torch.amp.GradScaler('cuda', ...) is the non-deprecated form on torch >= 2.5
        self._use_cuda_amp = (str(device) == "cuda" or getattr(device, "type", None) == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_cuda_amp)
        self.step_count = 0

    def _get_lr_multiplier(self) -> float:
        """Computes Cosine Annealing learning rate multiplier with warmup."""
        total_steps = self.cfg.epochs * self.steps_per_epoch
        if self.step_count < self.cfg.warmup:
            return self.step_count / max(1, self.cfg.warmup)
        
        progress = (self.step_count - self.cfg.warmup) / max(1, total_steps - self.cfg.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.steps_per_epoch = len(loader) // self.cfg.grad_accum
        
        self.optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self._use_cuda_amp, dtype=torch.bfloat16):
                # model forward expects explicit args or **kwargs
                pred = self.model(**batch)
                loss = weighted_huber_loss(pred.float(), batch["lfc"].float())
                
                # MoE Load Balancing Loss
                if self.model.cfg.use_moe:
                    loss = loss + self.cfg.lb_weight * self.model.aux_loss().float()
                
                # Latent GRN KL Divergence Loss
                if self.model.cfg.latent_grn:
                    kl_weight = self.model.cfg.latent_grn_kl_weight
                    loss = loss + kl_weight * self.model.kl_loss().float()
                
            loss = loss / self.cfg.grad_accum
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.cfg.grad_accum == 0:
                # LR Scheduling
                lr = self.cfg.lr * self._get_lr_multiplier()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step_count += 1
                
            total_loss += loss.item() * self.cfg.grad_accum
            for cb in self.callbacks:
                cb.on_batch_end(self, batch, loss)
                
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Validation pass in fp32 (no autocast).

        The paper's "deterministic fp32 eval" claim used to apply only to predict_test;
        keeping val eval in bf16 created a (small) inconsistency in how the best-checkpoint
        metric was computed vs the published test metric. fp32 here costs a few seconds
        per epoch and makes the early-stop signal byte-reproducible across runs.
        """
        self.model.eval()
        all_preds, all_true, all_masks = [], [], []
        for batch in loader:
            batch_gpu = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = self.model(**batch_gpu)
            all_preds.append(pred.float().cpu().numpy())
            all_true.append(batch["lfc"].numpy())
            all_masks.append(batch["deg_mask"].numpy())
        return compute_metrics(
            np.concatenate(all_preds),
            np.concatenate(all_true),
            np.concatenate(all_masks),
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        for cb in self.callbacks: cb.on_train_begin(self)
        
        best_val_score = -1e9
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.cfg.epochs):
            for cb in self.callbacks: cb.on_epoch_begin(self, epoch)
            
            train_loss = self.train_epoch(train_loader)
            
            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs - 1:
                metrics = self.evaluate(val_loader)
                val_score = metrics["pearson_deg50"]
                
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, {"val_deg50": val_score, "train_loss": train_loss})
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break
                        
        for cb in self.callbacks: cb.on_train_end(self)
        return {"best_state": best_state, "best_val_deg50": best_val_score}


def make_hierarchical_config(
    dataset: str, 
    size: str, 
    variant: str, 
    n_genes: int, 
    n_experts: int, 
    cb_dim: int,
    grn_propagation: bool = False,
    use_grn_mask: Optional[bool] = None,
    latent_grn: bool = False,
    latent_grn_kl_weight: Optional[float] = None,
) -> ModelConfig:
    """Constructs a hierarchical ModelConfig from flat parameters."""
    preset = SIZE_PRESETS[size]
    v = GRNVariant(variant)
    
    arch = ArchitectureConfig(
        d_model=preset["d_model"],
        n_heads=preset["n_heads"],
        n_layers=preset["n_layers"],
        n_genes=n_genes
    )
    pert = PerturbationConfig(
        d_pert=preset["d_pert"],
        chemberta_dim=cb_dim
    )
    grn = GRNConfig(
        use_grn_mask=(v != GRNVariant.NONE) if use_grn_mask is None else use_grn_mask,
        weighted_grn=(v == GRNVariant.TRRUST_WEIGHTED),
        grn_propagation=grn_propagation,
        latent_grn=latent_grn,
        **({"latent_grn_kl_weight": latent_grn_kl_weight}
           if latent_grn_kl_weight is not None else {}),
    )
    moe = MoEConfig(
        n_experts=n_experts
    )
    
    return ModelConfig(arch=arch, pert=pert, grn=grn, moe=moe)


def train(
    dataset: str, 
    split: str, 
    variant: str = "none", 
    seed: int = SEED,
    tc: Optional[TrainConfig] = None, 
    size: str = "base", 
    verbose: bool = True,
    grn_propagation: bool = False, 
    use_grn_mask: Optional[bool] = None,
    latent_grn: bool = False,
    latent_grn_kl_weight: Optional[float] = None,
    tag: str = ""
) -> Dict[str, Any]:
    """High-level training entry point."""
    tc = tc or TrainConfig(seed=seed)
    if tc.deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True so a nondeterministic op (e.g. scatter_add backward) prints a
        # warning instead of raising — required for SDPA + scatter on some kernels.
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunSpec(dataset=dataset, split=split, variant=variant, seed=seed, size=size, tag=tag)

    # 1. Load Data & Priors
    df = load_conditions(dataset)
    genes = load_genes(dataset)
    shared = build_shared(dataset, df)
    n_genes, cb_dim = shared["n_genes"], shared["cb_dim"]
    
    gp, _ = load_pathways(dataset)
    n_experts = gp.shape[1]
    
    grn = None if GRNVariant(variant) == GRNVariant.NONE else load_grn(dataset, variant, split)
    if GRNVariant(variant) != GRNVariant.NONE and grn is None:
        raise RuntimeError(f"GRN '{variant}' not found for split '{split}'")

    # 2. Prepare DataLoaders
    splits = make_splits(df, split, seed=SEED)
    tr_loader = make_loader(dataset, splits["train"], df, shared, tc.batch_size, shuffle=True)
    va_loader = make_loader(dataset, splits["val"], df, shared, tc.batch_size, shuffle=False)

    # 3. Initialize Model & Trainer
    cfg = make_hierarchical_config(
        dataset, size, variant, n_genes, n_experts, cb_dim,
        grn_propagation=grn_propagation, use_grn_mask=use_grn_mask,
        latent_grn=latent_grn, latent_grn_kl_weight=latent_grn_kl_weight
    )
    model = PathwayMoEPerturb(cfg, grn, gp)
    model.grad_checkpointing = tc.grad_ckpt
    
    if verbose:
        print(f"=== {run.name} === params={count_params(model):,} variant={variant}")

    callbacks = [LoggingCallback(RUNS_DIR / f"{run.name}.jsonl", verbose=verbose)]
    trainer = Trainer(model, tc, device, callbacks=callbacks)
    
    # 4. Fit
    results = trainer.fit(tr_loader, va_loader)
    
    # 5. Save
    save_checkpoint(run, results["best_state"], cfg, extra={
        "best_val_deg50": results["best_val_deg50"],
        "variant": variant,
        "split": split,
        "train_cfg": tc.to_dict()
    })
    
    return {"name": run.name, "best_val_deg50": results["best_val_deg50"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe_full")
    ap.add_argument("--split", default="unseen_drug")
    ap.add_argument("--variant", default="none")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--size", default="base")
    ap.add_argument("--latent-grn", action="store_true")
    a = ap.parse_args()
    train(a.dataset, a.split, a.variant, a.seed,
          TrainConfig(epochs=a.epochs, batch_size=a.batch_size, seed=a.seed), 
          a.size, latent_grn=a.latent_grn)
