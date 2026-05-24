"""Training as an importable function (robust; no fragile subprocess). Used by study.py and CLI."""
from __future__ import annotations
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

from pmoe.config import (ModelConfig, TrainConfig, RunSpec, SIZE_PRESETS, SEED,
                         RUNS_DIR, GRNVariant)
from pmoe.data.loader import load_conditions, load_genes
from pmoe.data.splits import make_splits
from pmoe.data.dataset import build_shared, make_loader
from pmoe.priors.grn import load_grn
from pmoe.priors.pathways import load_pathways
from pmoe.models.pathway_moe import PathwayMoEPerturb, count_params
from pmoe.eval.metrics import compute_metrics
from pmoe.io import save_checkpoint


def weighted_huber(pred, true, w_mover=3.0, delta=1.0):
    w = 1.0 + w_mover * true.abs()
    e = (pred - true).abs()
    quad = torch.minimum(e, torch.tensor(delta, device=e.device))
    return (w * (0.5 * quad ** 2 + delta * (e - quad))).mean()


@torch.no_grad()
def _eval(model, loader, device):
    model.eval(); P, T, D = [], [], []
    for b in loader:
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            P.append(model(bb).float().cpu().numpy())
        T.append(b["lfc"].numpy()); D.append(b["deg_mask"].numpy())
    return compute_metrics(np.concatenate(P), np.concatenate(T), np.concatenate(D))


def make_model_config(dataset, size, variant, n_genes, n_experts, cb_dim,
                      grn_propagation: bool = False,
                      use_grn_mask: bool | None = None) -> ModelConfig:
    preset = SIZE_PRESETS[size]
    v = GRNVariant(variant)
    mask = (v != GRNVariant.NONE) if use_grn_mask is None else use_grn_mask
    return ModelConfig(n_genes=n_genes, n_experts=n_experts, chemberta_dim=cb_dim,
                       use_grn_mask=mask,
                       weighted_grn=(v == GRNVariant.TRRUST_WEIGHTED),
                       grn_propagation=grn_propagation, **preset)


def train(dataset: str, split: str, variant: str = "none", seed: int = SEED,
          tc: TrainConfig | None = None, size: str = "base", verbose: bool = True,
          grn_propagation: bool = False, use_grn_mask: bool | None = None,
          tag: str = "") -> dict:
    tc = tc or TrainConfig(seed=seed)
    tc.seed = seed
    if tc.deterministic:
        torch.manual_seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = RunSpec(dataset=dataset, split=split, variant=variant, seed=seed, size=size, tag=tag)

    df = load_conditions(dataset); genes = load_genes(dataset)
    shared = build_shared(dataset, df)
    n_genes, cb_dim = shared["n_genes"], shared["cb_dim"]
    gp, _ = load_pathways(dataset)
    n_experts = gp.shape[1]
    grn = None if GRNVariant(variant) == GRNVariant.NONE else load_grn(dataset, variant, split)
    # fail loudly: a non-"none" variant MUST have a GRN, else we'd silently train without one
    # (this exact silent failure invalidated an earlier ground_truth positive control).
    if GRNVariant(variant) != GRNVariant.NONE and grn is None:
        raise RuntimeError(f"GRN for variant '{variant}' (split={split}) not found in priors — "
                           f"build it first (build_grn). Refusing to train silently without a GRN.")

    sp_ = make_splits(df, split, seed=SEED)            # FIXED split seed across model seeds
    tr, va = sp_["train"], sp_["val"]
    tl = make_loader(dataset, tr, df, shared, tc.batch_size, shuffle=True)
    vl = make_loader(dataset, va, df, shared, tc.batch_size, shuffle=False)

    cfg = make_model_config(dataset, size, variant, n_genes, n_experts, cb_dim,
                            grn_propagation=grn_propagation, use_grn_mask=use_grn_mask)
    model = PathwayMoEPerturb(cfg, grn, gp).to(device); model.grad_checkpoint = tc.grad_ckpt
    if verbose:
        print(f"=== {run.name} === params={count_params(model):,} variant={variant} "
              f"grn_edges={int(grn.nnz) if grn is not None else 0} device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay,
                            betas=(0.9, 0.95))
    spe = max(1, math.ceil(len(tr) / tc.batch_size / tc.grad_accum)); total = spe * tc.epochs

    def lr_at(s):
        if s < tc.warmup: return s / max(1, tc.warmup)
        p = (s - tc.warmup) / max(1, total - tc.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    log = (RUNS_DIR / f"{run.name}.jsonl").open("w")
    best, best_state, since, step = -1e9, None, 0, 0
    for ep in range(tc.epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(tl):
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(bb)
                loss = weighted_huber(pred.float(), bb["lfc"].float()) + tc.lb_weight * model.aux_loss().float()
            (loss / tc.grad_accum).backward()
            if (bi + 1) % tc.grad_accum == 0:
                for g in opt.param_groups: g["lr"] = tc.lr * lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if ep % tc.eval_every == 0 or ep == tc.epochs - 1:
            vm = _eval(model, vl, device); sc = vm["pearson_deg50"]
            log.write(json.dumps(dict(epoch=ep, val_deg50=sc, val_all=vm["pearson_all"])) + "\n"); log.flush()
            if verbose: print(f"  ep{ep:3d} val_DEG50={sc:.3f} val_all={vm['pearson_all']:.3f}", flush=True)
            if sc > best:
                best, since = sc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= tc.patience:
                    if verbose: print(f"  early stop ep{ep}", flush=True); break
    log.close()
    save_checkpoint(run, best_state, cfg, extra={"best_val_deg50": best, "variant": variant,
                                                 "split": split, "train_cfg": tc.__dict__})
    if verbose: print(f"  best val DEG50={best:.3f} -> {run.ckpt_path}", flush=True)
    return {"name": run.name, "best_val_deg50": best}


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
    a = ap.parse_args()
    train(a.dataset, a.split, a.variant, a.seed,
          TrainConfig(epochs=a.epochs, batch_size=a.batch_size, seed=a.seed), a.size)
