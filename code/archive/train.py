"""Train PathwayMoE-Perturb on a dataset/split. Logs to runs/<name>.jsonl, checkpoints best
val DEG-Pearson to E:/checkpoints. wandb optional (--wandb)."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from config import CKPT_DIR, RUNS_DIR, SEED, DATA_ROOT
from loader import load_conditions, load_genes
from splits import make_splits
from data import build_shared, make_loader
from model import ModelConfig, PathwayMoEPerturb, count_params
from metrics import compute_metrics
import scipy.sparse as sp


def load_priors(dataset, n_genes, grn_file="grn_mask.npz"):
    pri = DATA_ROOT / "data" / "priors" / dataset
    gp = sp.load_npz(pri / "pathways.npz").toarray() if (pri / "pathways.npz").exists() else None
    grn = sp.load_npz(pri / grn_file).toarray().astype(bool) if (pri / grn_file).exists() else None
    if grn is not None:
        print(f"GRN prior: {pri / grn_file} ({grn.sum()} edges)")
    return grn, gp


def weighted_huber(pred, true, w_mover=3.0, delta=1.0):
    w = 1.0 + w_mover * true.abs()
    err = pred - true
    abs_e = err.abs()
    quad = torch.minimum(abs_e, torch.tensor(delta, device=err.device))
    lin = abs_e - quad
    loss = 0.5 * quad ** 2 + delta * lin
    return (w * loss).mean()


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    preds, trues, degs = [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            p = model(b)
        preds.append(p.float().cpu().numpy())
        trues.append(batch["lfc"].numpy())
        degs.append(batch["deg_mask"].numpy())
    pred = np.concatenate(preds); true = np.concatenate(trues); deg = np.concatenate(degs)
    return compute_metrics(pred, true, deg)


def build_config(args, n_genes, n_experts, cb_dim):
    if args.size == "tiny":
        c = dict(d_model=64, n_heads=4, n_layers=2, d_pert=64)
    elif args.size == "small":
        c = dict(d_model=192, n_heads=6, n_layers=3, d_pert=192)
    else:  # base
        c = dict(d_model=256, n_heads=8, n_layers=4, d_pert=256)
    return ModelConfig(n_genes=n_genes, n_experts=n_experts, chemberta_dim=cb_dim,
                       use_grn_mask=not args.no_grn, use_moe=not args.no_moe,
                       use_pathway_prior=not args.no_pathway_prior,
                       pert_as_token=args.pert_as_token, dropout=args.dropout, **c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--split", default="unseen_drug",
                    choices=["unseen_drug", "unseen_cell_line", "unseen_both"])
    ap.add_argument("--size", default="base", choices=["tiny", "small", "base"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lb-weight", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--no-grn", action="store_true")
    ap.add_argument("--grn-file", default="grn_mask.npz", help="GRN variant npz in priors/<dataset>/")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-moe", action="store_true")
    ap.add_argument("--no-pathway-prior", action="store_true")
    ap.add_argument("--pert-as-token", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grn_tag = args.grn_file.replace("grn_", "").replace(".npz", "")
    name = args.name or f"{args.dataset}_{args.split}_{args.size}"
    if args.grn_file != "grn_mask.npz": name += f"_{grn_tag}"
    if args.seed != SEED: name += f"_s{args.seed}"
    if args.no_grn: name += "_nogrn"
    if args.no_moe: name += "_nomoe"
    if args.no_pathway_prior: name += "_noprior"
    if args.pert_as_token: name += "_perttoken"
    print(f"=== {name} on {device} ===")

    df = load_conditions(args.dataset)
    genes = load_genes(args.dataset)
    shared = build_shared(args.dataset, df)
    n_genes = shared["n_genes"]; cb_dim = shared["cb_dim"]
    grn, gp = load_priors(args.dataset, n_genes, args.grn_file)
    n_experts = gp.shape[1] if gp is not None else 16

    sp_ = make_splits(df, args.split, seed=SEED)   # FIXED split seed: identical test set across runs
    print(f"split {args.split}: train={len(sp_['train'])} val={len(sp_['val'])} test={len(sp_['test'])}")
    tr = make_loader(args.dataset, sp_["train"], df, shared, args.batch_size, shuffle=True)
    va = make_loader(args.dataset, sp_["val"], df, shared, args.batch_size, shuffle=False)

    cfg = build_config(args, n_genes, n_experts, cb_dim)
    model = PathwayMoEPerturb(cfg, grn, gp).to(device)
    model.grad_checkpoint = args.grad_ckpt
    print(f"params={count_params(model):,} d={cfg.d_model} L={cfg.n_layers} "
          f"E={cfg.n_experts} grn={cfg.use_grn_mask} moe={cfg.use_moe}")

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.wd,
                                  betas=(0.9, 0.95))
        print("optimizer: bnb.AdamW8bit")
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                                betas=(0.9, 0.95))
        print("optimizer: torch.AdamW")

    steps_per_epoch = max(1, math.ceil(len(sp_["train"]) / args.batch_size / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    run_log = (RUNS_DIR / f"{name}.jsonl").open("w")
    wb = None
    if args.wandb:
        try:
            import wandb; wb = wandb.init(project="pathwaymoe-perturb", name=name, config=vars(args))
        except Exception as e:
            print(f"wandb disabled: {e}")

    best, best_state, since = -1e9, None, 0
    step = 0; t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(tr):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(b)
                lfc_loss = weighted_huber(pred.float(), b["lfc"].float())
                aux = model.aux_loss().float()
                loss = lfc_loss + args.lb_weight * aux
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vm = evaluate_loader(model, va, device)
            score = vm["pearson_deg50"]
            rec = dict(epoch=epoch, step=step, train_loss=float(loss.item()),
                       lfc_loss=float(lfc_loss.item()), aux=float(aux.item()),
                       val_deg50=score, val_all=vm["pearson_all"],
                       val_dir50=vm["direction_acc50"], sec=round(time.time() - t0, 1))
            run_log.write(json.dumps(rec) + "\n"); run_log.flush()
            if wb: wb.log(rec)
            print(f"ep{epoch:3d} step{step:4d} loss={loss.item():.4f} "
                  f"val_DEG50={score:.3f} val_all={vm['pearson_all']:.3f} "
                  f"dir50={vm['direction_acc50']:.3f}")
            if score > best:
                best, since = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= args.patience:
                    print(f"early stop (no val improvement in {args.patience} evals)"); break

    if best_state is not None:
        ckpt = CKPT_DIR / f"{name}.pt"
        torch.save({"model": best_state, "cfg": cfg.__dict__, "best_val_deg50": best,
                    "args": vars(args)}, ckpt)
        print(f"best val DEG50={best:.3f} -> saved {ckpt}")
    run_log.close()
    if wb: wb.finish()
    return best


if __name__ == "__main__":
    main()
