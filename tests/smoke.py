"""Fast end-to-end smoke test on a tiny synthetic dataset. Run after install:
    .\.venv\Scripts\python.exe tests\smoke.py
Validates: synth -> drugs -> baselines featurizer -> model tiny fwd/bwd -> one train step -> metrics.
Exits non-zero on any failure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
os.environ.setdefault("VC_DATA_ROOT", r"E:\vc_project_data")
import numpy as np, torch

def main():
    import synth, drugs, data, splits, metrics
    from model import ModelConfig, PathwayMoEPerturb
    from loader import load_conditions

    print("1) build tiny synthetic dataset (n_genes=300)...")
    df, genes, pw, gp, grn, dm = synth.build(n_genes=300, n_pathways=8, seed=1337, n_cells_mean=120)
    synth.write(df, genes, pw, gp, grn, dm, dataset="synthetic_smoke")
    assert len(df) > 100

    print("2) drug feats...")
    f = drugs.build_drug_feats("synthetic_smoke"); assert len(f) > 10

    print("3) splits...")
    for mode in ["unseen_drug", "unseen_cell_line", "unseen_both"]:
        s = splits.make_splits(df, mode)
        assert len(s["train"]) and len(s["test"]), mode
        # leakage check for unseen_drug
        if mode == "unseen_drug":
            tr_d = set(df.iloc[s["train"]]["treatment"]); te_d = set(df.iloc[s["test"]]["treatment"])
            assert not (tr_d & te_d), "drug leakage!"
    print("   splits OK (no drug leakage)")

    print("4) model tiny fwd/bwd on cuda?", torch.cuda.is_available())
    shared = data.build_shared("synthetic_smoke", df)
    cfg = ModelConfig(n_genes=300, n_experts=gp.shape[1], d_model=64, n_heads=4, n_layers=2,
                      chemberta_dim=shared["cb_dim"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PathwayMoEPerturb(cfg, grn, gp.astype(bool) if hasattr(gp, "astype") else gp).to(dev)
    loader = data.make_loader("synthetic_smoke", np.arange(min(32, len(df))), df, shared,
                              batch_size=8, shuffle=True)
    batch = next(iter(loader))
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        pred = m(b)
    loss = torch.nn.functional.mse_loss(pred.float(), b["lfc"].float()) + 0.1 * m.aux_loss().float()
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(pred).all() and np.isfinite(gn), "NaN in fwd/bwd"
    print(f"   pred {tuple(pred.shape)} loss={loss.item():.3f} grad_norm={gn:.2f} OK")

    print("5) metrics sanity...")
    true = b["lfc"].cpu().numpy()
    r_good = metrics.compute_metrics(true + 0.1 * np.random.randn(*true.shape), true,
                                     b["deg_mask"].cpu().numpy())["pearson_deg50"]
    r_zero = metrics.compute_metrics(np.zeros_like(true), true,
                                     b["deg_mask"].cpu().numpy())["pearson_deg50"]
    assert r_good > r_zero, (r_good, r_zero)
    print(f"   DEG50 good={r_good:.3f} > zero={r_zero:.3f} OK")
    print("\nALL SMOKE TESTS PASSED")

if __name__ == "__main__":
    main()
