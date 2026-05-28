"""Linear baselines B1/B2/B3 for perturbation LFC prediction (sklearn only, no torch).

B1  Mean perturbation effect : per-drug mean LFC across training cell lines (global mean for
                               drugs unseen in train). Strong on the unseen-cell-line split.
B2  Ridge                    : input = ChemBERTa(SMILES) + one-hot(cell_line) + ctrl_mean(cell line);
                               output = LFC in R^N. alpha grid-searched on the val split.
B3  Ridge + biology          : B2 features + target-gene baseline expr, target-gene one-hot, and
                               target-gene pathway membership. (Feb-2026 bioRxiv: target-based
                               features beat SMILES-only.)

CLI runs all three over all three splits x four metric families -> results/baselines_<ds>.json.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from config import RESULTS_DIR, DATA_ROOT
from loader import load_conditions, load_genes, stack_arrays
from splits import make_splits
from metrics import evaluate
from drugs import load_drug_feats
import scipy.sparse as sp

ALPHAS = [1.0, 10.0, 100.0, 1000.0]


# ---------- shared feature building ----------
class Featurizer:
    def __init__(self, dataset: str, df: pd.DataFrame):
        self.genes = load_genes(dataset)
        self.gidx = {g: i for i, g in enumerate(self.genes)}
        self.cell_lines = sorted(df["cell_line"].unique().tolist())
        self.cl_enc = OneHotEncoder(categories=[self.cell_lines], handle_unknown="ignore",
                                    sparse_output=False)
        self.cl_enc.fit(df[["cell_line"]])
        dfeats = load_drug_feats(dataset).set_index("treatment")
        self.chemberta = {t: np.asarray(r) for t, r in dfeats["chemberta"].items()}
        self.cb_dim = len(next(iter(self.chemberta.values())))
        self.targets = sorted(df["target_gene"].fillna("").unique().tolist())
        self.tgt_enc = OneHotEncoder(categories=[self.targets], handle_unknown="ignore",
                                     sparse_output=False)
        self.tgt_enc.fit(df[["target_gene"]].fillna(""))
        pri = DATA_ROOT / "data" / "priors" / dataset / "pathways.npz"
        self.gene_pw = sp.load_npz(pri).toarray() if pri.exists() else None
        self.ctrl = stack_arrays(df, "ctrl_mean")

    def chem(self, df):
        return np.stack([self.chemberta.get(t, np.zeros(self.cb_dim)) for t in df["treatment"]])

    def base(self, df, idx):
        """B2 features."""
        cb = self.chem(df.iloc[idx])
        cl = self.cl_enc.transform(df.iloc[idx][["cell_line"]])
        ctrl = self.ctrl[idx]
        return np.concatenate([cb, cl, ctrl], axis=1).astype(np.float32)

    def bio(self, df, idx):
        """B3 = B2 + target baseline expr + target one-hot + target pathway membership."""
        sub = df.iloc[idx]
        tgt_oh = self.tgt_enc.transform(sub[["target_gene"]].fillna(""))
        tbe = np.array([[self.ctrl[i, self.gidx[t]]] if t in self.gidx else [0.0]
                        for i, t in zip(idx, sub["target_gene"].fillna(""))], np.float32)
        if self.gene_pw is not None:
            tpw = np.stack([self.gene_pw[self.gidx[t]] if t in self.gidx
                            else np.zeros(self.gene_pw.shape[1]) for t in sub["target_gene"].fillna("")])
        else:
            tpw = np.zeros((len(idx), 0), np.float32)
        return np.concatenate([self.base(df, idx), tbe, tgt_oh, tpw], axis=1).astype(np.float32)


# ---------- baselines ----------
class B1MeanEffect:
    name = "B1_mean_effect"

    def fit(self, df, train_idx, **kw):
        Y = stack_arrays(df, "lfc")
        self.global_mean = Y[train_idx].mean(0)
        self.per_drug = {}
        sub = df.iloc[train_idx]
        for drug, g in sub.groupby("treatment"):
            self.per_drug[drug] = Y[g.index.to_numpy()].mean(0)
        return self

    def predict(self, df, idx):
        sub = df.iloc[idx]
        return np.stack([self.per_drug.get(t, self.global_mean) for t in sub["treatment"]])


class _RidgeBase:
    feat = "base"

    def __init__(self):
        self.alpha = ALPHAS[0]

    def _X(self, fz, df, idx):
        return getattr(fz, self.feat)(df, idx)

    def fit(self, df, train_idx, val_idx=None, fz=None, deg=None):
        self.fz = fz
        Xtr = self._X(fz, df, train_idx)
        Y = stack_arrays(df, "lfc")
        Ytr = Y[train_idx]
        best, best_score = None, -1e9
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(Xtr, Ytr)
            if val_idx is not None and len(val_idx):
                pv = m.predict(self._X(fz, df, val_idx))
                dm = deg[val_idx] if deg is not None else None
                score = evaluate(pv, Y[val_idx], dm)["overall"]["pearson_deg50"]
            else:
                score = 0.0
            if score > best_score:
                best, best_score, self.alpha = m, score, a
        self.model = best
        return self

    def predict(self, df, idx):
        return self.model.predict(self._X(self.fz, df, idx))


class B2Ridge(_RidgeBase):
    name = "B2_ridge"; feat = "base"


class B3RidgeBio(_RidgeBase):
    name = "B3_ridge_bio"; feat = "bio"


def run(dataset: str):
    df = load_conditions(dataset)
    Y = stack_arrays(df, "lfc")
    deg = stack_arrays(df, "deg_mask")
    fz = Featurizer(dataset, df)
    results = {}
    for mode in ["unseen_drug", "unseen_cell_line", "unseen_both"]:
        sp_ = make_splits(df, mode)
        tr, va, te = sp_["train"], sp_["val"], sp_["test"]
        results[mode] = {}
        groups = df.iloc[te]["cell_line"].to_numpy()
        for B in (B1MeanEffect(), B2Ridge(), B3RidgeBio()):
            t0 = time.time()
            B.fit(df, tr, val_idx=va, fz=fz, deg=deg)
            pred = B.predict(df, te)
            m = evaluate(pred, Y[te], deg[te], groups)["overall"]
            m["alpha"] = getattr(B, "alpha", None)
            m["fit_sec"] = round(time.time() - t0, 1)
            results[mode][B.name] = {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in m.items()}
            print(f"[{mode}] {B.name:16s} DEG50={m['pearson_deg50']:.3f} "
                  f"all={m['pearson_all']:.3f} dir50={m['direction_acc50']:.3f} "
                  f"({m['fit_sec']}s, n_test={len(te)})")
    out = RESULTS_DIR / f"baselines_{dataset}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    run(ap.parse_args().dataset)
