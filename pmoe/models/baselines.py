"""Non-deep baselines for perturbation LFC prediction.

Ported from v1 ``code/baselines.py`` (B1/B2/B3) with imports rewired to ``pmoe.*`` and the
val-selection metric inlined so the baselines have no hard dependency on ``pmoe.eval`` (which a
sibling agent owns). Adds the **GRN-propagation baseline** (audit item E): a pure GRN
message-passing predictor with no deep net, whose entire inductive bias is the GRN.

B1  Mean perturbation effect : per-drug mean LFC across training cell lines (global mean for
                               drugs unseen in train).
B2  Ridge                    : input = ChemBERTa + one-hot(cell_line) + ctrl_mean; output = LFC.
B3  Ridge + biology          : B2 features + target-gene baseline expr, target one-hot, target
                               pathway membership.
GRNPropagationBaseline       : seed an initial per-target signal learned on train, propagate it
                               along GRN edges for H hops with a least-squares-fitted per-hop scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmoe.config import SEED


# ---------- helpers ----------
def _stack(df: pd.DataFrame, col: str) -> np.ndarray:
    """Densify a list-column into (n, N); bool for deg_mask, float32 otherwise.

    Local copy of ``pmoe.data.loader.stack_arrays`` to keep baselines importable without the data
    package (and to work on inline synthetic frames in tests).
    """
    arr = np.stack([np.asarray(x) for x in df[col].to_numpy()])
    return arr.astype(bool) if col == "deg_mask" else arr.astype(np.float32)


def _deg_pearson(pred: np.ndarray, true: np.ndarray, k: int = 50) -> float:
    """Mean per-row Pearson r over the top-k genes by |true| (val-selection metric for ridge)."""
    n, N = true.shape
    k = min(k, N)
    rs = []
    for i in range(n):
        order = np.argsort(-np.abs(true[i]))[:k]
        a, b = pred[i, order], true[i, order]
        a = a - a.mean(); b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        rs.append(float((a * b).sum() / denom) if denom > 1e-12 else 0.0)
    return float(np.mean(rs)) if rs else 0.0


# ---------- shared feature building (self-contained) ----------
class Featurizer:
    """Builds B2/B3 design matrices from a conditions DataFrame.

    ``genes``/``chemberta``/``gene_pw`` may be passed explicitly (tests, or when the priors package
    is unavailable). Otherwise they are loaded lazily from ``pmoe.data`` / ``pmoe.priors``.
    """

    def __init__(self, dataset: str, df: pd.DataFrame, genes=None, chemberta=None, gene_pw=None):
        if genes is None:
            from pmoe.data.loader import load_genes
            genes = load_genes(dataset)
        self.genes = list(genes)
        self.gidx = {g: i for i, g in enumerate(self.genes)}

        self.cell_lines = sorted(df["cell_line"].unique().tolist())
        self._cl_index = {c: i for i, c in enumerate(self.cell_lines)}
        self.targets = sorted(df["target_gene"].fillna("").unique().tolist())
        self._tgt_index = {t: i for i, t in enumerate(self.targets)}

        if chemberta is None:
            from pmoe.priors.drugs import load_drug_feats
            dfeats = load_drug_feats(dataset).set_index("treatment")
            chemberta = {t: np.asarray(r) for t, r in dfeats["chemberta"].items()}
        self.chemberta = {t: np.asarray(v, np.float32) for t, v in chemberta.items()}
        self.cb_dim = len(next(iter(self.chemberta.values()))) if self.chemberta else 0

        if gene_pw is None:
            try:
                from pmoe.config import priors_dir
                pri = priors_dir(dataset) / "pathways.npz"
                gene_pw = sp.load_npz(pri).toarray() if pri.exists() else None
            except Exception:
                gene_pw = None
        self.gene_pw = None if gene_pw is None else (
            gene_pw.toarray() if sp.issparse(gene_pw) else np.asarray(gene_pw))
        self.ctrl = _stack(df, "ctrl_mean")

    def _onehot(self, keys, index):
        M = np.zeros((len(keys), len(index)), np.float32)
        for r, kk in enumerate(keys):
            j = index.get(kk)
            if j is not None:
                M[r, j] = 1.0
        return M

    def chem(self, sub: pd.DataFrame) -> np.ndarray:
        return np.stack([self.chemberta.get(t, np.zeros(self.cb_dim, np.float32))
                         for t in sub["treatment"]]).astype(np.float32)

    def base(self, df, idx) -> np.ndarray:
        """B2 features: chemberta + one-hot(cell_line) + ctrl_mean."""
        sub = df.iloc[idx]
        cb = self.chem(sub)
        cl = self._onehot(sub["cell_line"].tolist(), self._cl_index)
        ctrl = self.ctrl[idx]
        return np.concatenate([cb, cl, ctrl], axis=1).astype(np.float32)

    def bio(self, df, idx) -> np.ndarray:
        """B3 = B2 + target baseline expr + target one-hot + target pathway membership."""
        sub = df.iloc[idx]
        tgts = sub["target_gene"].fillna("").tolist()
        tgt_oh = self._onehot(tgts, self._tgt_index)
        tbe = np.array([[self.ctrl[i, self.gidx[t]]] if t in self.gidx else [0.0]
                        for i, t in zip(idx, tgts)], np.float32)
        if self.gene_pw is not None:
            tpw = np.stack([self.gene_pw[self.gidx[t]] if t in self.gidx
                            else np.zeros(self.gene_pw.shape[1]) for t in tgts]).astype(np.float32)
        else:
            tpw = np.zeros((len(idx), 0), np.float32)
        return np.concatenate([self.base(df, idx), tbe, tgt_oh, tpw], axis=1).astype(np.float32)


# ---------- baselines ----------
class B1MeanEffect:
    name = "B1_mean_effect"

    def fit(self, df, train_idx, **kw):
        Y = _stack(df, "lfc")
        train_idx = np.asarray(train_idx)
        self.global_mean = Y[train_idx].mean(0)
        self.per_drug = {}
        sub = df.iloc[train_idx]
        # positional row -> absolute Y row index
        pos = {r: train_idx[p] for p, r in enumerate(sub.index)}
        for drug, g in sub.groupby("treatment"):
            rows = np.array([pos[r] for r in g.index])
            self.per_drug[drug] = Y[rows].mean(0)
        return self

    def predict(self, df, idx):
        sub = df.iloc[idx]
        return np.stack([self.per_drug.get(t, self.global_mean) for t in sub["treatment"]])


class _RidgeBase:
    feat = "base"
    ALPHAS = [1.0, 10.0, 100.0, 1000.0]

    def __init__(self):
        self.alpha = self.ALPHAS[0]

    def _X(self, fz, df, idx):
        return getattr(fz, self.feat)(df, np.asarray(idx))

    def fit(self, df, train_idx, val_idx=None, fz=None, deg=None):
        from sklearn.linear_model import Ridge
        assert fz is not None, "_RidgeBase needs a Featurizer (fz=...)"
        self.fz = fz
        train_idx = np.asarray(train_idx)
        Xtr = self._X(fz, df, train_idx)
        Y = _stack(df, "lfc")
        Ytr = Y[train_idx]
        best, best_score = None, -1e9
        for a in self.ALPHAS:
            m = Ridge(alpha=a).fit(Xtr, Ytr)
            if val_idx is not None and len(val_idx):
                val_idx = np.asarray(val_idx)
                pv = m.predict(self._X(fz, df, val_idx))
                score = _deg_pearson(pv, Y[val_idx], k=50)
            else:
                score = 0.0
            if score > best_score:
                best, best_score, self.alpha = m, score, a
        self.model = best
        return self

    def predict(self, df, idx):
        return self.model.predict(self._X(self.fz, df, np.asarray(idx)))


class B2Ridge(_RidgeBase):
    name = "B2_ridge"
    feat = "base"


class B3RidgeBio(_RidgeBase):
    name = "B3_ridge_bio"
    feat = "bio"


# ---------- GRN-propagation baseline (audit item E) ----------
class GRNPropagationBaseline:
    """Pure GRN message-passing predictor (no deep net). Audit item E.

    Idea: a perturbation primarily hits its *target gene*; the transcriptomic response then spreads
    to the target's regulatory neighbourhood along GRN edges. We model LFC as the linear
    propagation of an initial per-target signal over the (row-normalized) GRN adjacency for H hops,
    with a single fitted per-hop scale vector solved in closed form (least squares) on the training
    conditions. The model's *entire* inductive bias is the GRN, so it isolates "does this GRN carry
    predictive structure" from "can the deep architecture exploit it".

    Fit
    ---
    1. ``seed[c]`` (n_train, N): a one-hot-ish initial signal for condition ``c`` placed on its
       target gene, scaled by the *learned per-target initial effect* -- the mean training LFC at
       the target gene for that target (so a target whose own gene moves +2 LFC seeds +2 there).
       Conditions with an unknown / off-panel target get a zero seed (fall back to global mean).
    2. Build the propagation operators ``P^0..P^H`` where ``P`` is the row-normalized symmetrized
       GRN adjacency (|w| for weighted GRNs). Hop ``0`` is the seed itself.
    3. For each hop ``h`` form features ``F_h = seed @ P^h`` (n_train, N) and solve, in closed form,
       for per-hop scalar weights ``a_h`` (and a global LFC offset ``mu``) minimising
       ``|| mu + sum_h a_h F_h  -  LFC_train ||^2`` via least squares (a tiny (H+2)-dim system).

    Predict: re-seed each query condition, propagate, apply the fitted ``a_h`` + ``mu``.
    """

    name = "GRN_propagation"

    def __init__(self, n_hops: int = 3, seed: int = SEED):
        self.n_hops = int(n_hops)
        self.seed = seed

    # -- internals --
    def _operator(self, grn, N) -> np.ndarray:
        """Row-normalized symmetric adjacency P (np.ndarray NxN). |w| for weighted GRNs."""
        A = grn.toarray() if sp.issparse(grn) else np.asarray(grn)
        A = np.abs(A.astype(np.float64))
        A = np.maximum(A, A.T)                     # undirected propagation
        np.fill_diagonal(A, 0.0)
        deg = A.sum(1, keepdims=True)
        deg[deg == 0] = 1.0
        return A / deg

    def _gene_index(self, df):
        # genes are addressed by column position; build target_gene -> column via meta if present,
        # else assume target_gene strings are like "g{j}" or fall back to a name->col map.
        return getattr(self, "_gidx", None)

    def _seed_matrix(self, df, idx) -> np.ndarray:
        sub = df.iloc[np.asarray(idx)]
        n = len(sub)
        S = np.zeros((n, self.N), np.float64)
        for r, t in enumerate(sub["target_gene"].fillna("").tolist()):
            j = self.gidx.get(t)
            if j is not None:
                S[r, j] = self.target_effect.get(t, 0.0)
        return S

    def _features(self, S) -> np.ndarray:
        """Stack hop features -> (n, H+1, N). Hop 0 = seed, hop h = seed @ P^h."""
        feats = [S]
        cur = S
        for _ in range(self.n_hops):
            cur = cur @ self.P
            feats.append(cur)
        return np.stack(feats, axis=1)             # (n, H+1, N)

    # -- API --
    def fit(self, df, train_idx, grn, val_idx=None, genes=None):
        train_idx = np.asarray(train_idx)
        Y = _stack(df, "lfc").astype(np.float64)
        self.N = Y.shape[1]

        # gene column index: explicit list, else meta, else infer "g{j}" naming, else order seen.
        if genes is None:
            genes = list(df.attrs.get("genes", [])) or [f"g{j}" for j in range(self.N)]
        self.gidx = {g: i for i, g in enumerate(genes)}

        self.P = self._operator(grn, self.N)
        self.global_mean = Y[train_idx].mean(0)

        # learned per-target initial effect = mean train LFC at the target's own gene column
        self.target_effect = {}
        sub = df.iloc[train_idx]
        pos = {r: train_idx[p] for p, r in enumerate(sub.index)}
        for t, g in sub.groupby(sub["target_gene"].fillna("")):
            j = self.gidx.get(t)
            if j is None:
                continue
            rows = np.array([pos[r] for r in g.index])
            self.target_effect[t] = float(Y[rows, j].mean())

        # build train features and solve closed-form per-hop scales + offset
        S = self._seed_matrix(df, train_idx)
        F = self._features(S)                      # (n, H+1, N)
        n = F.shape[0]
        # design: [1, F_0..F_H] flattened over (n*N); target = LFC flattened
        Phi = np.concatenate([np.ones((n, 1, self.N)), F], axis=1)   # (n, H+2, N)
        Phi = Phi.transpose(0, 2, 1).reshape(n * self.N, self.n_hops + 2)
        y = Y[train_idx].reshape(n * self.N)
        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        self.mu = float(coef[0])
        self.a = coef[1:]                          # (H+1,)
        return self

    def predict(self, df, idx) -> np.ndarray:
        idx = np.asarray(idx)
        S = self._seed_matrix(df, idx)
        F = self._features(S)                      # (n, H+1, N)
        pred = self.mu + np.tensordot(F, self.a, axes=([1], [0]))    # (n, N)
        return pred.astype(np.float32)


def count_params(m) -> int:
    """Total parameter count (works for torch modules; baselines have ~none)."""
    try:
        return sum(p.numel() for p in m.parameters())
    except AttributeError:
        return 0
