"""Train/val/test splits with leakage control + cluster grouping for the bootstrap stats.

Three modes (as in the spec):
  unseen_drug      : hold out a fraction of drugs across all cell lines. Drugs are first
                     Butina-clustered on Morgan fingerprints so analogs (Tanimoto>0.8) never
                     straddle the train/test boundary.
  unseen_cell_line : hold out whole cell lines.
  unseen_both      : hold out (drug-cluster x cell-line) combos not seen together.

Ported from v1 ``code/splits.py``. New in v2: ``make_splits`` also returns a ``"groups"`` array
aligned to ``df`` rows (the per-condition cluster label) so :mod:`pmoe.eval.stats` can run the
cluster bootstrap (audit item C). Groups are the drug cluster for ``unseen_drug``/``unseen_both``
and the cell line for ``unseen_cell_line``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pmoe.config import SEED, Split

TANIMOTO_CUTOFF = 0.8

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - exercised only when rdkit is absent
    _HAVE_RDKIT = False


def _butina_clusters(smiles: list[str], cutoff: float = TANIMOTO_CUTOFF) -> dict[str, int]:
    """Map smiles -> cluster id; drugs with Tanimoto>cutoff land in the same cluster.

    Falls back to one-cluster-per-unique-smiles when rdkit is unavailable or there is < 2
    valid molecules to compare.
    """
    uniq = [s for s in dict.fromkeys(smiles) if s]
    if not _HAVE_RDKIT or len(uniq) < 2:
        return {s: i for i, s in enumerate(dict.fromkeys(smiles))}
    fps = []
    valid = []
    for s in uniq:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
        valid.append(s)
    n = len(fps)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - x for x in sims)
    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    out: dict[str, int] = {}
    for cid, members in enumerate(clusters):
        for idx in members:
            out[valid[idx]] = cid
    nxt = len(clusters)
    for s in smiles:  # anything unmapped (invalid smiles) -> own cluster
        if s not in out:
            out[s] = nxt
            nxt += 1
    return out


def _drug_cluster_col(df: pd.DataFrame) -> np.ndarray:
    """Per-row drug-cluster label (Butina/Tanimoto when smiles present, else drug name)."""
    smiles = df["smiles"].fillna("").tolist()
    if any(smiles):
        cmap = _butina_clusters(smiles)
        return np.array([cmap.get(s, -1) if s else -hash(t) % (1 << 30)
                         for s, t in zip(smiles, df["treatment"])])
    # no smiles -> cluster by drug name
    names = df["treatment"].astype("category")
    return names.cat.codes.to_numpy()


def _split_val(train_idx: np.ndarray, val_frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation slice out of the training pool (random, seeded)."""
    perm = rng.permutation(train_idx)
    n_val = max(1, int(len(perm) * val_frac))
    return perm[n_val:], perm[:n_val]


def make_splits(df: pd.DataFrame, mode: str | Split, seed: int = SEED, val_frac: float = 0.1,
                test_frac: float = 0.2) -> dict[str, np.ndarray]:
    """Build leakage-controlled index splits + a cluster-group label per row.

    Args:
        df: conditions DataFrame (needs ``cell_line``, ``treatment``, ``smiles``).
        mode: one of the :class:`~pmoe.config.Split` values (str or enum).
        seed: RNG seed (defaults to :data:`~pmoe.config.SEED`).
        val_frac: fraction of the training pool carved off for validation.
        test_frac: fraction of clusters/cell-lines held out for the test set.

    Returns:
        dict with index arrays ``"train"``, ``"val"``, ``"test"`` into ``df`` rows, plus
        ``"groups"``: an ``(len(df),)`` array of cluster labels aligned to df rows (drug cluster
        for unseen_drug/unseen_both, cell line for unseen_cell_line) used by the cluster bootstrap.
    """
    mode = Split(mode).value if not isinstance(mode, str) else mode
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(df))

    if mode == "unseen_drug":
        clusters = _drug_cluster_col(df)
        groups = clusters
        uniq = np.unique(clusters)
        rng.shuffle(uniq)
        n_test = max(1, int(len(uniq) * test_frac))
        test_clusters = set(uniq[:n_test].tolist())
        is_test = np.array([c in test_clusters for c in clusters])
        test_idx = idx_all[is_test]
        train_pool = idx_all[~is_test]
        train_idx, val_idx = _split_val(train_pool, val_frac, rng)

    elif mode == "unseen_cell_line":
        lines = df["cell_line"].to_numpy()
        # groups for the cluster bootstrap are the cell lines, encoded to ints aligned to rows
        groups = pd.Series(lines).astype("category").cat.codes.to_numpy()
        uniq = np.unique(lines)
        rng.shuffle(uniq)
        n_test = max(1, int(round(len(uniq) * test_frac)))
        test_lines = set(uniq[:n_test].tolist())
        is_test = np.array([l in test_lines for l in lines])
        test_idx = idx_all[is_test]
        train_pool = idx_all[~is_test]
        train_idx, val_idx = _split_val(train_pool, val_frac, rng)

    elif mode == "unseen_both":
        clusters = _drug_cluster_col(df)
        groups = clusters
        lines = df["cell_line"].to_numpy()
        uc = np.unique(clusters)
        ul = np.unique(lines)
        rng.shuffle(uc)
        rng.shuffle(ul)
        test_clusters = set(uc[: max(1, int(len(uc) * test_frac))].tolist())
        test_lines = set(ul[: max(1, int(round(len(ul) * test_frac)))].tolist())
        # test = held-out drug AND held-out cell line; train = neither (strict, no leakage)
        is_test = np.array([c in test_clusters and l in test_lines
                            for c, l in zip(clusters, lines)])
        is_train_pool = np.array([c not in test_clusters and l not in test_lines
                                  for c, l in zip(clusters, lines)])
        test_idx = idx_all[is_test]
        train_pool = idx_all[is_train_pool]
        train_idx, val_idx = _split_val(train_pool, val_frac, rng)
    else:
        raise ValueError(f"unknown split mode: {mode}")

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
        "groups": np.asarray(groups),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    for cl in ["A549", "MCF7", "K562", "A375"]:
        for d in range(40):
            rows.append({"cell_line": cl, "treatment": f"drug{d}", "smiles": "", "dose": 1.0})
    _df = pd.DataFrame(rows)
    for m in ["unseen_drug", "unseen_cell_line", "unseen_both"]:
        s = make_splits(_df, m)
        print(m, {k: len(v) for k, v in s.items()},
              "RDKit" if _HAVE_RDKIT else "no-rdkit")
