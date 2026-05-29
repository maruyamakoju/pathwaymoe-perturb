"""Gene regulatory network (GRN) priors — the study's INDEPENDENT VARIABLE.

This is where the data-leakage flaw from ``AUDIT.md`` (items A & F) is fixed:

* Data-derived variants (``coexpr``, ``coexpr_lfc``) are built from **training conditions only**
  (``df.iloc[train_idx]``). They RAISE ``ValueError`` if ``train_idx is None`` so a leaky GRN can
  never be built by accident. The ``.meta.json`` records ``leakage_safe=True`` only in that case.
* ``trrust_weighted`` keeps TRRUST's sign (Activation=+w, Repression=-w, Unknown=+/-small) as a
  float CSR instead of binarizing/symmetrizing away the information (audit item F).

All variants live in the SAME gene space and (where applicable) are matched to the same edge
count (TRRUST's, ``match_edges``) so the experiment isolates GRN *quality*, not density.

Ported from v1 ``code/build_grn_variants.py`` and ``code/build_real_priors.py``.

Testability: ``build_grn`` accepts optional ``genes=`` (a gene-symbol list) and ``df=`` (a
conditions DataFrame) so unit tests need no real dataset on disk. When ``genes`` is given, the
TRRUST symbol->Ensembl mapping / vocabulary is skipped for the data-derived variants, which only
need the gene count and the conditions arrays.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmoe.config import (
    DATA_DERIVED,
    PRIORS_ROOT,
    SEED,
    GRNVariant,
    grn_filename,
    priors_dir,
)
from pmoe.data.loader import load_conditions, load_genes, stack_arrays

VOCAB = PRIORS_ROOT.parent / "tahoe100m" / "metadata" / "gene_vocabulary.jsonl"
TRRUST_TSV = PRIORS_ROOT / "trrust_human.tsv"
# CollecTRI signed regulons (45,856 edges / 1,183 TFs). Downloaded from Zenodo record 8192729
# (CollecTRI_regulons.csv) because omnipathdb.org — the package's normal source — is unreachable.
# Columns: source,target,weight(+1 activation / -1 repression),resources,references,sign_decision.
COLLECTRI_CSV = PRIORS_ROOT / "CollecTRI_regulons.csv"

# weights for trrust_weighted edges keyed by the TRRUST `mode` column
_W_ACT = 1.0
_W_REP = -1.0
_W_UNK = 0.25   # magnitude for Unknown; sign decided deterministically per-edge


# --------------------------------------------------------------------------- helpers
def _sym2ens() -> dict[str, str]:
    """Map gene symbol -> Ensembl id from the Tahoe gene vocabulary."""
    m: dict[str, str] = {}
    with open(VOCAB) as f:
        for line in f:
            r = json.loads(line)
            s, e = r.get("gene_symbol"), r.get("ensembl_id")
            if s and e and s != e:
                m[s] = e
    return m


def _read_trrust() -> pd.DataFrame:
    """Read TRRUST v2 (source, target, mode, pmid)."""
    return pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                       names=["source", "target", "mode", "pmid"])


def _gidx(genes: list[str]) -> dict[str, int]:
    return {g: i for i, g in enumerate(genes)}


def _trrust_rows(genes: list[str]):
    """Return list of (i, j, mode) mapped into the gene space (symbol->Ensembl, self-loops dropped)."""
    gidx = _gidx(genes)
    s2e = _sym2ens()
    net = _read_trrust()
    rows = []
    seen = set()
    for s, t, mode in zip(net["source"].astype(str), net["target"].astype(str), net["mode"].astype(str)):
        es, et = s2e.get(s, s), s2e.get(t, t)
        if es in gidx and et in gidx and es != et:
            i, j = gidx[es], gidx[et]
            if (i, j) not in seen:
                seen.add((i, j))
                rows.append((i, j, mode))
    return rows


def _trrust_edges(genes: list[str]) -> set[tuple[int, int]]:
    return {(i, j) for i, j, _ in _trrust_rows(genes)}


def _build_csr(n: int, edges, dtype=bool) -> sp.csr_matrix:
    """Build an (n,n) CSR from an iterable of (i, j) or (i, j, w)."""
    edges = list(edges)
    if not edges:
        return sp.csr_matrix((n, n), dtype=dtype)
    if len(edges[0]) == 3:
        r, c, w = zip(*edges)
        data = np.asarray(w, dtype=dtype if dtype != bool else np.float32)
    else:
        r, c = zip(*edges)
        data = np.ones(len(edges), dtype=dtype)
    return sp.csr_matrix((data, (list(r), list(c))), shape=(n, n), dtype=dtype)


def _topk_corr_edges(M: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Top-k absolute-correlation gene pairs from M (rows=samples, cols=genes), symmetrized.

    Identical numerics to v1 ``build_grn_variants._topk_corr_edges``.
    """
    M = M.astype(np.float32)
    M = M - M.mean(0, keepdims=True)
    sd = M.std(0) + 1e-6
    C = (M.T @ M) / M.shape[0] / np.outer(sd, sd)
    np.fill_diagonal(C, 0.0)
    A = np.abs(C)
    iu = np.triu_indices_from(A, k=1)
    vals = A[iu]
    if k >= len(vals):
        sel = np.arange(len(vals))
    else:
        sel = np.argpartition(-vals, k)[:k]
    edges: set[tuple[int, int]] = set()
    for s in sel:
        i, j = int(iu[0][s]), int(iu[1][s])
        edges.add((i, j))
        edges.add((j, i))  # symmetric
    return edges


def _random_edges(n: int, n_edges: int, seed: int) -> set[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    rand: set[tuple[int, int]] = set()
    while len(rand) < n_edges:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j:
            rand.add((i, j))
    return rand


def _collectri_rows(genes: list[str]):
    """Return (i, j, signed-weight) CollecTRI edges mapped into the gene space.

    Symbols are mapped to Ensembl via the Tahoe vocabulary (same as TRRUST); self-loops and
    duplicate (i, j) pairs are dropped. ``weight`` is +1.0 (activation) / -1.0 (repression)
    from the CollecTRI ``weight`` column.
    """
    gidx = _gidx(genes)
    s2e = _sym2ens()
    net = pd.read_csv(COLLECTRI_CSV, usecols=["source", "target", "weight"])
    rows = []
    seen = set()
    for s, t, w in zip(net["source"].astype(str), net["target"].astype(str), net["weight"]):
        es, et = s2e.get(s, s), s2e.get(t, t)
        if es in gidx and et in gidx and es != et:
            i, j = gidx[es], gidx[et]
            if (i, j) not in seen:
                seen.add((i, j))
                rows.append((i, j, float(w)))
    return rows


def _collectri_edges(genes: list[str]) -> set[tuple[int, int]]:
    return {(i, j) for i, j, _ in _collectri_rows(genes)}


def _trrust_weighted_rows(genes: list[str]):
    """(i, j, signed-weight) edges from TRRUST mode column for the weighted variant."""
    rows = []
    for i, j, mode in _trrust_rows(genes):
        m = mode.strip().lower()
        if m == "activation":
            w = _W_ACT
        elif m == "repression":
            w = _W_REP
        else:  # Unknown / anything else -> small, deterministically signed
            w = _W_UNK if ((i + j) % 2 == 0) else -_W_UNK
        rows.append((i, j, w))
    return rows


# --------------------------------------------------------------------------- main API
def build_grn(
    dataset: str,
    variant: "GRNVariant | str",
    *,
    train_idx: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    match_edges: int | None = None,
    seed: int = SEED,
    split: "str | None" = None,
    genes: list[str] | None = None,
) -> sp.csr_matrix:
    """Build a GRN prior for ``variant`` and persist it (npz + .meta.json).

    Parameters
    ----------
    dataset : dataset name (used for paths and, if ``genes``/``df`` omitted, to load them).
    variant : a :class:`~pmoe.config.GRNVariant` (or its string value).
    train_idx : row indices of TRAIN conditions. **Required** for ``coexpr`` / ``coexpr_lfc``;
        those raise ``ValueError`` if it is ``None`` (the leakage guard).
    df : conditions DataFrame; loaded from disk if omitted. Used only by the data-derived variants.
    match_edges : edge budget for ``random``. Defaults to the TRRUST edge count for this gene space
        (fair density). For ``coexpr*`` the budget is ``match_edges`` (default TRRUST count).
    seed : RNG seed (random variant).
    split : split name (e.g. ``"unseen_drug"``); makes data-derived files split-specific
        (``grn_coexpr_lfc__unseen_drug.npz``) as required by audit A. Ignored for other variants.
    genes : optional explicit gene list (for tests); loaded from disk if omitted.

    Returns
    -------
    scipy.sparse.csr_matrix : (N, N). ``bool`` for topology variants, ``float32`` for
    ``trrust_weighted``.
    """
    variant = GRNVariant(variant)
    # Split is also a str-Enum; normalise to its plain value so grn_filename names files
    # like grn_coexpr_lfc__unseen_drug.npz (not ...__Split.UNSEEN_DRUG.npz).
    if split is not None and hasattr(split, "value"):
        split = split.value
    if genes is None:
        genes = load_genes(dataset)
    n = len(genes)

    leakage_safe = True   # topology/curated variants carry no test info
    save_split = split

    if variant == GRNVariant.NONE:
        m = sp.csr_matrix((n, n), dtype=bool)
        source = "none"

    elif variant == GRNVariant.TRRUST:
        m = _build_csr(n, _trrust_edges(genes), dtype=bool)
        source = "TRRUST_v2"

    elif variant == GRNVariant.TRRUST_WEIGHTED:
        m = _build_csr(n, _trrust_weighted_rows(genes), dtype=np.float32)
        source = "TRRUST_v2_signed"

    elif variant == GRNVariant.RANDOM:
        n_edges = match_edges if match_edges is not None else max(len(_trrust_edges(genes)), 100)
        m = _build_csr(n, _random_edges(n, n_edges, seed), dtype=bool)
        source = "random_matched_density"

    elif variant == GRNVariant.RANDOM_DENSE:
        # density control for CollecTRI: same edge count, random topology
        n_edges = match_edges if match_edges is not None else max(len(_collectri_edges(genes)), 100)
        m = _build_csr(n, _random_edges(n, n_edges, seed), dtype=bool)
        source = "random_matched_collectri_density"

    elif variant == GRNVariant.COLLECTRI:
        m = _build_csr(n, _collectri_edges(genes), dtype=bool)
        source = "CollecTRI"

    elif variant == GRNVariant.COLLECTRI_WEIGHTED:
        m = _build_csr(n, _collectri_rows(genes), dtype=np.float32)
        source = "CollecTRI_signed"

    elif variant == GRNVariant.GROUND_TRUTH:
        gt = sp.load_npz(priors_dir(dataset) / "grn_mask.npz").tocoo()
        edges = set(zip(gt.row.tolist(), gt.col.tolist()))
        m = _build_csr(n, edges, dtype=bool)
        source = "synthetic_ground_truth"

    elif variant in DATA_DERIVED:
        # ---- LEAKAGE GUARD (audit A) ----
        if train_idx is None:
            raise ValueError(
                f"variant {variant.value} is data-derived and MUST be built from training "
                "conditions only; pass train_idx (got None)."
            )
        if df is None:
            df = load_conditions(dataset)
        train_idx = np.asarray(train_idx)
        df_train = df.iloc[train_idx]                       # TRAIN-ONLY rows
        col = "pert_mean" if variant == GRNVariant.COEXPR else "lfc"
        M = stack_arrays(df_train, col)
        if match_edges is None:
            match_edges = max(len(_trrust_edges(genes)), 100)
        edges = _topk_corr_edges(M, match_edges // 2)
        m = _build_csr(n, edges, dtype=bool)
        source = f"coexpr_{'expression' if variant == GRNVariant.COEXPR else 'LFC_response'}_train_only"

    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown variant {variant!r}")

    _save(dataset, variant, m, source, leakage_safe, save_split)
    return m


def _save(dataset, variant, m, source, leakage_safe, split):
    """Persist npz + meta json under priors_dir(dataset)/grn_filename(variant, split)."""
    if variant == GRNVariant.NONE:
        return   # no file for the empty GRN
    # NOTE: GRNVariant subclasses str, so pass .value (config.grn_filename treats str args
    # literally and would otherwise embed "GRNVariant.COEXPR_LFC" in the filename).
    fname = grn_filename(variant.value, split if variant in DATA_DERIVED else None)
    out = priors_dir(dataset)
    sp.save_npz(out / fname, m)
    meta_name = fname[:-len(".npz")] + ".meta.json"
    (out / meta_name).write_text(json.dumps({
        "variant": variant.value,
        "source": source,
        "n_edges": int(m.nnz),
        "n_genes": int(m.shape[0]),
        "weighted": bool(m.dtype != bool),
        "leakage_safe": bool(leakage_safe),
        "split": (str(split) if split is not None else None),
    }, indent=2))


def load_grn(dataset: str, variant: "GRNVariant | str", split=None) -> sp.csr_matrix | None:
    """Load a saved GRN. Returns ``None`` for the empty ``none`` variant or a missing file.

    Float CSR for ``trrust_weighted``; bool CSR otherwise.
    """
    variant = GRNVariant(variant)
    if split is not None and hasattr(split, "value"):
        split = split.value
    if variant == GRNVariant.NONE:
        return None
    fname = grn_filename(variant.value, split if variant in DATA_DERIVED else None)
    p = priors_dir(dataset) / fname
    if not p.exists():
        return None
    return sp.load_npz(p).tocsr()
