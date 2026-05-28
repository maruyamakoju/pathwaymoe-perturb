"""Gene -> pathway multi-hot matrix aligned to a dataset's gene list.

Source: Reactome (``Ensembl2Reactome_All_Levels.txt`` in ``PRIORS_ROOT``), collapsed to the
largest top-level pathways for MoE expert grouping. Offline fallback: deterministic seeded random
assignment to ``N_PATHWAYS_FALLBACK`` groups.

Ported from v1 ``code/build_real_priors.build_pathways`` (Ensembl-keyed Reactome) and
``code/pathways.py`` (fallback). Output: ``priors/<dataset>/pathways.npz`` (sparse bool NxP)
plus ``pathway_names.txt`` and ``pathways.meta.json``.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import scipy.sparse as sp

from pmoe.config import PRIORS_ROOT, SEED, priors_dir
from pmoe.data.loader import load_genes

N_PATHWAYS_FALLBACK = 20
MIN_GENES = 5


def _from_reactome(genes: list[str], max_pathways: int, min_genes: int = MIN_GENES):
    """Build (N,P) bool CSR from the Ensembl-keyed Reactome dump, keeping the largest pathways.

    Pathway selection ties (same gene count) are broken by reactome_id so re-builds are
    deterministic across line-order shuffles in the Reactome dump.
    """
    gidx = {g: i for i, g in enumerate(genes)}
    rfile = PRIORS_ROOT / "Ensembl2Reactome_All_Levels.txt"
    pw_genes: dict[str, set] = {}
    pw_name: dict[str, str] = {}
    # columns: ensembl_id, reactome_id, url, name, evidence, species
    for line in rfile.read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 6 or p[5] != "Homo sapiens":
            continue
        ens, rid, name = p[0], p[1], p[3]
        if ens in gidx:
            pw_genes.setdefault(rid, set()).add(gidx[ens])
            pw_name[rid] = name
    top = sorted(((k, v) for k, v in pw_genes.items() if len(v) >= min_genes),
                 key=lambda kv: (-len(kv[1]), kv[0]))[:max_pathways]
    if not top:
        raise RuntimeError("no Reactome pathways matched the gene list")
    names = [pw_name[k] for k, _ in top]
    m = sp.lil_matrix((len(genes), len(top)), dtype=bool)
    for p, (_, gs) in enumerate(top):
        for g in gs:
            m[g, p] = True
    return m.tocsr(), names, "Reactome(real)"


def _fallback(genes: list[str], n_pathways: int = N_PATHWAYS_FALLBACK, seed: int = SEED):
    """Deterministic random gene->pathway assignment when Reactome is unavailable."""
    rng = np.random.default_rng(seed)
    n = len(genes)
    m = sp.lil_matrix((n, n_pathways), dtype=bool)
    primary = rng.integers(0, n_pathways, n)
    for g in range(n):
        m[g, primary[g]] = True
        if rng.random() < 0.3:
            m[g, rng.integers(0, n_pathways)] = True
    names = [f"pathway_{i}" for i in range(n_pathways)]
    return m.tocsr(), names, "fallback"


def build_pathways(dataset: str, max_pathways: int = 40) -> sp.csr_matrix:
    """Build the gene->pathway (N,P) bool matrix and persist it to ``priors/<dataset>/``.

    Uses Reactome if ``Ensembl2Reactome_All_Levels.txt`` is present, else a deterministic
    seeded fallback. On fallback a :class:`UserWarning` is emitted so a downstream metric is
    never silently based on a random pathway assignment.
    """
    genes = load_genes(dataset)
    rfile = PRIORS_ROOT / "Ensembl2Reactome_All_Levels.txt"
    if rfile.exists():
        # Parsing / matching errors should propagate; only the missing-file case falls back.
        m, names, source = _from_reactome(genes, max_pathways)
    else:
        warnings.warn(
            f"Reactome dump not found at {rfile}; using seeded random pathway assignment. "
            "Downstream MoE pathway prior is NOT real biology.",
            UserWarning, stacklevel=2,
        )
        m, names, source = _fallback(genes)

    pri = priors_dir(dataset)
    sp.save_npz(pri / "pathways.npz", m)
    (pri / "pathway_names.txt").write_text("\n".join(names))
    (pri / "pathways.meta.json").write_text(json.dumps(
        {"source": source, "n_pathways": len(names),
         "coverage_genes": int((m.sum(1) > 0).sum())}, indent=2))
    return m


def load_pathways(dataset: str) -> tuple[sp.csr_matrix, list[str]]:
    """Load cached (matrix, names); build if missing."""
    pri = priors_dir(dataset)
    npz = pri / "pathways.npz"
    if not npz.exists():
        m = build_pathways(dataset)
    else:
        m = sp.load_npz(npz)
    names_f = pri / "pathway_names.txt"
    names = names_f.read_text().splitlines() if names_f.exists() else [f"pathway_{i}" for i in range(m.shape[1])]
    return m.tocsr(), names
