"""Build a gene -> pathway multi-hot matrix aligned to a dataset's gene list.

Source: Reactome (Ensembl2Reactome_All_Levels.txt) if present in priors/, collapsed to top-level
pathways for MoE expert grouping. Offline fallback: seeded random assignment to N_PATHWAYS groups.

Output: priors/<dataset>/pathways.npz (sparse bool NxP) + pathway_names.txt.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from config import DATA_ROOT
from loader import load_genes

N_PATHWAYS_FALLBACK = 20


def from_reactome(genes, reactome_file: Path, max_pathways=64):
    gidx = {g: i for i, g in enumerate(genes)}
    pw_genes: dict[str, set] = {}
    # file columns: source_id, reactome_id, url, name, evidence, species  (gene-symbol variants vary)
    for line in reactome_file.read_text(errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        gene, pw_name = parts[0], parts[3]
        if gene in gidx:
            pw_genes.setdefault(pw_name, set()).add(gidx[gene])
    # keep the largest pathways as experts
    top = sorted(pw_genes.items(), key=lambda kv: -len(kv[1]))[:max_pathways]
    names = [k for k, _ in top]
    m = sp.lil_matrix((len(genes), len(names)), dtype=bool)
    for p, (_, gs) in enumerate(top):
        for g in gs:
            m[g, p] = True
    return m.tocsr(), names, f"reactome:{len(names)}_pathways"


def fallback(genes, n_pathways=N_PATHWAYS_FALLBACK, seed=1337):
    rng = np.random.default_rng(seed); n = len(genes)
    m = sp.lil_matrix((n, n_pathways), dtype=bool)
    primary = rng.integers(0, n_pathways, n)
    for g in range(n):
        m[g, primary[g]] = True
        if rng.random() < 0.3:
            m[g, rng.integers(0, n_pathways)] = True
    names = [f"pathway_{i}" for i in range(n_pathways)]
    return m.tocsr(), names, "fallback"


def build(dataset: str):
    genes = load_genes(dataset)
    rfile = DATA_ROOT / "data" / "priors" / "Ensembl2Reactome_All_Levels.txt"
    try:
        if rfile.exists():
            m, names, source = from_reactome(genes, rfile)
            if m.shape[1] == 0:
                raise RuntimeError("no pathways matched")
        else:
            raise FileNotFoundError("reactome file absent")
    except Exception as e:
        m, names, source = fallback(genes)
        source = f"fallback({type(e).__name__})"
    pri = DATA_ROOT / "data" / "priors" / dataset; pri.mkdir(parents=True, exist_ok=True)
    sp.save_npz(pri / "pathways.npz", m)
    (pri / "pathway_names.txt").write_text("\n".join(names))
    (pri / "pathways.meta.json").write_text(json.dumps(
        {"source": source, "n_pathways": len(names)}, indent=2))
    print(f"pathways[{dataset}] source={source} P={len(names)}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="synthetic")
    build(ap.parse_args().dataset)
