"""Build REAL GRN + pathway priors aligned to the Tahoe gene list (Ensembl IDs).

Inputs (downloaded separately):
  priors/grn_network.csv                  CollecTRI/DoRothEA TF->target (gene SYMBOLS) from decoupler
  priors/Ensembl2Reactome_All_Levels.txt  Reactome gene(Ensembl)->pathway
  tahoe100m/metadata/gene_vocabulary.jsonl  ensembl_id <-> gene_symbol map

Outputs (overwrite the fallback priors):
  priors/tahoe/grn_mask.npz, pathways.npz, pathway_names.txt, *.meta.json

Tahoe genes are Ensembl IDs, so Reactome (Ensembl-keyed) matches directly; the GRN (symbol-keyed)
is mapped to Ensembl via the vocabulary. This makes the real-data run a *fair* test of the GRN/
pathway inductive bias.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import DATA_ROOT
from loader import load_genes

PRIORS = DATA_ROOT / "data" / "priors"
VOCAB = DATA_ROOT / "data" / "tahoe100m" / "metadata" / "gene_vocabulary.jsonl"


def sym2ens_map():
    m = {}
    with open(VOCAB) as f:
        for line in f:
            r = json.loads(line)
            sym, ens = r.get("gene_symbol"), r.get("ensembl_id")
            if sym and ens and sym != ens:
                m[sym] = ens
    return m


def build_grn(dataset="tahoe"):
    genes = load_genes(dataset)
    gidx = {g: i for i, g in enumerate(genes)}
    s2e = sym2ens_map()
    # Prefer CollecTRI/DoRothEA CSV if present (decoupler export); else TRRUST v2 TSV (curated,
    # symbol-based, downloaded directly when omnipath is unreachable).
    csv = PRIORS / "grn_network.csv"
    trrust = PRIORS / "trrust_human.tsv"
    if csv.exists():
        net = pd.read_csv(csv)
        src_name = "CollecTRI/DoRothEA(real)"
    elif trrust.exists():
        net = pd.read_csv(trrust, sep="\t", header=None,
                          names=["source", "target", "mode", "pmid"])
        src_name = "TRRUST_v2(real)"
    else:
        raise FileNotFoundError("no GRN source (grn_network.csv or trrust_human.tsv)")
    cols = {c.lower(): c for c in net.columns}
    sc = cols.get("source") or cols.get("tf")
    tc = cols.get("target")
    rows, colsi = [], []
    n_hit = 0
    for s, t in zip(net[sc].astype(str), net[tc].astype(str)):
        es, et = s2e.get(s, s), s2e.get(t, t)   # symbol->ensembl (or pass through if already ENSG)
        if es in gidx and et in gidx:
            rows.append(gidx[es]); colsi.append(gidx[et]); n_hit += 1
    n = len(genes)
    m = sp.csr_matrix((np.ones(len(rows), bool), (rows, colsi)), shape=(n, n))
    out = PRIORS / dataset; out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "grn_mask.npz", m.astype(bool))
    (out / "grn_mask.meta.json").write_text(json.dumps(
        {"source": src_name, "n_edges": int(m.nnz),
         "net_edges": len(net), "mapped_edges": n_hit, "n_genes": n}, indent=2))
    print(f"GRN real: {m.nnz} edges among {n} Tahoe genes "
          f"(from {len(net)} network edges, density {m.nnz/n**2:.4f})")
    return m


def build_pathways(dataset="tahoe", max_pathways=40, min_genes=5):
    genes = load_genes(dataset)
    gidx = {g: i for i, g in enumerate(genes)}
    f = PRIORS / "Ensembl2Reactome_All_Levels.txt"
    pw_genes: dict[str, set] = {}
    pw_name: dict[str, str] = {}
    for line in f.read_text(errors="ignore").splitlines():
        p = line.split("\t")
        if len(p) < 6 or p[5] != "Homo sapiens":
            continue
        ens, rid, name = p[0], p[1], p[3]
        if ens in gidx:
            pw_genes.setdefault(rid, set()).add(gidx[ens]); pw_name[rid] = name
    # keep the largest pathways that have enough genes present
    top = sorted(((k, v) for k, v in pw_genes.items() if len(v) >= min_genes),
                 key=lambda kv: -len(kv[1]))[:max_pathways]
    names = [pw_name[k] for k, _ in top]
    n = len(genes)
    m = sp.lil_matrix((n, len(top)), dtype=bool)
    for p, (_, gs) in enumerate(top):
        for g in gs:
            m[g, p] = True
    out = PRIORS / dataset
    sp.save_npz(out / "pathways.npz", m.tocsr())
    (out / "pathway_names.txt").write_text("\n".join(names))
    (out / "pathways.meta.json").write_text(json.dumps(
        {"source": "Reactome(real)", "n_pathways": len(names),
         "coverage_genes": int((m.tocsr().sum(1) > 0).sum())}, indent=2))
    print(f"pathways real: P={len(names)}, genes covered={int((m.tocsr().sum(1)>0).sum())}/{n}")
    print("  top pathways:", names[:6])
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="tahoe")
    a = ap.parse_args()
    build_grn(a.dataset)
    build_pathways(a.dataset)
