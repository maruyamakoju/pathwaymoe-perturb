"""Synthetic Tahoe-shaped perturbation dataset with KNOWN causal structure.

We embed the exact generative process the model is designed to exploit:
  LFC = on-target effect (scaled by cell-line target baseline)        # context
        propagated through a signed gene-regulatory network (GRN)     # propagation
        with edges concentrated within Reactome-like pathways         # modularity
        + compensatory pathway feedback + noise.

Because the GRN, pathway membership, and drug targets are ground truth, this dataset is a real
test of the thesis: a GRN-/pathway-constrained model SHOULD beat a linear ridge baseline here,
and we can verify the interpretability story (drug -> expected pathway/experts) exactly.

Writes (dataset name "synthetic"):
  processed/conditions.parquet, genes.txt, meta.json
  ../priors/synthetic/{grn_mask.npz, pathways.npz, pathway_names.txt, drug_feats.parquet, *.meta.json}
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmoe.config import DATA_ROOT, SEED, dataset_dir

# A curated library of real drugs (valid SMILES) with a signaling/oncology bias so the
# interpretability story is meaningful. (drug_name, SMILES, pathway_tag, sign)
# sign: -1 inhibitor (knocks target down), +1 agonist.
DRUGS = [
    ("trametinib", "CC1=C2C(=C(N(C1=O)C)NC3=C(C=C(C=C3)I)F)C(=O)N(C(=O)N2C4=CC=CC=C4)C5CC5", "MAPK", -1),
    ("selumetinib", "CN1C=NC2=C1C(=C(C=C2)NC3=C(C=C(C=C3)Br)Cl)C(=O)NOCCO", "MAPK", -1),
    ("dabrafenib", "CC(C)(C)C1=NC(=C(S1)C2=NC(=NC=C2)N)C3=C(C(=CC=C3)NS(=O)(=O)C4=C(C=CC=C4F)F)F", "MAPK", -1),
    ("vemurafenib", "CCCS(=O)(=O)NC1=CC=C(C(=C1)F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl", "MAPK", -1),
    ("erlotinib", "COCCOC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC", "RTK", -1),
    ("gefitinib", "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4", "RTK", -1),
    ("lapatinib", "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC=CC=C5F)Cl", "RTK", -1),
    ("imatinib", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", "RTK", -1),
    ("dasatinib", "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO", "RTK", -1),
    ("nilotinib", "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1", "RTK", -1),
    ("sorafenib", "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F", "RTK", -1),
    ("sunitinib", "CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C", "RTK", -1),
    ("palbociclib", "CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C", "CellCycle", -1),
    ("ribociclib", "CN(C)C(=O)C1=CC2=CN=C(N=C2N1C3CCCC3)NC4=NC=C(C=C4)N5CCNCC5", "CellCycle", -1),
    ("abemaciclib", "CCN1CCN(CC1)CC2=CC=C(N=C2)NC3=NC=C(C(=N3)C4=CC5=C(C=C4)N=C(N5C(C)C)C)F", "CellCycle", -1),
    ("alisertib", "COC1=C(C=C2C(=C1)C=CC3=C2C(=O)N(C(=N3)C4=CC=CC=C4F)CC(=O)O)OC", "CellCycle", -1),
    ("paclitaxel", "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C", "CellCycle", -1),
    ("docetaxel", "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1O)OC(=O)C(C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)(CO4)OC(=O)C)O)C)O", "CellCycle", -1),
    ("vincristine", "CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C78CCN9C7C(C=CC9)(C(C(C8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O", "CellCycle", -1),
    ("doxorubicin", "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O", "DNADamage", -1),
    ("etoposide", "CC1OCC2C(O1)OC3C(C2O)COC3C4=CC5=C(C=C4)OCO5", "DNADamage", -1),
    ("cisplatin", "N.N.Cl[Pt]Cl", "DNADamage", -1),
    ("olaparib", "C1CC1C(=O)N2CCN(CC2)C(=O)C3=CC=CC(=C3CC4=NNC(=O)C5=CC=CC=C45)F", "DNADamage", -1),
    ("nutlin-3", "CC(C)OC1=CC(=CC=C1C2=NC(C(N2C(=O)N3CCNC(=O)C3)C4=CC=C(C=C4)Cl)C5=CC=C(C=C5)Cl)OC(C)C", "P53", +1),
    ("MG132", "CC(C)CC(C=O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)OCC1=CC=CC=C1", "Proteostasis", -1),
    ("bortezomib", "CC(C)CC(B(O)O)NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2", "Proteostasis", -1),
    ("carfilzomib", "CC(C)CC(C(=O)NC(CC1=CC=CC=C1)C(=O)NC(CC2=CC=CC=C2)C(=O)NC(CC(C)C)C3(CO3)C)NC(=O)CN4CCOCC4", "Proteostasis", -1),
    ("MK-2206", "C1CC1C2=CC=C(C=C2)C3=CC=C4C(=C3)C(=NN4)C5=CC=CC=C5N", "PI3K_AKT", -1),
    ("ipatasertib", "CC(C(=O)N1CCC(CC1)C2=NC3=C(N2C(C)C)C=CC(=C3)Cl)NC4=NC=C(C=C4)Cl", "PI3K_AKT", -1),
    ("alpelisib", "CC1=C(SC(=N1)NC(=O)C2CCCN2C(=O)C(C)(C)C(F)(F)F)C3=CC(=NC=C3)C(F)(F)F", "PI3K_AKT", -1),
    ("rapamycin", "CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)O)C)C)O)OC)C)C)C)OC", "mTOR", -1),
    ("everolimus", "CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)OCCO)C)C)O)OC)C)C)C)OC", "mTOR", -1),
    ("ABT-199", "CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)CN3CCN(CC3)C4=CC=C(C=C4)C(=O)NS(=O)(=O)C5=CC(=C(C=C5)NCC6CCOCC6)[N+](=O)[O-])C", "Apoptosis", -1),
    ("ABT-737", "CN(C)CCC(CSC1=CC=CC=C1)NC2=CC(=C(C=C2)S(=O)(=O)NC(=O)C3=CC=C(C=C3)N4CCN(CC4)CC5=CC=CC=C5)[N+](=O)[O-]", "Apoptosis", -1),
    ("ruxolitinib", "C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3", "JAK_STAT", -1),
    ("tofacitinib", "CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N", "JAK_STAT", -1),
    ("fedratinib", "CCS(=O)(=O)NCCCNC1=CC=C(C=C1)C2=CN=C(N=C2)NC3=CC=C(C=C3)OC(C)C", "JAK_STAT", -1),
    ("INCB18424", "C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3", "JAK_STAT", -1),
    ("givinostat", "CC(C)NCC1=CC=C(C=C1)C2=CC=C(C=C2)COC(=O)NC3=CC=C(C=C3)C(=O)NO", "Epigenetic", -1),
    ("vorinostat", "C1=CC=C(C=C1)NC(=O)CCCCCCC(=O)NO", "Epigenetic", -1),
    ("panobinostat", "CC1=C(C=C2C(=C1)NC=C2CCNCC3=CC=C(C=C3)C=CC(=O)NO)C", "Epigenetic", -1),
    ("entinostat", "CC(C)NCC1=CC=C(C=C1)COC(=O)NCC2=CC=C(C=C2)C(=O)NC3=CC=CC=C3N", "Epigenetic", -1),
    ("JQ1", "Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2", "Epigenetic", -1),
    ("tunicamycin", "CC(C)CCCCCCCCCC=CC(=O)NC1C(C(C(OC1OC2C(C(C(C(O2)CO)O)O)NC(=O)C)CO)O)O", "UPR", -1),
    ("thapsigargin", "CCCCCCCC(=O)OC1CC2(C(C3(C(C2OC(=O)C)C(=C(C3OC(=O)CCC)C)OC(=O)C)C)C(C1(C)OC(=O)C)O)C", "UPR", -1),
    ("metformin", "CN(C)C(=N)NC(=N)N", "Metabolism", -1),
    ("2DG", "C(C1C(C(C(C(O1)O)O)O)O)O", "Metabolism", -1),
]

PATHWAY_TAGS = ["MAPK","RTK","CellCycle","DNADamage","P53","Proteostasis","PI3K_AKT","mTOR",
                "Apoptosis","JAK_STAT","Epigenetic","UPR","Metabolism","Translation","Inflammation",
                "Hypoxia","Cholesterol","OxStress","WntSignaling","Hippo"]
CELL_LINES = ["A549","MCF7","K562","A375","HCT116","HL60","PC9","SKBR3","U937","HepG2"]
DOSES = [0.1, 1.0, 10.0]


def _validate_drugs():
    try:
        from rdkit import Chem
    except Exception:
        return [(n, s, p, sg) for (n, s, p, sg) in DRUGS]  # cannot validate; keep all
    good = []
    for n, s, p, sg in DRUGS:
        if Chem.MolFromSmiles(s) is not None:
            good.append((n, s, p, sg))
    return good


def build(n_genes=2000, n_pathways=20, hops=3, seed=SEED, n_cells_mean=200,
          unique_targets=False, n_extra_drugs=0):
    """unique_targets=True: each drug hits a DISTINCT TF target (no shared hubs). Holding out a drug
    then holds out its target's direct effect, so predicting it REQUIRES propagating along the GRN
    from that (novel) target to known genes -> the GRN becomes essential (positive-control regime)."""
    rng = np.random.default_rng(seed)
    n_pathways = min(n_pathways, len(PATHWAY_TAGS))
    pw_names = PATHWAY_TAGS[:n_pathways]
    genes = [f"G{ i:05d}" for i in range(n_genes)]

    # --- pathway membership: each gene in 1-2 pathways ---
    gene_pw = sp.lil_matrix((n_genes, n_pathways), dtype=bool)
    primary_pw = rng.integers(0, n_pathways, size=n_genes)
    for g in range(n_genes):
        gene_pw[g, primary_pw[g]] = True
        if rng.random() < 0.3:
            gene_pw[g, rng.integers(0, n_pathways)] = True
    gene_pw = gene_pw.tocsr()

    # --- GRN: ~10% genes are TFs; edges preferentially within shared pathway ---
    n_tf = max(20, n_genes // 10)
    tfs = rng.choice(n_genes, size=n_tf, replace=False)
    rows, cols, wts = [], [], []
    for tf in tfs:
        tf_pw = primary_pw[tf]
        same = np.where(primary_pw == tf_pw)[0]
        n_t = rng.integers(8, 30)
        # 80% targets within pathway, 20% cross-pathway
        in_pw = rng.choice(same, size=min(n_t, len(same)), replace=False)
        cross = rng.choice(n_genes, size=max(0, n_t // 4), replace=False)
        targets = np.unique(np.concatenate([in_pw, cross]))
        for tgt in targets:
            if tgt == tf:
                continue
            rows.append(tgt); cols.append(tf)              # A[target, tf]
            wts.append(rng.normal(0, 0.9))
    A = sp.csr_matrix((wts, (rows, cols)), shape=(n_genes, n_genes), dtype=np.float32)
    grn_mask = (A != 0)                                     # directed adjacency for attention mask

    # --- cell-line baselines: pathway-activity weighted ---
    pw_activity = rng.normal(0, 1.0, size=(len(CELL_LINES), n_pathways)).astype(np.float32)
    gene_base = rng.normal(2.0, 0.7, size=n_genes).astype(np.float32)
    ctrl = np.zeros((len(CELL_LINES), n_genes), dtype=np.float32)
    gp_dense = gene_pw.toarray().astype(np.float32)
    for ci in range(len(CELL_LINES)):
        ctrl[ci] = np.clip(gene_base + gp_dense @ pw_activity[ci] * 0.4
                           + rng.normal(0, 0.2, n_genes), 0, None)

    drugs = _validate_drugs()
    if n_extra_drugs > 0:
        # auto-generated drugs (empty SMILES -> splits fall back to drug-name clusters, giving many
        # independent test clusters for statistical power). Chemistry is unused: target is an input.
        for i in range(n_extra_drugs):
            drugs.append((f"syn{i:04d}", "", PATHWAY_TAGS[i % n_pathways], -1))
    # Targets must be INFERABLE FROM CHEMISTRY for unseen-drug generalization to be possible:
    # each pathway/MoA class gets a small set of TF "hub" targets, and every drug in that class
    # hits one of its class hubs. Real drug classes (MEK inhibitors, CDK inhibitors, ...) share
    # targets and have related scaffolds, so ChemBERTa(SMILES) correlates with the target hub.
    tag_to_pw = {t: i for i, t in enumerate(pw_names)}
    tags = sorted({tag for (_, _, tag, _) in drugs})
    tag_hubs = {}
    for tag in tags:
        if tag in tag_to_pw:
            cand = np.intersect1d(np.where(primary_pw == tag_to_pw[tag])[0], tfs)
            if len(cand) == 0:
                cand = np.where(primary_pw == tag_to_pw[tag])[0]
        else:
            cand = tfs
        n_hub = min(3, len(cand))
        tag_hubs[tag] = rng.choice(cand, size=max(1, n_hub), replace=False).tolist()
    drug_target = {}
    if unique_targets:
        # each drug -> a DISTINCT TF target (novel-target regime; GRN essential for unseen drugs)
        pool = rng.permutation(tfs)
        for i, (name, smi, tag, sign) in enumerate(drugs):
            drug_target[name] = int(pool[i % len(pool)])
    else:
        for i, (name, smi, tag, sign) in enumerate(drugs):
            hubs = tag_hubs[tag]
            drug_target[name] = int(hubs[i % len(hubs)])

    # propagation operator: sum_{h=0}^{H} (decay*A)^h applied to initial delta
    decay = 0.7
    def propagate(delta0):
        delta = delta0.copy(); total = delta0.copy()
        for _ in range(hops):
            delta = decay * (A @ delta)
            total = total + delta
        return total

    rows_out = []
    drug_meta = []
    for (name, smi, tag, sign) in drugs:
        gt = drug_target[name]
        drug_meta.append({"name": name, "smiles": smi, "target_gene": genes[gt], "pathway": tag})
        for ci, cl in enumerate(CELL_LINES):
            for dose in DOSES:
                base = ctrl[ci]
                dose_factor = np.log1p(dose) / np.log1p(DOSES[-1])
                # on-target effect scaled by target's baseline expression in THIS cell line
                amp = sign * 6.0 * dose_factor * (base[gt] / (gene_base[gt] + 1e-3))
                delta0 = np.zeros(n_genes, dtype=np.float32); delta0[gt] = amp
                lfc = propagate(delta0)
                # compensatory pathway feedback (small, opposes mean pathway shift)
                pw_shift = (gp_dense.T @ lfc) / (gp_dense.sum(0) + 1e-6)
                lfc -= 0.15 * (gp_dense @ pw_shift)
                # cell-level noise -> condition pseudobulk noise ~ 1/sqrt(n_cells)
                n_cells = int(rng.poisson(n_cells_mean)) + 20
                lfc = lfc + rng.normal(0, 0.4 / np.sqrt(n_cells), n_genes).astype(np.float32)
                pert = np.clip(base + lfc, 0, None)
                lfc = (pert - base).astype(np.float32)
                # DEG significance proxy: z = |lfc| / (pooled_sd/sqrt(n))
                se = 0.5 / np.sqrt(n_cells)
                deg = (np.abs(lfc) > 0.5) & (np.abs(lfc) / se > 3.0)
                rows_out.append(dict(
                    condition_id=f"{cl}|{name}|{dose}", cell_line=cl, treatment=name,
                    smiles=smi, target_gene=genes[gt], dose=np.float32(dose),
                    dose_log=np.float32(np.log1p(dose)), n_cells=np.int32(n_cells),
                    n_ctrl_cells=np.int32(n_cells_mean),
                    ctrl_mean=base.astype(np.float32), pert_mean=pert.astype(np.float32),
                    lfc=lfc, deg_mask=deg))
    df = pd.DataFrame(rows_out)
    return df, genes, pw_names, gene_pw, grn_mask, drug_meta


def write(df, genes, pw_names, gene_pw, grn_mask, drug_meta, dataset="synthetic"):
    out = dataset_dir(dataset); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "conditions.parquet")
    (out / "genes.txt").write_text("\n".join(genes))
    meta = dict(dataset=dataset, N_GENES=len(genes), cell_lines=CELL_LINES,
                drugs=drug_meta, dose_unit="uM", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                n_conditions=len(df), notes="synthetic; GRN+pathway ground truth in priors/synthetic")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    pri = DATA_ROOT / "data" / "priors" / dataset; pri.mkdir(parents=True, exist_ok=True)
    sp.save_npz(pri / "grn_mask.npz", grn_mask.tocsr().astype(bool))
    sp.save_npz(pri / "pathways.npz", gene_pw.tocsr())
    (pri / "pathway_names.txt").write_text("\n".join(pw_names))
    (pri / "grn_mask.meta.json").write_text(json.dumps(
        {"source": "synthetic-groundtruth", "n_edges": int(grn_mask.nnz)}, indent=2))
    return out, pri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=2000)
    ap.add_argument("--n-pathways", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--unique-targets", action="store_true")
    ap.add_argument("--n-extra-drugs", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    df, genes, pw, gp, grn, dm = build(a.n_genes, a.n_pathways, seed=a.seed,
                                       unique_targets=a.unique_targets, n_extra_drugs=a.n_extra_drugs)
    print(f"[{a.dataset}] conditions={len(df)} genes={len(genes)} pathways={len(pw)} "
          f"drugs={len(dm)} grn_edges={grn.nnz} unique_targets={a.unique_targets}")
    print(f"mean |lfc|={np.abs(np.stack(df['lfc'].to_numpy())).mean():.3f} "
          f"mean DEGs/cond={np.stack(df['deg_mask'].to_numpy()).sum(1).mean():.1f}")
    if a.dry_run:
        print("[dry-run] not writing"); return
    out, pri = write(df, genes, pw, gp, grn, dm, dataset=a.dataset)
    print(f"wrote {out}\n      {pri}")


if __name__ == "__main__":
    main()
