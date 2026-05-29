"""Per-variant behaviour tests for pmoe.priors.grn (offline, CPU, no real dataset).

Covers:
  * random  -> exactly match_edges edges (matched density).
  * ground_truth -> round-trips an existing grn_mask.npz.
  * trrust_weighted -> float, signed (has negative weights) CSR.
  * none -> empty matrix, load_grn returns None.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pmoe.priors.grn import build_grn, load_grn
from pmoe.config import GRNVariant, priors_dir

N_GENES = 60


def _genes(n=N_GENES):
    return [f"G{i:05d}" for i in range(n)]


def test_random_matches_requested_edge_count():
    genes = _genes()
    want = 137
    m = build_grn("synthetic_smoke", GRNVariant.RANDOM, match_edges=want, genes=genes, seed=7)
    assert m.shape == (N_GENES, N_GENES)
    assert m.nnz == want
    assert m.dtype == bool
    # determinism
    m2 = build_grn("synthetic_smoke", GRNVariant.RANDOM, match_edges=want, genes=genes, seed=7)
    assert (m != m2).nnz == 0


def test_ground_truth_roundtrips(tmp_path, monkeypatch):
    # write a tiny grn_mask.npz into a throwaway dataset's priors dir, then load via ground_truth
    ds = "gt_roundtrip_test"
    genes = _genes()
    rng = np.random.default_rng(0)
    r = rng.integers(0, N_GENES, 40)
    c = rng.integers(0, N_GENES, 40)
    keep = r != c
    truth = sp.csr_matrix((np.ones(keep.sum(), bool), (r[keep], c[keep])), shape=(N_GENES, N_GENES))
    sp.save_npz(priors_dir(ds) / "grn_mask.npz", truth)

    m = build_grn(ds, GRNVariant.GROUND_TRUTH, genes=genes)
    assert m.shape == truth.shape
    # same edge set
    a = set(zip(*truth.tocoo().nonzero()))
    b = set(zip(*m.tocoo().nonzero()))
    assert a == b

    loaded = load_grn(ds, GRNVariant.GROUND_TRUTH)
    assert loaded is not None and (loaded != m).nnz == 0


def test_trrust_weighted_is_signed_float(tmp_path, monkeypatch):
    """trrust_weighted must produce a float CSR with BOTH positive (Activation) and negative
    (Repression) weights -- not just "any non-zero", which the previous assertion allowed.

    We mock the TRRUST file and the symbol->Ensembl vocab so the test is hermetic (doesn't
    depend on E:\\vc_project_data files).
    """
    genes = [f"ENSG{i:08d}" for i in range(20)]
    # Fixture TRRUST: A->B Activation, C->D Repression, E->F Unknown.
    trrust_text = "\n".join([
        "GENEA\tGENEB\tActivation\tpmid1",
        "GENEC\tGENED\tRepression\tpmid2",
        "GENEE\tGENEF\tUnknown\tpmid3",
    ]) + "\n"
    vocab_text = "\n".join([
        '{"gene_symbol":"GENEA","ensembl_id":"ENSG00000000"}',
        '{"gene_symbol":"GENEB","ensembl_id":"ENSG00000001"}',
        '{"gene_symbol":"GENEC","ensembl_id":"ENSG00000002"}',
        '{"gene_symbol":"GENED","ensembl_id":"ENSG00000003"}',
        '{"gene_symbol":"GENEE","ensembl_id":"ENSG00000004"}',
        '{"gene_symbol":"GENEF","ensembl_id":"ENSG00000005"}',
    ]) + "\n"
    trrust_path = tmp_path / "trrust_human.tsv"; trrust_path.write_text(trrust_text)
    vocab_path = tmp_path / "gene_vocabulary.jsonl"; vocab_path.write_text(vocab_text)

    import pmoe.priors.grn as grn_mod
    monkeypatch.setattr(grn_mod, "TRRUST_TSV", trrust_path)
    monkeypatch.setattr(grn_mod, "VOCAB", vocab_path)

    m = build_grn("synthetic_smoke", GRNVariant.TRRUST_WEIGHTED, genes=genes)
    assert np.issubdtype(m.dtype, np.floating), "weighted GRN must be float"
    assert m.nnz == 3, f"expected exactly 3 TRRUST edges, got {m.nnz}"
    data = m.data
    assert (data > 0).any(), "Activation edge should produce a positive weight"
    assert (data < 0).any(), "Repression edge should produce a negative weight"


def _mock_collectri(tmp_path, monkeypatch):
    """Mock CollecTRI_regulons.csv + the symbol->Ensembl vocab; return the gene list.

    Fixture network: TFA->TGB activation(+1), TFA->TGC repression(-1), TFD->TGE activation(+1),
    plus a self-loop and an out-of-space edge that must both be dropped.
    """
    genes = [f"ENSG{i:08d}" for i in range(20)]
    csv_text = "source,target,weight,resources,references,sign_decision\n" + "\n".join([
        "TFA,TGB,1.0,r,ref,PMID",
        "TFA,TGC,-1.0,r,ref,PMID",
        "TFD,TGE,1.0,r,ref,PMID",
        "TFA,TFA,1.0,r,ref,PMID",          # self-loop -> dropped
        "TFA,NOTINSPACE,1.0,r,ref,PMID",   # target not in gene space -> dropped
    ]) + "\n"
    vocab_text = "\n".join([
        '{"gene_symbol":"TFA","ensembl_id":"ENSG00000000"}',
        '{"gene_symbol":"TGB","ensembl_id":"ENSG00000001"}',
        '{"gene_symbol":"TGC","ensembl_id":"ENSG00000002"}',
        '{"gene_symbol":"TFD","ensembl_id":"ENSG00000003"}',
        '{"gene_symbol":"TGE","ensembl_id":"ENSG00000004"}',
    ]) + "\n"
    csv_path = tmp_path / "CollecTRI_regulons.csv"; csv_path.write_text(csv_text)
    vocab_path = tmp_path / "gene_vocabulary.jsonl"; vocab_path.write_text(vocab_text)
    import pmoe.priors.grn as grn_mod
    monkeypatch.setattr(grn_mod, "COLLECTRI_CSV", csv_path)
    monkeypatch.setattr(grn_mod, "VOCAB", vocab_path)
    return genes


def test_collectri_topology_drops_selfloops_and_out_of_space(tmp_path, monkeypatch):
    genes = _mock_collectri(tmp_path, monkeypatch)
    m = build_grn("synthetic_smoke", GRNVariant.COLLECTRI, genes=genes)
    assert m.dtype == bool
    assert m.nnz == 3, f"expected 3 in-space edges (self-loop + out-of-space dropped), got {m.nnz}"


def test_collectri_weighted_is_signed_float(tmp_path, monkeypatch):
    genes = _mock_collectri(tmp_path, monkeypatch)
    m = build_grn("synthetic_smoke", GRNVariant.COLLECTRI_WEIGHTED, genes=genes)
    assert np.issubdtype(m.dtype, np.floating), "weighted CollecTRI must be float"
    assert m.nnz == 3
    assert (m.data > 0).any(), "activation edge -> positive weight"
    assert (m.data < 0).any(), "repression edge -> negative weight"
    # meta records the curated source as leakage-safe (carries no test info)
    import json
    from pmoe.config import priors_dir, grn_filename
    meta = json.loads((priors_dir("synthetic_smoke") /
                       (grn_filename(GRNVariant.COLLECTRI_WEIGHTED)[:-4] + ".meta.json")).read_text())
    assert meta["leakage_safe"] is True
    assert meta["weighted"] is True


def test_random_dense_matches_collectri_edge_count(tmp_path, monkeypatch):
    """random_dense defaults to the CollecTRI in-space edge count (floored at 100, like random).

    On the real 2,000-HVG space CollecTRI has 2,193 edges so the floor never binds; the explicit
    match_edges path is what the study uses to pin the density control exactly to CollecTRI.
    """
    genes = _mock_collectri(tmp_path, monkeypatch)
    n_ct = build_grn("synthetic_smoke", GRNVariant.COLLECTRI, genes=genes).nnz
    m = build_grn("synthetic_smoke", GRNVariant.RANDOM_DENSE, genes=genes, seed=3)
    assert m.dtype == bool
    assert m.nnz == max(n_ct, 100)            # 100-edge floor for tiny fixtures
    m_exact = build_grn("synthetic_smoke", GRNVariant.RANDOM_DENSE,
                        match_edges=n_ct, genes=genes, seed=3)
    assert m_exact.nnz == n_ct                # explicit budget = exact CollecTRI density


def test_none_is_empty():
    genes = _genes()
    m = build_grn("synthetic_smoke", GRNVariant.NONE, genes=genes)
    assert m.nnz == 0
    assert load_grn("synthetic_smoke", GRNVariant.NONE) is None
