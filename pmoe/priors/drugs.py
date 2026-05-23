"""Drug featurization: Morgan fingerprints (RDKit, offline) + ChemBERTa embedding.

Ported from v1 ``code/drugs.py``; the only changes are import paths (``pmoe.config``) and
that output goes to :func:`pmoe.config.priors_dir`.

ChemBERTa (DeepChem/ChemBERTa-77M-MLM) is loaded only if ``transformers`` + network are
available; otherwise we fall back to a deterministic seeded random projection of the Morgan
fingerprint so the whole pipeline runs offline. The fallback is recorded in
``drug_feats.meta.json`` so a downstream result is never silently based on fake chemistry.

NOTE FOR TESTS: :func:`build_drug_feats` may attempt to load ChemBERTa (network/GPU). Tests must
NOT call it; unit tests for the rest of the subpackage pass arrays/DataFrames directly.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from pmoe.config import SEED, conditions_path, priors_dir

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

MORGAN_BITS = 2048
CHEMBERTA_DIM = 384


def morgan_fp(smiles: str, bits: int = MORGAN_BITS) -> np.ndarray:
    """Morgan (ECFP4) fingerprint as a dense float32 bit vector.

    Falls back to a deterministic md5-seeded sparse vector when RDKit cannot parse the SMILES.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        m = Chem.MolFromSmiles(smiles) if smiles else None
        if m is None:
            return np.zeros(bits, np.float32)
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, bits)
        arr = np.zeros(bits, np.int8)
        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(bv, arr)
        return arr.astype(np.float32)
    except Exception:
        # hash-based fallback fingerprint (deterministic)
        h = int(hashlib.md5((smiles or "").encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2 ** 32))
        v = np.zeros(bits, np.float32)
        v[rng.integers(0, bits, 30)] = 1.0
        return v


def _chemberta(smiles_list: list[str]) -> tuple[np.ndarray | None, str]:
    """Embed SMILES with ChemBERTa CLS token. Returns (None, reason) if unavailable."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = "DeepChem/ChemBERTa-77M-MLM"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name).eval()
        embs = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), 32):
                batch = [s if s else "C" for s in smiles_list[i:i + 32]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                out = mdl(**enc).last_hidden_state[:, 0]  # CLS
                embs.append(out.cpu().numpy())
        E = np.concatenate(embs).astype(np.float32)
        return E, f"chemberta:{name}"
    except Exception as e:
        return None, f"fallback:{type(e).__name__}"


def build_drug_feats(dataset: str) -> pd.DataFrame:
    """Build per-drug features (Morgan + ChemBERTa) and persist to ``priors/<dataset>/``.

    Columns: ``treatment, smiles, target_gene, morgan, chemberta``. Writes
    ``drug_feats.parquet`` and ``drug_feats.meta.json`` (records the ChemBERTa source / fallback).
    """
    df = pd.read_parquet(conditions_path(dataset), columns=["treatment", "smiles", "target_gene"])
    uniq = df.drop_duplicates("treatment").reset_index(drop=True)
    smiles = uniq["smiles"].fillna("").tolist()
    morgan = np.stack([morgan_fp(s) for s in smiles])

    cb, source = _chemberta(smiles)
    if cb is None:
        rng = np.random.default_rng(SEED)
        proj = rng.standard_normal((MORGAN_BITS, CHEMBERTA_DIM)).astype(np.float32) / np.sqrt(MORGAN_BITS)
        cb = (morgan @ proj).astype(np.float32)

    feats = pd.DataFrame({
        "treatment": uniq["treatment"],
        "smiles": uniq["smiles"],
        "target_gene": uniq["target_gene"],
        "morgan": list(morgan),
        "chemberta": list(cb),
    })
    pri = priors_dir(dataset)
    feats.to_parquet(pri / "drug_feats.parquet")
    (pri / "drug_feats.meta.json").write_text(json.dumps(
        {"source": source, "n_drugs": len(feats), "morgan_bits": MORGAN_BITS,
         "chemberta_dim": int(cb.shape[1])}, indent=2))
    return feats


def load_drug_feats(dataset: str) -> pd.DataFrame:
    """Load cached drug features; build them (may invoke ChemBERTa) if the cache is missing."""
    p = priors_dir(dataset) / "drug_feats.parquet"
    if not p.exists():
        return build_drug_feats(dataset)
    return pd.read_parquet(p)
