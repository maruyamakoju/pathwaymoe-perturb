# ⚠️ Read before using anything in this folder

## VOID — do not cite
- `trametinib_gates.json`, `nutlin-3_gates.json`

These were presented as a drug "MoA discovery" but are **not valid**:
1. They were extracted from the **synthetic** dataset (gene IDs `G00000–G01999`, not real
   gene symbols), so they cannot be cross-validated against MAPK/p53 or any real pathway.
   The drug names are synthetic labels, not the real compounds.
2. The underlying LRI gate is **degenerate** (a flat 0.5 matrix; see
   `gate_degeneracy_report*.json`). The high-confidence "edges" (>0.8) were **eval-time
   sampling noise** (per-edge std 0.24 around a 0.5 mean), not learned regulatory logic.

Full analysis: `../LATENT_GRN_STUDY.md` and `../../WORK_LOG.md`.

## VALID — the actual audit output
- `gate_degeneracy_report.json`            — original long_latent (50ep) gate audit
- `gate_degeneracy_report_long_latent.json`— same, recomputed with the corrected metrics
- `gate_degeneracy_report_gateonly.json`   — steelman gate-only model
- `gate_degeneracy_report_gateonly_lowkl.json` — steelman gate-only, KL=1e-5

Reproduce: `python code/archive/analyze_gate_degeneracy.py <run_name> synthetic_hard_v2 <suffix>`
