# `code/archive/` — v1 provenance

These are the v1 scripts that produced the early results in this project. They are
superseded by the typed, tested `pmoe/` package and **must not be re-run** for the
published numbers — most of them contain bugs the audit fixed (`AUDIT.md` items A–K).
They live here for provenance: WORK_LOG.md, LATEST_STATUS.md, and several
`results/*.md` files reference them by name.

## v1 -> v2 mapping

| `code/archive/<file>.py`   | Replaced by `pmoe.*`                       | Notes |
|---|---|---|
| `config.py`                | `pmoe.config`                              | flat config; v2 is hierarchical |
| `loader.py`                | `pmoe.data.loader`                         | |
| `splits.py`                | `pmoe.data.splits`                         | v2 also returns cluster `groups` for the cluster bootstrap |
| `data.py`                  | `pmoe.data.dataset`                        | |
| `drugs.py`                 | `pmoe.priors.drugs`                        | |
| `grn.py`                   | `pmoe.priors.grn`                          | v2 adds the variant spectrum + leakage guard |
| `pathways.py`              | `pmoe.priors.pathways`                     | |
| `build_real_priors.py`     | `pmoe.priors.grn` + `pmoe.priors.pathways` | |
| `build_grn_variants.py`    | `pmoe.priors.grn`                          | **v1 source of AUDIT.md leakage bug A** (built coexpr from all conditions incl. test) |
| `metrics.py`               | `pmoe.eval.metrics`                        | |
| `stats.py`                 | `pmoe.eval.stats`                          | v1 was naive bootstrap; v2 has cluster bootstrap + Holm + effect size |
| `model.py`                 | `pmoe.models.{pathway_moe,layers}`         | v1 source of AUDIT.md bug J (silent train-without-GRN) |
| `baselines.py`             | `pmoe.models.baselines`                    | |
| `train.py`                 | `pmoe.experiments.train`                   | v1 had the silent grn=None fallback; v2 raises |
| `eval.py` / `eval_ablations.py` / `run_study.py` | `pmoe.eval.loading` (consolidated) | v1 triplicated `model_predict`; one copy had a GRN-reload bug |
| `figures.py` / `study_figures.py` / `figure_comparison_clean.py` | `pmoe.eval.{report,money_figure}` | v1 reads HVG-leaky v1 JSON layout |
| `preprocess.py`            | (removed)                                  | h5ad path; was never referenced |
| `preprocess_tahoe.py`      | `pmoe.data.preprocess_tahoe`               | older non-streaming path |
| `preprocess_tahoe_stream.py` | `pmoe.data.preprocess_tahoe`             | ported in commit 695f130 |
| `synth.py`                 | `pmoe.data.synth`                          | ported in commit 695f130 |

## One-off analyses (no v2 counterpart needed)

| File | Purpose | Result |
|---|---|---|
| `analyze_gate_degeneracy.py` | Audit the LRI gate ("posterior collapse" check) | proved the LRI gate is a flat-0.5 no-op (WORK_LOG.md step 0) |
| `run_long_static.py`         | Matched 50-epoch static run to falsify the Gemini "breakthrough" | static beats latent at every matched epoch |
| `run_steelman.py`            | Best-effort revival of the dynamic-GRN idea (gate-only, low-KL) | still <= no-GRN floor |
| `run_multiseed_confirm.py`   | 3-seed confirmation of static > latent | static wins every seed |
| `run_long_study.py`          | One-off long training of static_baseline / latent_grn | see WORK_LOG.md |
| `run_latent_study.py`        | Initial latent-GRN sweep | superseded by the audit |
| `run_tahoe_latent.py`        | Attempted real-Tahoe latent training | interrupted, no usable ckpt |
| `visualize_gates.py`         | Old gate visualizer | invalidated by the gate-degeneracy finding |

## `v1_runners/`

`run_all.ps1`, `run_remaining.ps1`, `run_tahoe.ps1` are the v1 PowerShell drivers
(call `code/train.py`, `code/eval.py`, etc.). The canonical reproduction script is
now `../reproduce.ps1` at the repo root, which uses the `pmoe.*` package.
