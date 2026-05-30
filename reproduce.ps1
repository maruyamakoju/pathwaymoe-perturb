# End-to-end reproduction of the GRN-quality study. Activate .venv first.
# Real-data steps assume the Tahoe-100M subset has been built (code\preprocess_tahoe_stream.py).
$ErrorActionPreference = "Stop"
$env:VC_DATA_ROOT = "E:\vc_project_data"; $env:PYTHONUNBUFFERED = "1"
# Workspace size for CuBLAS deterministic matmul (silences the use_deterministic_algorithms warning
# on Linear/MoE matmuls). Required for fully bit-reproducible training runs; harmless otherwise.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$py = ".\.venv\Scripts\python.exe"

Write-Host "== tests ==" -ForegroundColor Cyan
& $py -m pytest -q

Write-Host "== synthetic (shared-target) + positive control ==" -ForegroundColor Cyan
& $py -m pmoe.data.synth --dataset synthetic
& $py -m pmoe.experiments.study --dataset synthetic --splits unseen_drug `
      --variants none random coexpr coexpr_lfc ground_truth --seeds 1337 1 2 --epochs 55 --batch 64

Write-Host "== synthetic (novel-target) mechanism: mask vs soft message-passing ==" -ForegroundColor Cyan
& $py -m pmoe.data.synth --dataset synthetic_hard_big --unique-targets --n-extra-drugs 150
# Train on CUDA; analyze on CPU because the near-zero-variance metric is sensitive to CUDA
# atomic-add ordering in MoE index_add_ (~5% DEG50 swing per process invocation). See AUDIT.md N.
& $py -m pmoe.experiments.mechanism --dataset synthetic_hard_big --seeds 1337 1 2 --epochs 35 --batch 96 --train-only
& $py -m pmoe.experiments.mechanism --dataset synthetic_hard_big --seeds 1337 1 2 --analyze-only --eval-device cpu

Write-Host "== real Tahoe-100M (requires built subset) ==" -ForegroundColor Cyan
# & $py -m pmoe.data.preprocess_tahoe --dataset tahoe_full --n-hvg 2000   # build from shards first
# GRN-quality spectrum incl. dense curated CollecTRI + matched-density random control.
# CollecTRI needs CollecTRI_regulons.csv under <priors> (Zenodo 8192729; omnipathdb.org is down).
# batch=48 fits the 4090 at 24GB; analyze on CPU to keep eval bit-reproducible (AUDIT.md N).
& $py -m pmoe.experiments.study --dataset tahoe_full --splits unseen_drug `
      --variants none random trrust trrust_weighted coexpr coexpr_lfc collectri collectri_weighted random_dense `
      --seeds 1337 1 2 --epochs 25 --batch 48 --eval-device cpu
& $py -m pmoe.eval.report --dataset tahoe_full

Write-Host "== drug->target mechanistic grounding (with_targets) ==" -ForegroundColor Cyan
# Needs tahoe_drug_metadata.parquet under <priors> (HF tahoebio/Tahoe-100M, config drug_metadata).
& $py -m pmoe.experiments.drug_target --dataset tahoe_full --split unseen_drug --seeds 1337 1 2 --batch 48

Write-Host "DONE. See results\PAPER.md, results\study_*_v2.json, results\drug_target_*.json, results\fig_grn_study_*_v2.png" -ForegroundColor Green
