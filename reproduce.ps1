# End-to-end reproduction of the GRN-quality study. Activate .venv first.
# Real-data steps assume the Tahoe-100M subset has been built (code\preprocess_tahoe_stream.py).
$ErrorActionPreference = "Stop"
$env:VC_DATA_ROOT = "E:\vc_project_data"; $env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"

Write-Host "== tests ==" -ForegroundColor Cyan
& $py -m pytest -q

Write-Host "== synthetic (shared-target) + positive control ==" -ForegroundColor Cyan
& $py code\synth.py --dataset synthetic
& $py -m pmoe.experiments.study --dataset synthetic --splits unseen_drug `
      --variants none random coexpr coexpr_lfc ground_truth --seeds 1337 1 2 --epochs 55 --batch 64

Write-Host "== synthetic (novel-target) mechanism: mask vs soft message-passing ==" -ForegroundColor Cyan
& $py code\synth.py --dataset synthetic_hard_big --unique-targets --n-extra-drugs 150
& $py -m pmoe.experiments.mechanism --dataset synthetic_hard_big --seeds 1337 1 2 --epochs 35 --batch 96

Write-Host "== real Tahoe-100M (requires built subset) ==" -ForegroundColor Cyan
# & $py code\preprocess_tahoe_stream.py --dataset tahoe_full --n-hvg 2000   # build from shards first
& $py -m pmoe.experiments.study --dataset tahoe_full --splits unseen_drug `
      --variants none random trrust trrust_weighted coexpr coexpr_lfc --seeds 1337 1 2 --epochs 25 --batch 96
& $py -m pmoe.eval.report --dataset tahoe_full

Write-Host "DONE. See results\PAPER.md, results\study_*_v2.json, results\fig_grn_study_*_v2.png" -ForegroundColor Green
