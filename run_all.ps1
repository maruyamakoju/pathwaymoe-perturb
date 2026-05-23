# End-to-end orchestration (native Windows). Activate the venv first:  .\.venv\Scripts\activate
$ErrorActionPreference = "Stop"
$env:VC_DATA_ROOT = "E:\vc_project_data"
$py = ".\.venv\Scripts\python.exe"
$DS = if ($args.Count -ge 1) { $args[0] } else { "synthetic" }

Write-Host "== 1. data + priors ($DS) ==" -ForegroundColor Cyan
if ($DS -eq "synthetic") { & $py code\synth.py }
& $py code\drugs.py $DS
# grn/pathways already written by synth for synthetic; rebuild for real datasets:
if ($DS -ne "synthetic") { & $py code\grn.py --dataset $DS; & $py code\pathways.py --dataset $DS }

Write-Host "== 2. baselines ==" -ForegroundColor Cyan
& $py code\baselines.py --dataset $DS

Write-Host "== 3. train (3 splits) ==" -ForegroundColor Cyan
foreach ($s in @("unseen_drug","unseen_cell_line","unseen_both")) {
  & $py code\train.py --dataset $DS --split $s --size base --grad-ckpt
}

Write-Host "== 4. eval + figures ==" -ForegroundColor Cyan
& $py code\eval.py --dataset $DS
& $py code\figures.py --dataset $DS
Write-Host "Done. See results\comparison_$DS.md" -ForegroundColor Green
