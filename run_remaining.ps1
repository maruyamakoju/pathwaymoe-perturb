# Remaining training: other 2 splits (full) + 4 ablations on unseen_drug (shorter).
$env:VC_DATA_ROOT = "E:\vc_project_data"; $env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
& $py -u code\train.py --dataset synthetic --split unseen_cell_line --size base --batch-size 64 --epochs 80 --patience 12 *>&1 | Tee-Object runs\train_unseen_cell_line.log
& $py -u code\train.py --dataset synthetic --split unseen_both       --size base --batch-size 64 --epochs 80 --patience 12 *>&1 | Tee-Object runs\train_unseen_both.log
foreach ($ab in @("--no-grn","--no-moe","--no-pathway-prior","--pert-as-token")) {
  & $py -u code\train.py --dataset synthetic --split unseen_drug --size base --batch-size 64 --epochs 60 --patience 10 $ab *>&1 | Tee-Object ("runs\ablation" + ($ab -replace '--','_') + ".log")
}
Write-Host "ALL REMAINING RUNS DONE"
