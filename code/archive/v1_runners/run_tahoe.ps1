# Train PathwayMoE on REAL Tahoe subset with REAL priors (TRRUST GRN + Reactome pathways).
$env:VC_DATA_ROOT = "E:\vc_project_data"; $env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
foreach ($s in @("unseen_drug","unseen_cell_line","unseen_both")) {
  & $py -u code\train.py --dataset tahoe --split $s --size base --batch-size 64 --epochs 80 --patience 12 *>&1 | Tee-Object ("runs\tahoe_" + $s + ".log")
}
# ablation: does the REAL GRN help on REAL data?
& $py -u code\train.py --dataset tahoe --split unseen_drug --size base --batch-size 64 --epochs 80 --patience 12 --no-grn *>&1 | Tee-Object runs\tahoe_unseen_drug_nogrn.log
Write-Host "TAHOE RUNS DONE"
