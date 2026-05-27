> ⚠️ **SUPERSEDED / DO NOT CITE (V1, leakage-tainted).** These numbers come from the V1
> pipeline with HVG-selection leakage (AUDIT.md bug B) and corrupted-low baselines. Under the
> leakage-free V2 protocol the baselines are ~2× higher and BEAT PathwayMoE on the OOD splits
> (unseen_cell_line: ridge 0.868 > MoE 0.824, p<0.001; unseen_both: mean-effect 0.607 > MoE 0.581).
> See `results/REAL_TAHOE_OOD_AUDIT.md` and `study_tahoe_full_v2.json`. The "PathwayMoE beats
> baselines on hard OOD" claim does NOT hold.

# Comparison — tahoe (metric: DEG-Pearson @top-50)

| Model | Params | unseen_drug | unseen_cell_line | unseen_both |
|---|---|---|---|---|
| B1_mean_effect | 0 | 0.393 | 0.432 | 0.324 |
| B2_ridge | 1M | 0.715 | 0.396 | 0.180 |
| B3_ridge_bio | 1M | 0.715 | 0.396 | 0.180 |
| PathwayMoE | 45M | 0.704 | 0.551 | 0.512 |
