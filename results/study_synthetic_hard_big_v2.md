# GRN-quality study (v2, leakage-free, cluster-bootstrap 95% CI)

Data manifest: {"dataset": "synthetic_hard_big", "n_genes": 2000, "n_conditions": 5910, "n_cells_total": null, "n_shards": null, "genes_sha1": "73fd13e31c46", "cell_lines": 10}


## unseen_drug  (n_test=1230, clusters=39)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | 0.054 | [-0.048, 0.158] |
| baseline: B2_ridge | 0.008 | [-0.073, 0.100] |
| baseline: B3_ridge_bio | 0.031 | [-0.087, 0.152] |
| **MoE / GRN=none** | **0.024** | [-0.055, 0.099] |
| **MoE / GRN=random** | **0.019** | [-0.060, 0.094] |
| **MoE / GRN=ground_truth** | **0.021** | [-0.059, 0.098] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|
| random_vs_none | -0.005 | [-0.016, +0.007] | 0.862 | no |
| ground_truth_vs_none | -0.003 | [-0.012, +0.006] | 0.862 | no |
