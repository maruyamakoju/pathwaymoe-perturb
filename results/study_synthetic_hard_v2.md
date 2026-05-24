# GRN-quality study (v2, leakage-free, cluster-bootstrap 95% CI)

Data manifest: {"dataset": "synthetic_hard", "n_genes": 2000, "n_conditions": 1410, "n_cells_total": null, "n_shards": null, "genes_sha1": "73fd13e31c46", "cell_lines": 10}


## unseen_drug  (n_test=270, clusters=9)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | -0.237 | [-0.437, -0.040] |
| baseline: B2_ridge | 0.024 | [-0.202, 0.230] |
| baseline: B3_ridge_bio | 0.014 | [-0.215, 0.226] |
| **MoE / GRN=none** | **-0.080** | [-0.275, 0.105] |
| **MoE / GRN=random** | **-0.059** | [-0.254, 0.121] |
| **MoE / GRN=ground_truth** | **-0.062** | [-0.252, 0.120] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|
| random_vs_none | +0.022 | [-0.003, +0.049] | 0.097 | no |
| ground_truth_vs_none | +0.019 | [+0.007, +0.032] | 0.000 | YES |
