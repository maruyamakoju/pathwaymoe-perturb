# GRN-quality study (v2, leakage-free, cluster-bootstrap 95% CI)

Data manifest: {"dataset": "tahoe_full", "n_genes": 2000, "n_conditions": 8875, "n_cells_total": 22664673, "n_shards": 803, "genes_sha1": "034932db76c4", "cell_lines": 50}


## unseen_drug  (n_test=1708, clusters=39)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | 0.601 | [0.537, 0.655] |
| baseline: B2_ridge | 0.582 | [0.507, 0.646] |
| baseline: B3_ridge_bio | 0.582 | [0.507, 0.646] |
| **MoE / GRN=none** | **0.629** | [0.551, 0.697] |
| **MoE / GRN=random** | **0.622** | [0.538, 0.694] |
| **MoE / GRN=trrust** | **0.626** | [0.545, 0.697] |
| **MoE / GRN=trrust_weighted** | **0.628** | [0.546, 0.698] |
| **MoE / GRN=coexpr** | **0.625** | [0.542, 0.695] |
| **MoE / GRN=coexpr_lfc** | **0.624** | [0.544, 0.695] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|
| random_vs_none | -0.007 | [-0.015, -0.000] | 0.283 | no |
| trrust_vs_none | -0.003 | [-0.009, +0.004] | 1.000 | no |
| trrust_weighted_vs_none | -0.001 | [-0.006, +0.004] | 1.000 | no |
| coexpr_vs_none | -0.004 | [-0.011, +0.001] | 0.772 | no |
| coexpr_lfc_vs_none | -0.005 | [-0.011, +0.001] | 0.519 | no |
| coexpr_lfc_vs_trrust | -0.003 | [-0.007, +0.001] | 0.826 | no |
| coexpr_lfc_vs_random | +0.002 | [-0.003, +0.007] | 1.000 | no |
| trrust_vs_random | +0.005 | [+0.001, +0.009] | 0.124 | no |

## unseen_cell_line  (n_test=1767, clusters=10)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | 0.850 | [0.809, 0.883] |
| baseline: B2_ridge | 0.868 | [0.826, 0.899] |
| baseline: B3_ridge_bio | 0.868 | [0.826, 0.899] |
| **MoE / GRN=none** | **0.824** | [0.787, 0.851] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|

## unseen_both  (n_test=359, clusters=33)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | 0.607 | [0.545, 0.659] |
| baseline: B2_ridge | 0.390 | [0.295, 0.478] |
| baseline: B3_ridge_bio | 0.390 | [0.295, 0.478] |
| **MoE / GRN=none** | **0.581** | [0.517, 0.643] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|
