# GRN-quality study (v2, leakage-free, cluster-bootstrap 95% CI)

Data manifest: {"dataset": "tahoe_full", "n_genes": 2000, "n_conditions": 8875, "n_cells_total": 22664673, "n_shards": 803, "genes_sha1": "034932db76c4", "cell_lines": 50}


## unseen_drug  (n_test=1707, clusters=39)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | 0.608 | [0.546, 0.664] |
| baseline: B2_ridge | 0.582 | [0.507, 0.646] |
| baseline: B3_ridge_bio | 0.582 | [0.507, 0.646] |
| **MoE / GRN=none** | **0.630** | [0.552, 0.701] |
| **MoE / GRN=random** | **0.623** | [0.539, 0.696] |
| **MoE / GRN=trrust** | **0.627** | [0.545, 0.699] |
| **MoE / GRN=trrust_weighted** | **0.632** | [0.555, 0.701] |
| **MoE / GRN=coexpr** | **0.626** | [0.543, 0.698] |
| **MoE / GRN=coexpr_lfc** | **0.625** | [0.544, 0.698] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|
| random_vs_none | -0.007 | [-0.015, -0.000] | 0.280 | no |
| trrust_vs_none | -0.003 | [-0.009, +0.004] | 0.844 | no |
| trrust_weighted_vs_none | +0.002 | [-0.000, +0.004] | 0.540 | no |
| coexpr_vs_none | -0.004 | [-0.011, +0.002] | 0.744 | no |
| coexpr_lfc_vs_none | -0.005 | [-0.011, +0.001] | 0.516 | no |
| coexpr_lfc_vs_trrust | -0.003 | [-0.007, +0.001] | 0.744 | no |
| coexpr_lfc_vs_random | +0.002 | [-0.003, +0.007] | 0.844 | no |
| trrust_vs_random | +0.005 | [+0.001, +0.009] | 0.112 | no |
