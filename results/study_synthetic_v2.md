# GRN-quality study (v2, leakage-free, cluster-bootstrap 95% CI)

Data manifest: {"dataset": "synthetic", "n_genes": 2000, "n_conditions": 1410, "n_cells_total": null, "n_shards": null, "genes_sha1": "73fd13e31c46", "cell_lines": 10}


## unseen_drug  (n_test=270, clusters=9)

| Model | DEG-Pearson@50 | 95% CI (cluster) |
|---|---|---|
| baseline: B1_mean_effect | 0.595 | [0.412, 0.771] |
| baseline: B2_ridge | -0.048 | [-0.375, 0.288] |
| baseline: B3_ridge_bio | 0.631 | [0.431, 0.795] |
| **MoE / GRN=none** | **0.596** | [0.305, 0.856] |
| **MoE / GRN=random** | **0.595** | [0.308, 0.858] |
| **MoE / GRN=coexpr** | **0.593** | [0.298, 0.853] |
| **MoE / GRN=coexpr_lfc** | **0.594** | [0.304, 0.852] |
| **MoE / GRN=ground_truth** | **0.610** | [0.319, 0.869] |
| GRN-prop / GRN=random | 0.167 | [0.093, 0.232] |
| GRN-prop / GRN=coexpr | 0.167 | [0.093, 0.232] |
| GRN-prop / GRN=coexpr_lfc | 0.167 | [0.093, 0.232] |
| GRN-prop / GRN=ground_truth | 0.158 | [0.081, 0.243] |

**Contrasts (paired cluster bootstrap, Holm-corrected):**

| contrast | Δ DEG-Pearson | 95% CI | p (Holm) | significant&meaningful |
|---|---|---|---|---|
| random_vs_none | -0.001 | [-0.012, +0.010] | 1.000 | no |
| coexpr_vs_none | -0.003 | [-0.014, +0.006] | 1.000 | no |
| coexpr_lfc_vs_none | -0.002 | [-0.011, +0.007] | 1.000 | no |
| ground_truth_vs_none | +0.014 | [-0.001, +0.029] | 0.340 | no |
| coexpr_lfc_vs_random | -0.001 | [-0.020, +0.015] | 1.000 | no |
