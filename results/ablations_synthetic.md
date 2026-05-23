# Ablations — unseen_drug TEST set

| Variant | DEG-Pearson@50 | Δ vs full | all-Pearson | dir@50 |
|---|---|---|---|---|
| full | 0.724 | -0.000 | 0.439 | 0.853 |
| no_GRN_mask | 0.644 | -0.080 | 0.385 | 0.823 |
| no_MoE | 0.720 | -0.004 | 0.404 | 0.819 |
| no_pathway_prior | 0.744 | +0.021 | 0.391 | 0.836 |
| pert_as_token | 0.691 | -0.033 | 0.482 | 0.858 |
