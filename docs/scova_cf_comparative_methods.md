# SCOVA-CF two-group comparative methods study

This frozen, descriptive simulation compares SCOVA-CF with interacting linear ANCOVA,
an independently implemented cross-fitted AIPW estimator, propensity-score matching,
and (where installed) EconML DRLearner.

The study has two groups, five baseline covariates, and eight cells crossing linear or
interaction outcome surfaces, linear or nonlinear confounding surfaces, and adequate or
poor overlap. SCOVA-CF, ANCOVA, independent AIPW, and DRLearner target the eligible
study-population ATE. One-to-one propensity-score matching targets the matched-treated
ATT and is displayed separately with its treated-retention fraction.

Reports present pooled summaries and cell-level bias, RMSE, median absolute error,
95th-percentile absolute error, and maximum absolute error. The tail-error columns
are descriptive diagnostics, not qualification gates.

The current EconML baseline records its exact v0.16 recipe: automatic propensity and
outcome nuisance models, two internal cross-fitting folds, and `min_propensity=1e-6`.
It is a frozen baseline recipe, not a claim that it is optimal or stable in every DGP.

This is performance evidence within these simulated DGPs only. It does not validate
exchangeability, prove positivity, qualify SCOVA-CF, create a support profile, or
certify causal validity in an applied dataset.

The `SCOVA-CF two-group comparative methods smoke` GitHub Actions workflow is manual-only,
defaults to five replications per cell, and accepts no more than 25. Its output is explicitly
incomplete methods evidence; the frozen final design remains 1,000 replications per cell.
