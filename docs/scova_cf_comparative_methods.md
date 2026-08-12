# SCOVA-CF two-group comparative methods study

This frozen, descriptive simulation compares SCOVA-CF with interacting linear ANCOVA,
an independently implemented cross-fitted AIPW estimator, propensity-score matching,
and (where installed) EconML DRLearner.

The current v2 study has two groups, five baseline covariates, and eight cells crossing linear or
interaction outcome surfaces, linear or nonlinear confounding surfaces, and adequate or
poor overlap. SCOVA-CF, ANCOVA, independent AIPW, and DRLearner target the eligible
study-population ATE. One-to-one propensity-score matching targets the matched-treated
ATT and is displayed separately with its treated-retention fraction.

Reports present pooled summaries and cell-level bias, RMSE, median absolute error,
95th-percentile absolute error, and maximum absolute error. The tail-error columns
are descriptive diagnostics, not qualification gates.

The historical v1 pilot used EconML's auto/2-fold/`min_propensity=1e-6` recipe. The
current v2 design retains that baseline and adds a separately named conservative recipe:
histogram-gradient-boosting nuisance learners, five folds, and `min_propensity=0.01`.
The v1 and v2 results are separate evidence sets and must not be pooled.

This is performance evidence within these simulated DGPs only. It does not validate
exchangeability, prove positivity, qualify SCOVA-CF, create a support profile, or
certify causal validity in an applied dataset.

The `SCOVA-CF two-group comparative methods smoke` GitHub Actions workflow is manual-only,
defaults to five replications per cell, and accepts no more than 25. Its output is explicitly
incomplete methods evidence; the frozen final design remains 1,000 replications per cell.
