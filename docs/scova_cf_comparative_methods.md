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
95th-percentile absolute error, and maximum absolute error. They separately report
numerical failure rate and retained-estimate warning rate: a warning is a diagnostic
flag on an estimate that was still computed, not a refusal. The tail-error columns
are descriptive diagnostics, not qualification gates.

The historical v1 pilot used EconML's auto/2-fold/`min_propensity=1e-6` recipe. The
current v2 design retains that baseline and adds a separately named conservative recipe:
histogram-gradient-boosting nuisance learners, five folds, and `min_propensity=0.01`.
The v1 and v2 results are separate evidence sets and must not be pooled.

This is performance evidence within these simulated DGPs only. It does not validate
exchangeability, prove positivity, qualify SCOVA-CF, create a support profile, or
certify causal validity in an applied dataset.

The `SCOVA-CF two-group comparative methods pilot` GitHub Actions workflow is manual-only,
defaults to five replications per cell, and accepts no more than 50. Its output is explicitly
incomplete methods evidence; the frozen final design remains 1,000 replications per cell.
The separate `SCOVA-CF two-group comparative methods final study` workflow has no replication
input and always runs that frozen 1,000-replication denominator. It uses the same eight-cell
sharding and checksum-bound aggregation as the pilot.

## Completed v2 50-replication pilot

The completed [run 31563232342](https://github.com/imh-ds/scova/actions/runs/31563232342)
executed 50 replications in each of the eight frozen cells (2,400 method records). Its
aggregate artifact checksum is `d28f26b9f4c1bd42bb0ea013a560fbef13351683dfe1e141616f977a33fe6499`.
It is incomplete relative to the predeclared 1,000-replication final denominator and must
not be treated as a definitive ranking.

In this pilot, SCOVA-CF had a 0% numerical failure rate and a 100% retained-estimate warning
rate. The latter is expected: the reset deliberately leaves the default observational support
policy uncalibrated, which produces a support warning without suppressing the estimate. It does
not indicate 400 failed fits. The artifact remains available from the run's
`cf-comparative-methods-v2-pilot` Actions artifact.
