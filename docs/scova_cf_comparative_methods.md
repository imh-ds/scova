# SCOVA-CF two-group comparative methods study

This frozen, descriptive simulation compares SCOVA-CF with interacting linear ANCOVA,
an independently implemented cross-fitted AIPW estimator, propensity-score matching,
and (where installed) EconML DRLearner.

The study has two groups, five baseline covariates, and eight cells crossing linear or
interaction outcome surfaces, linear or nonlinear confounding surfaces, and adequate or
poor overlap. SCOVA-CF, ANCOVA, independent AIPW, and DRLearner target the eligible
study-population ATE. One-to-one propensity-score matching targets the matched-treated
ATT and is displayed separately with its treated-retention fraction.

This is performance evidence within these simulated DGPs only. It does not validate
exchangeability, prove positivity, qualify SCOVA-CF, create a support profile, or
certify causal validity in an applied dataset.
