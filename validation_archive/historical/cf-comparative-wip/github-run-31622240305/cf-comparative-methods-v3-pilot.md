# SCOVA-CF two-group comparative methods study

Descriptive methods evidence within frozen simulated DGPs only. It does not qualify SCOVA-CF, certify observational causal identification, or create a profile.

- Protocol checksum: `d4c3ec0fa98131c4370d3a42b753f1f8c4ffb2ffb1f97021c617fd45404139fe`
- Dependency-lock checksum: `2131bd968e061c12847a100a896903ee1748baa4d5dcdd66d8bcd18a20559f09`
- Frozen commit: `e066558f8d0e02e3b4fe5c19d7d67d8142e5795f`
- Execution completeness: `False`

## Study-population ATE estimators

Failure rate counts records with no numerical estimate or standard error. Warning rate counts retained estimates whose status is not `ok`; it is a diagnostic, not a numerical failure.
Coverage is reported only when the adapter's standard error is an ATE/ATT sampling-uncertainty estimate. DRLearner CATE-spread and matched-pair proxy intervals are retained in the raw artifact but are not evaluated as coverage.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| econml-drlearner | not evaluated | -10.564 | 280.032 | â€” | 0.000 | 0.000 | — |
| econml-drlearner-conservative | not evaluated | -0.010 | 0.365 | â€” | 0.000 | 0.000 | — |
| independent-aipw | influence-function | -0.442 | 1.145 | 0.925 | 0.000 | 0.000 | — |
| linear-ancova | model-based | 0.059 | 0.247 | 0.350 | 0.000 | 0.000 | — |
| scova-cf | influence-function | -0.004 | 0.098 | 0.975 | 0.000 | 1.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psm-att | not evaluated | 0.010 | 0.237 | â€” | 0.000 | 0.000 | 0.712 |

## Cell-level ATE diagnostics

Tail-error columns are descriptive diagnostics, not pass/fail criteria.

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | econml-drlearner | 0.036 | 0.066 | 0.027 | 0.117 | 0.138 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | econml-drlearner-conservative | 0.431 | 0.527 | 0.296 | 0.897 | 1.020 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | independent-aipw | 0.359 | 0.542 | 0.258 | 0.961 | 1.085 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | linear-ancova | -0.189 | 0.193 | 0.195 | 0.231 | 0.233 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | scova-cf | 0.065 | 0.088 | 0.084 | 0.124 | 0.130 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | econml-drlearner | 279.472 | 475.729 | 0.164 | 865.860 | 977.309 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | econml-drlearner-conservative | 0.343 | 0.359 | 0.374 | 0.471 | 0.493 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | independent-aipw | 0.258 | 0.693 | 0.546 | 1.166 | 1.297 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | linear-ancova | -0.418 | 0.432 | 0.458 | 0.553 | 0.574 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | scova-cf | -0.025 | 0.172 | 0.181 | 0.227 | 0.229 |
| cmp-v3-smooth-nonlinear-threshold-adequate | econml-drlearner | 0.052 | 0.091 | 0.097 | 0.130 | 0.136 |
| cmp-v3-smooth-nonlinear-threshold-adequate | econml-drlearner-conservative | -0.200 | 0.457 | 0.263 | 0.741 | 0.797 |
| cmp-v3-smooth-nonlinear-threshold-adequate | independent-aipw | -0.276 | 0.440 | 0.202 | 0.740 | 0.795 |
| cmp-v3-smooth-nonlinear-threshold-adequate | linear-ancova | 0.181 | 0.193 | 0.218 | 0.226 | 0.226 |
| cmp-v3-smooth-nonlinear-threshold-adequate | scova-cf | -0.046 | 0.096 | 0.069 | 0.134 | 0.134 |
| cmp-v3-smooth-nonlinear-threshold-poor | econml-drlearner | -265.482 | 593.626 | 0.081 | 1061.926 | 1327.387 |
| cmp-v3-smooth-nonlinear-threshold-poor | econml-drlearner-conservative | -0.285 | 0.337 | 0.227 | 0.559 | 0.626 |
| cmp-v3-smooth-nonlinear-threshold-poor | independent-aipw | -1.730 | 2.239 | 1.526 | 3.787 | 4.336 |
| cmp-v3-smooth-nonlinear-threshold-poor | linear-ancova | 0.366 | 0.368 | 0.374 | 0.402 | 0.404 |
| cmp-v3-smooth-nonlinear-threshold-poor | scova-cf | 0.044 | 0.080 | 0.063 | 0.115 | 0.117 |
| cmp-v3-threshold-smooth-nonlinear-adequate | econml-drlearner | 0.006 | 0.079 | 0.060 | 0.116 | 0.120 |
| cmp-v3-threshold-smooth-nonlinear-adequate | econml-drlearner-conservative | -0.064 | 0.195 | 0.194 | 0.250 | 0.252 |
| cmp-v3-threshold-smooth-nonlinear-adequate | independent-aipw | -0.026 | 0.222 | 0.220 | 0.334 | 0.350 |
| cmp-v3-threshold-smooth-nonlinear-adequate | linear-ancova | 0.074 | 0.090 | 0.089 | 0.128 | 0.129 |
| cmp-v3-threshold-smooth-nonlinear-adequate | scova-cf | -0.011 | 0.074 | 0.055 | 0.116 | 0.122 |
| cmp-v3-threshold-smooth-nonlinear-poor | econml-drlearner | -98.627 | 220.534 | 0.073 | 394.535 | 493.129 |
| cmp-v3-threshold-smooth-nonlinear-poor | econml-drlearner-conservative | -0.093 | 0.178 | 0.130 | 0.266 | 0.266 |
| cmp-v3-threshold-smooth-nonlinear-poor | independent-aipw | -1.508 | 1.948 | 1.689 | 2.963 | 3.230 |
| cmp-v3-threshold-smooth-nonlinear-poor | linear-ancova | 0.165 | 0.175 | 0.175 | 0.233 | 0.245 |
| cmp-v3-threshold-smooth-nonlinear-poor | scova-cf | -0.008 | 0.090 | 0.035 | 0.147 | 0.153 |
| cmp-v3-threshold-threshold-adequate | econml-drlearner | 0.082 | 0.089 | 0.066 | 0.133 | 0.140 |
| cmp-v3-threshold-threshold-adequate | econml-drlearner-conservative | -0.009 | 0.295 | 0.268 | 0.411 | 0.425 |
| cmp-v3-threshold-threshold-adequate | independent-aipw | -0.066 | 0.420 | 0.511 | 0.521 | 0.521 |
| cmp-v3-threshold-threshold-adequate | linear-ancova | 0.099 | 0.104 | 0.089 | 0.143 | 0.151 |
| cmp-v3-threshold-threshold-adequate | scova-cf | 0.014 | 0.042 | 0.027 | 0.068 | 0.078 |
| cmp-v3-threshold-threshold-poor | econml-drlearner | -0.052 | 0.120 | 0.099 | 0.200 | 0.222 |
| cmp-v3-threshold-threshold-poor | econml-drlearner-conservative | -0.202 | 0.427 | 0.429 | 0.633 | 0.673 |
| cmp-v3-threshold-threshold-poor | independent-aipw | -0.549 | 0.693 | 0.443 | 1.154 | 1.260 |
| cmp-v3-threshold-threshold-poor | linear-ancova | 0.192 | 0.201 | 0.165 | 0.269 | 0.274 |
| cmp-v3-threshold-threshold-poor | scova-cf | -0.064 | 0.094 | 0.062 | 0.161 | 0.185 |

## Cell-level ATT diagnostics

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | psm-att | -0.210 | 0.216 | 0.213 | 0.267 | 0.268 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | psm-att | -0.460 | 0.469 | 0.444 | 0.572 | 0.576 |
| cmp-v3-smooth-nonlinear-threshold-adequate | psm-att | 0.154 | 0.170 | 0.155 | 0.224 | 0.227 |
| cmp-v3-smooth-nonlinear-threshold-poor | psm-att | 0.341 | 0.349 | 0.343 | 0.433 | 0.447 |
| cmp-v3-threshold-smooth-nonlinear-adequate | psm-att | 0.023 | 0.037 | 0.031 | 0.059 | 0.064 |
| cmp-v3-threshold-smooth-nonlinear-poor | psm-att | 0.040 | 0.070 | 0.062 | 0.101 | 0.108 |
| cmp-v3-threshold-threshold-adequate | psm-att | 0.059 | 0.065 | 0.070 | 0.083 | 0.086 |
| cmp-v3-threshold-threshold-poor | psm-att | 0.130 | 0.154 | 0.124 | 0.222 | 0.223 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
