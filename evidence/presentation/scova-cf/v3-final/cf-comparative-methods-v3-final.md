# SCOVA-CF two-group comparative methods study

Descriptive methods evidence within frozen simulated DGPs only. It does not qualify SCOVA-CF, certify observational causal identification, or create a profile.

- Protocol checksum: `d4c3ec0fa98131c4370d3a42b753f1f8c4ffb2ffb1f97021c617fd45404139fe`
- Dependency-lock checksum: `2131bd968e061c12847a100a896903ee1748baa4d5dcdd66d8bcd18a20559f09`
- Frozen commit: `84d15b51f3908f783c99232031279eafb3eeab9e`
- Execution completeness: `True`

## Study-population ATE estimators

Failure rate counts records with no numerical estimate or standard error. Warning rate counts retained estimates whose status is not `ok`; it is a diagnostic, not a numerical failure.
Coverage is reported only when the adapter's standard error is an ATE/ATT sampling-uncertainty estimate. DRLearner CATE-spread and matched-pair proxy intervals are retained in the raw artifact but are not evaluated as coverage.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| econml-drlearner | not evaluated | -25.861 | 357.915 | â€” | 0.000 | 0.000 | — |
| econml-drlearner-conservative | not evaluated | 0.046 | 0.431 | â€” | 0.000 | 0.000 | — |
| independent-aipw | influence-function | -0.027 | 1.146 | 0.970 | 0.000 | 0.000 | — |
| linear-ancova | model-based | 0.060 | 0.266 | 0.357 | 0.000 | 0.000 | — |
| scova-cf | influence-function | 0.000 | 0.105 | 0.959 | 0.000 | 1.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psm-att | not evaluated | 0.002 | 0.277 | â€” | 0.000 | 0.000 | 0.717 |

## Cell-level ATE diagnostics

Tail-error columns are descriptive diagnostics, not pass/fail criteria.

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | econml-drlearner | 2.506 | 55.400 | 0.068 | 0.180 | 1304.505 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | econml-drlearner-conservative | 0.196 | 0.357 | 0.250 | 0.701 | 1.277 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | independent-aipw | 0.203 | 0.484 | 0.252 | 0.984 | 3.463 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | linear-ancova | -0.232 | 0.245 | 0.233 | 0.362 | 0.469 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | scova-cf | 0.002 | 0.087 | 0.061 | 0.176 | 0.300 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | econml-drlearner | 22.888 | 348.154 | 0.108 | 624.489 | 3111.626 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | econml-drlearner-conservative | 0.440 | 0.640 | 0.463 | 1.202 | 2.254 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | independent-aipw | 0.986 | 1.807 | 1.071 | 3.734 | 8.823 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | linear-ancova | -0.409 | 0.419 | 0.411 | 0.556 | 0.669 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | scova-cf | -0.020 | 0.124 | 0.086 | 0.244 | 0.425 |
| cmp-v3-smooth-nonlinear-threshold-adequate | econml-drlearner | -0.507 | 18.082 | 0.083 | 0.197 | 571.808 |
| cmp-v3-smooth-nonlinear-threshold-adequate | econml-drlearner-conservative | -0.046 | 0.284 | 0.192 | 0.548 | 0.990 |
| cmp-v3-smooth-nonlinear-threshold-adequate | independent-aipw | -0.096 | 0.343 | 0.217 | 0.669 | 1.641 |
| cmp-v3-smooth-nonlinear-threshold-adequate | linear-ancova | 0.178 | 0.195 | 0.180 | 0.311 | 0.469 |
| cmp-v3-smooth-nonlinear-threshold-adequate | scova-cf | -0.003 | 0.089 | 0.060 | 0.175 | 0.293 |
| cmp-v3-smooth-nonlinear-threshold-poor | econml-drlearner | -109.907 | 659.913 | 0.173 | 1568.891 | 5627.517 |
| cmp-v3-smooth-nonlinear-threshold-poor | econml-drlearner-conservative | -0.275 | 0.554 | 0.377 | 1.098 | 1.896 |
| cmp-v3-smooth-nonlinear-threshold-poor | independent-aipw | -0.878 | 1.724 | 1.079 | 3.499 | 6.751 |
| cmp-v3-smooth-nonlinear-threshold-poor | linear-ancova | 0.427 | 0.438 | 0.427 | 0.578 | 0.729 |
| cmp-v3-smooth-nonlinear-threshold-poor | scova-cf | 0.023 | 0.132 | 0.087 | 0.265 | 0.467 |
| cmp-v3-threshold-smooth-nonlinear-adequate | econml-drlearner | 0.456 | 78.453 | 0.055 | 0.163 | 1846.064 |
| cmp-v3-threshold-smooth-nonlinear-adequate | econml-drlearner-conservative | 0.022 | 0.266 | 0.180 | 0.527 | 0.828 |
| cmp-v3-threshold-smooth-nonlinear-adequate | independent-aipw | -0.047 | 0.354 | 0.195 | 0.683 | 2.219 |
| cmp-v3-threshold-smooth-nonlinear-adequate | linear-ancova | 0.066 | 0.092 | 0.070 | 0.173 | 0.256 |
| cmp-v3-threshold-smooth-nonlinear-adequate | scova-cf | -0.004 | 0.080 | 0.052 | 0.160 | 0.281 |
| cmp-v3-threshold-smooth-nonlinear-poor | econml-drlearner | -27.434 | 357.265 | 0.091 | 771.260 | 2887.957 |
| cmp-v3-threshold-smooth-nonlinear-poor | econml-drlearner-conservative | 0.048 | 0.459 | 0.301 | 0.915 | 1.544 |
| cmp-v3-threshold-smooth-nonlinear-poor | independent-aipw | -0.068 | 1.416 | 0.831 | 2.845 | 6.124 |
| cmp-v3-threshold-smooth-nonlinear-poor | linear-ancova | 0.157 | 0.171 | 0.156 | 0.269 | 0.375 |
| cmp-v3-threshold-smooth-nonlinear-poor | scova-cf | 0.024 | 0.116 | 0.072 | 0.228 | 0.409 |
| cmp-v3-threshold-threshold-adequate | econml-drlearner | 2.604 | 80.677 | 0.070 | 0.190 | 2551.236 |
| cmp-v3-threshold-threshold-adequate | econml-drlearner-conservative | 0.021 | 0.285 | 0.188 | 0.554 | 1.103 |
| cmp-v3-threshold-threshold-adequate | independent-aipw | -0.041 | 0.359 | 0.221 | 0.715 | 2.206 |
| cmp-v3-threshold-threshold-adequate | linear-ancova | 0.084 | 0.109 | 0.083 | 0.200 | 0.294 |
| cmp-v3-threshold-threshold-adequate | scova-cf | -0.007 | 0.081 | 0.054 | 0.160 | 0.272 |
| cmp-v3-threshold-threshold-poor | econml-drlearner | -97.492 | 569.587 | 0.122 | 1388.076 | 4693.766 |
| cmp-v3-threshold-threshold-poor | econml-drlearner-conservative | -0.033 | 0.448 | 0.295 | 0.915 | 1.420 |
| cmp-v3-threshold-threshold-poor | independent-aipw | -0.275 | 1.291 | 0.770 | 2.568 | 5.089 |
| cmp-v3-threshold-threshold-poor | linear-ancova | 0.211 | 0.225 | 0.211 | 0.336 | 0.487 |
| cmp-v3-threshold-threshold-poor | scova-cf | -0.013 | 0.114 | 0.071 | 0.224 | 0.435 |

## Cell-level ATT diagnostics

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | psm-att | -0.260 | 0.274 | 0.261 | 0.406 | 0.555 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | psm-att | -0.526 | 0.535 | 0.528 | 0.680 | 0.850 |
| cmp-v3-smooth-nonlinear-threshold-adequate | psm-att | 0.162 | 0.184 | 0.161 | 0.302 | 0.479 |
| cmp-v3-smooth-nonlinear-threshold-poor | psm-att | 0.392 | 0.406 | 0.395 | 0.561 | 0.819 |
| cmp-v3-threshold-smooth-nonlinear-adequate | psm-att | 0.011 | 0.073 | 0.049 | 0.143 | 0.250 |
| cmp-v3-threshold-smooth-nonlinear-poor | psm-att | 0.032 | 0.087 | 0.058 | 0.172 | 0.323 |
| cmp-v3-threshold-threshold-adequate | psm-att | 0.051 | 0.093 | 0.062 | 0.183 | 0.339 |
| cmp-v3-threshold-threshold-poor | psm-att | 0.158 | 0.183 | 0.159 | 0.310 | 0.475 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
