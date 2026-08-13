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
| econml-drlearner | not evaluated | -28.782 | 409.758 | â€” | 0.000 | 0.000 | — |
| econml-drlearner-conservative | not evaluated | 0.041 | 0.387 | â€” | 0.000 | 0.000 | — |
| independent-aipw | influence-function | -0.119 | 1.053 | 0.980 | 0.000 | 0.000 | — |
| linear-ancova | model-based | 0.065 | 0.265 | 0.355 | 0.000 | 0.000 | — |
| scova-cf | influence-function | -0.000 | 0.103 | 0.970 | 0.000 | 1.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psm-att | not evaluated | 0.007 | 0.275 | â€” | 0.000 | 0.000 | 0.718 |

## Cell-level ATE diagnostics

Tail-error columns are descriptive diagnostics, not pass/fail criteria.

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | econml-drlearner | -0.029 | 0.111 | 0.067 | 0.209 | 0.352 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | econml-drlearner-conservative | 0.181 | 0.314 | 0.238 | 0.550 | 1.020 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | independent-aipw | 0.168 | 0.399 | 0.239 | 0.941 | 1.207 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | linear-ancova | -0.220 | 0.236 | 0.224 | 0.322 | 0.469 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | scova-cf | 0.002 | 0.087 | 0.052 | 0.170 | 0.194 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | econml-drlearner | 89.700 | 384.183 | 0.106 | 959.129 | 1765.202 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | econml-drlearner-conservative | 0.313 | 0.509 | 0.373 | 1.006 | 1.188 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | independent-aipw | 0.620 | 1.664 | 0.847 | 3.161 | 6.362 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | linear-ancova | -0.393 | 0.402 | 0.391 | 0.532 | 0.610 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | scova-cf | -0.026 | 0.138 | 0.101 | 0.250 | 0.425 |
| cmp-v3-smooth-nonlinear-threshold-adequate | econml-drlearner | 0.048 | 0.110 | 0.093 | 0.181 | 0.217 |
| cmp-v3-smooth-nonlinear-threshold-adequate | econml-drlearner-conservative | -0.097 | 0.299 | 0.208 | 0.599 | 0.801 |
| cmp-v3-smooth-nonlinear-threshold-adequate | independent-aipw | -0.148 | 0.327 | 0.205 | 0.652 | 0.828 |
| cmp-v3-smooth-nonlinear-threshold-adequate | linear-ancova | 0.175 | 0.203 | 0.185 | 0.333 | 0.350 |
| cmp-v3-smooth-nonlinear-threshold-adequate | scova-cf | -0.009 | 0.090 | 0.070 | 0.150 | 0.217 |
| cmp-v3-smooth-nonlinear-threshold-poor | econml-drlearner | -214.083 | 848.864 | 0.156 | 1860.443 | 3200.481 |
| cmp-v3-smooth-nonlinear-threshold-poor | econml-drlearner-conservative | -0.288 | 0.539 | 0.388 | 0.975 | 1.340 |
| cmp-v3-smooth-nonlinear-threshold-poor | independent-aipw | -0.833 | 1.531 | 1.101 | 2.930 | 4.336 |
| cmp-v3-smooth-nonlinear-threshold-poor | linear-ancova | 0.439 | 0.447 | 0.437 | 0.576 | 0.629 |
| cmp-v3-smooth-nonlinear-threshold-poor | scova-cf | 0.021 | 0.111 | 0.061 | 0.222 | 0.310 |
| cmp-v3-threshold-smooth-nonlinear-adequate | econml-drlearner | 0.033 | 0.086 | 0.069 | 0.142 | 0.243 |
| cmp-v3-threshold-smooth-nonlinear-adequate | econml-drlearner-conservative | 0.100 | 0.268 | 0.189 | 0.523 | 0.743 |
| cmp-v3-threshold-smooth-nonlinear-adequate | independent-aipw | 0.055 | 0.379 | 0.164 | 0.731 | 1.626 |
| cmp-v3-threshold-smooth-nonlinear-adequate | linear-ancova | 0.071 | 0.095 | 0.079 | 0.158 | 0.238 |
| cmp-v3-threshold-smooth-nonlinear-adequate | scova-cf | 0.008 | 0.080 | 0.054 | 0.142 | 0.200 |
| cmp-v3-threshold-smooth-nonlinear-poor | econml-drlearner | -1.558 | 210.183 | 0.127 | 558.158 | 1019.473 |
| cmp-v3-threshold-smooth-nonlinear-poor | econml-drlearner-conservative | 0.041 | 0.361 | 0.233 | 0.728 | 0.960 |
| cmp-v3-threshold-smooth-nonlinear-poor | independent-aipw | -0.363 | 1.313 | 0.828 | 2.827 | 3.505 |
| cmp-v3-threshold-smooth-nonlinear-poor | linear-ancova | 0.163 | 0.178 | 0.154 | 0.285 | 0.308 |
| cmp-v3-threshold-smooth-nonlinear-poor | scova-cf | 0.004 | 0.111 | 0.073 | 0.227 | 0.271 |
| cmp-v3-threshold-threshold-adequate | econml-drlearner | 0.062 | 0.114 | 0.081 | 0.193 | 0.398 |
| cmp-v3-threshold-threshold-adequate | econml-drlearner-conservative | 0.044 | 0.226 | 0.161 | 0.447 | 0.561 |
| cmp-v3-threshold-threshold-adequate | independent-aipw | 0.008 | 0.277 | 0.194 | 0.538 | 0.563 |
| cmp-v3-threshold-threshold-adequate | linear-ancova | 0.076 | 0.104 | 0.079 | 0.182 | 0.249 |
| cmp-v3-threshold-threshold-adequate | scova-cf | -0.003 | 0.083 | 0.067 | 0.150 | 0.194 |
| cmp-v3-threshold-threshold-poor | econml-drlearner | -104.429 | 656.407 | 0.162 | 1405.983 | 2991.814 |
| cmp-v3-threshold-threshold-poor | econml-drlearner-conservative | 0.030 | 0.455 | 0.262 | 0.891 | 1.393 |
| cmp-v3-threshold-threshold-poor | independent-aipw | -0.459 | 1.246 | 0.899 | 2.280 | 2.940 |
| cmp-v3-threshold-threshold-poor | linear-ancova | 0.212 | 0.227 | 0.213 | 0.339 | 0.400 |
| cmp-v3-threshold-threshold-poor | scova-cf | -0.000 | 0.113 | 0.067 | 0.210 | 0.323 |

## Cell-level ATT diagnostics

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-adequate | psm-att | -0.252 | 0.269 | 0.268 | 0.358 | 0.459 |
| cmp-v3-smooth-nonlinear-smooth-nonlinear-poor | psm-att | -0.507 | 0.515 | 0.498 | 0.639 | 0.719 |
| cmp-v3-smooth-nonlinear-threshold-adequate | psm-att | 0.150 | 0.182 | 0.158 | 0.293 | 0.323 |
| cmp-v3-smooth-nonlinear-threshold-poor | psm-att | 0.414 | 0.426 | 0.421 | 0.563 | 0.628 |
| cmp-v3-threshold-smooth-nonlinear-adequate | psm-att | 0.011 | 0.074 | 0.056 | 0.135 | 0.180 |
| cmp-v3-threshold-smooth-nonlinear-poor | psm-att | 0.038 | 0.089 | 0.069 | 0.157 | 0.219 |
| cmp-v3-threshold-threshold-adequate | psm-att | 0.048 | 0.087 | 0.063 | 0.163 | 0.193 |
| cmp-v3-threshold-threshold-poor | psm-att | 0.154 | 0.179 | 0.170 | 0.306 | 0.350 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
