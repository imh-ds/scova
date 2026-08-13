# SCOVA-CF two-group comparative methods study

Descriptive methods evidence within frozen simulated DGPs only. It does not qualify SCOVA-CF, certify observational causal identification, or create a profile.

- Protocol checksum: `322768362888744e23233189a094a9f92534b019fff4a414d5d5abdd91d9424c`
- Dependency-lock checksum: `2131bd968e061c12847a100a896903ee1748baa4d5dcdd66d8bcd18a20559f09`
- Frozen commit: `d55158bc3bedbf4fcdf08b0db773efe7f387c65e`
- Execution completeness: `False`

## Study-population ATE estimators

Failure rate counts records with no numerical estimate or standard error. Warning rate counts retained estimates whose status is not `ok`; it is a diagnostic, not a numerical failure.

| Method | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| econml-drlearner | -0.378 | 286.909 | 0.138 | 0.000 | 0.000 | — |
| econml-drlearner-conservative | -0.041 | 0.559 | 0.098 | 0.000 | 0.000 | — |
| independent-aipw | -0.606 | 1.802 | 0.943 | 0.000 | 0.000 | — |
| linear-ancova | 0.044 | 0.115 | 0.858 | 0.000 | 0.000 | — |
| scova-cf | -0.003 | 0.105 | 0.980 | 0.000 | 1.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psm-att | -0.043 | 0.150 | 0.790 | 0.000 | 0.000 | 0.665 |

## Cell-level ATE diagnostics

Tail-error columns are descriptive diagnostics, not pass/fail criteria.

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v1-interaction-linear-adequate | econml-drlearner | 0.036 | 0.095 | 0.054 | 0.181 | 0.199 |
| cmp-v1-interaction-linear-adequate | econml-drlearner-conservative | -0.023 | 0.335 | 0.258 | 0.590 | 0.804 |
| cmp-v1-interaction-linear-adequate | independent-aipw | -0.137 | 0.589 | 0.247 | 0.992 | 2.378 |
| cmp-v1-interaction-linear-adequate | linear-ancova | 0.019 | 0.102 | 0.073 | 0.192 | 0.227 |
| cmp-v1-interaction-linear-adequate | scova-cf | -0.006 | 0.080 | 0.055 | 0.154 | 0.220 |
| cmp-v1-interaction-linear-poor | econml-drlearner | 0.163 | 0.306 | 0.226 | 0.482 | 1.270 |
| cmp-v1-interaction-linear-poor | econml-drlearner-conservative | -0.616 | 0.841 | 0.567 | 1.482 | 1.535 |
| cmp-v1-interaction-linear-poor | independent-aipw | -2.265 | 3.239 | 2.488 | 5.281 | 7.797 |
| cmp-v1-interaction-linear-poor | linear-ancova | 0.108 | 0.154 | 0.104 | 0.310 | 0.356 |
| cmp-v1-interaction-linear-poor | scova-cf | 0.004 | 0.150 | 0.108 | 0.282 | 0.359 |
| cmp-v1-interaction-nonlinear-adequate | econml-drlearner | 0.030 | 0.103 | 0.089 | 0.188 | 0.215 |
| cmp-v1-interaction-nonlinear-adequate | econml-drlearner-conservative | 0.115 | 0.329 | 0.197 | 0.637 | 1.129 |
| cmp-v1-interaction-nonlinear-adequate | independent-aipw | -0.006 | 0.330 | 0.188 | 0.701 | 0.784 |
| cmp-v1-interaction-nonlinear-adequate | linear-ancova | 0.067 | 0.109 | 0.079 | 0.200 | 0.249 |
| cmp-v1-interaction-nonlinear-adequate | scova-cf | 0.005 | 0.086 | 0.052 | 0.191 | 0.220 |
| cmp-v1-interaction-nonlinear-poor | econml-drlearner | -48.187 | 747.211 | 0.123 | 1559.584 | 4080.791 |
| cmp-v1-interaction-nonlinear-poor | econml-drlearner-conservative | 0.177 | 0.628 | 0.455 | 1.084 | 1.319 |
| cmp-v1-interaction-nonlinear-poor | independent-aipw | -0.647 | 2.293 | 1.110 | 3.214 | 12.026 |
| cmp-v1-interaction-nonlinear-poor | linear-ancova | 0.175 | 0.204 | 0.194 | 0.311 | 0.329 |
| cmp-v1-interaction-nonlinear-poor | scova-cf | -0.006 | 0.139 | 0.095 | 0.260 | 0.310 |
| cmp-v1-linear-linear-adequate | econml-drlearner | 0.004 | 0.066 | 0.043 | 0.124 | 0.157 |
| cmp-v1-linear-linear-adequate | econml-drlearner-conservative | 0.023 | 0.260 | 0.168 | 0.516 | 0.695 |
| cmp-v1-linear-linear-adequate | independent-aipw | -0.080 | 0.438 | 0.243 | 0.973 | 1.197 |
| cmp-v1-linear-linear-adequate | linear-ancova | -0.000 | 0.068 | 0.047 | 0.131 | 0.171 |
| cmp-v1-linear-linear-adequate | scova-cf | 0.002 | 0.068 | 0.049 | 0.140 | 0.163 |
| cmp-v1-linear-linear-poor | econml-drlearner | 0.002 | 0.087 | 0.061 | 0.163 | 0.203 |
| cmp-v1-linear-linear-poor | econml-drlearner-conservative | -0.547 | 0.769 | 0.459 | 1.351 | 2.306 |
| cmp-v1-linear-linear-poor | independent-aipw | -1.995 | 2.629 | 1.792 | 5.072 | 6.246 |
| cmp-v1-linear-linear-poor | linear-ancova | 0.001 | 0.070 | 0.054 | 0.128 | 0.170 |
| cmp-v1-linear-linear-poor | scova-cf | 0.000 | 0.104 | 0.062 | 0.215 | 0.260 |
| cmp-v1-linear-nonlinear-adequate | econml-drlearner | -0.007 | 0.072 | 0.043 | 0.155 | 0.178 |
| cmp-v1-linear-nonlinear-adequate | econml-drlearner-conservative | 0.137 | 0.333 | 0.238 | 0.558 | 0.757 |
| cmp-v1-linear-nonlinear-adequate | independent-aipw | -0.043 | 0.350 | 0.257 | 0.636 | 1.055 |
| cmp-v1-linear-nonlinear-adequate | linear-ancova | -0.011 | 0.067 | 0.048 | 0.132 | 0.153 |
| cmp-v1-linear-nonlinear-adequate | scova-cf | -0.014 | 0.080 | 0.054 | 0.139 | 0.207 |
| cmp-v1-linear-nonlinear-poor | econml-drlearner | 44.937 | 316.557 | 0.070 | 502.338 | 1635.679 |
| cmp-v1-linear-nonlinear-poor | econml-drlearner-conservative | 0.411 | 0.641 | 0.484 | 1.184 | 1.296 |
| cmp-v1-linear-nonlinear-poor | independent-aipw | 0.325 | 1.596 | 1.012 | 2.928 | 4.379 |
| cmp-v1-linear-nonlinear-poor | linear-ancova | -0.010 | 0.054 | 0.033 | 0.106 | 0.151 |
| cmp-v1-linear-nonlinear-poor | scova-cf | -0.006 | 0.099 | 0.061 | 0.202 | 0.253 |

## Cell-level ATT diagnostics

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v1-interaction-linear-adequate | psm-att | 0.016 | 0.107 | 0.073 | 0.200 | 0.231 |
| cmp-v1-interaction-linear-poor | psm-att | 0.044 | 0.123 | 0.075 | 0.231 | 0.287 |
| cmp-v1-interaction-nonlinear-adequate | psm-att | 0.019 | 0.111 | 0.079 | 0.213 | 0.229 |
| cmp-v1-interaction-nonlinear-poor | psm-att | 0.134 | 0.187 | 0.150 | 0.327 | 0.398 |
| cmp-v1-linear-linear-adequate | psm-att | -0.133 | 0.149 | 0.141 | 0.231 | 0.273 |
| cmp-v1-linear-linear-poor | psm-att | -0.177 | 0.201 | 0.177 | 0.320 | 0.499 |
| cmp-v1-linear-nonlinear-adequate | psm-att | -0.108 | 0.135 | 0.105 | 0.267 | 0.294 |
| cmp-v1-linear-nonlinear-poor | psm-att | -0.139 | 0.160 | 0.148 | 0.258 | 0.297 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
