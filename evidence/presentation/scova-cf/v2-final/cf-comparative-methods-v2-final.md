# SCOVA-CF two-group comparative methods study

Descriptive methods evidence within frozen simulated DGPs only. It does not qualify SCOVA-CF, certify observational causal identification, or create a profile.

- Protocol checksum: `322768362888744e23233189a094a9f92534b019fff4a414d5d5abdd91d9424c`
- Dependency-lock checksum: `2131bd968e061c12847a100a896903ee1748baa4d5dcdd66d8bcd18a20559f09`
- Frozen commit: `75525370b067404b61d6843c19c38cae45002bda`
- Execution completeness: `True`

## Study-population ATE estimators

Failure rate counts records with no numerical estimate or standard error. Warning rate counts retained estimates whose status is not `ok`; it is a diagnostic, not a numerical failure.
Coverage is reported only when the adapter's standard error is an ATE/ATT sampling-uncertainty estimate. DRLearner CATE-spread and matched-pair proxy intervals are retained in the raw artifact but are not evaluated as coverage.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| econml-drlearner | not evaluated | -2.107 | 211.002 | â€” | 0.000 | 0.000 | — |
| econml-drlearner-conservative | not evaluated | -0.050 | 0.526 | â€” | 0.000 | 0.000 | — |
| independent-aipw | influence-function | -0.636 | 1.760 | 0.959 | 0.000 | 0.000 | — |
| linear-ancova | model-based | 0.046 | 0.112 | 0.874 | 0.000 | 0.000 | — |
| scova-cf | influence-function | -0.001 | 0.112 | 0.966 | 0.000 | 1.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | Retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| psm-att | not evaluated | -0.036 | 0.144 | â€” | 0.000 | 0.000 | 0.664 |

## Cell-level ATE diagnostics

Tail-error columns are descriptive diagnostics, not pass/fail criteria.

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v1-interaction-linear-adequate | econml-drlearner | 0.019 | 0.091 | 0.061 | 0.178 | 0.303 |
| cmp-v1-interaction-linear-adequate | econml-drlearner-conservative | -0.110 | 0.353 | 0.229 | 0.724 | 1.162 |
| cmp-v1-interaction-linear-adequate | independent-aipw | -0.244 | 0.556 | 0.304 | 1.097 | 4.007 |
| cmp-v1-interaction-linear-adequate | linear-ancova | 0.024 | 0.094 | 0.061 | 0.188 | 0.366 |
| cmp-v1-interaction-linear-adequate | scova-cf | -0.023 | 0.098 | 0.065 | 0.201 | 0.333 |
| cmp-v1-interaction-linear-poor | econml-drlearner | -2.697 | 90.575 | 0.243 | 0.485 | 2864.227 |
| cmp-v1-interaction-linear-poor | econml-drlearner-conservative | -0.559 | 0.801 | 0.587 | 1.515 | 2.202 |
| cmp-v1-interaction-linear-poor | independent-aipw | -2.369 | 3.339 | 2.331 | 6.452 | 13.687 |
| cmp-v1-interaction-linear-poor | linear-ancova | 0.115 | 0.154 | 0.115 | 0.289 | 0.447 |
| cmp-v1-interaction-linear-poor | scova-cf | 0.011 | 0.167 | 0.104 | 0.344 | 0.586 |
| cmp-v1-interaction-nonlinear-adequate | econml-drlearner | 3.254 | 82.603 | 0.068 | 0.195 | 2517.376 |
| cmp-v1-interaction-nonlinear-adequate | econml-drlearner-conservative | 0.122 | 0.313 | 0.212 | 0.621 | 1.137 |
| cmp-v1-interaction-nonlinear-adequate | independent-aipw | -0.050 | 0.435 | 0.226 | 0.850 | 3.348 |
| cmp-v1-interaction-nonlinear-adequate | linear-ancova | 0.075 | 0.118 | 0.084 | 0.228 | 0.326 |
| cmp-v1-interaction-nonlinear-adequate | scova-cf | -0.005 | 0.089 | 0.059 | 0.176 | 0.342 |
| cmp-v1-interaction-nonlinear-poor | econml-drlearner | 16.578 | 448.022 | 0.113 | 978.176 | 4080.791 |
| cmp-v1-interaction-nonlinear-poor | econml-drlearner-conservative | 0.207 | 0.565 | 0.396 | 1.103 | 1.975 |
| cmp-v1-interaction-nonlinear-poor | independent-aipw | -0.332 | 1.812 | 1.079 | 3.829 | 12.026 |
| cmp-v1-interaction-nonlinear-poor | linear-ancova | 0.154 | 0.185 | 0.159 | 0.311 | 0.482 |
| cmp-v1-interaction-nonlinear-poor | scova-cf | -0.002 | 0.131 | 0.084 | 0.256 | 0.533 |
| cmp-v1-linear-linear-adequate | econml-drlearner | 0.002 | 0.068 | 0.046 | 0.135 | 0.225 |
| cmp-v1-linear-linear-adequate | econml-drlearner-conservative | -0.055 | 0.291 | 0.194 | 0.579 | 1.082 |
| cmp-v1-linear-linear-adequate | independent-aipw | -0.212 | 0.444 | 0.252 | 0.954 | 2.384 |
| cmp-v1-linear-linear-adequate | linear-ancova | 0.000 | 0.066 | 0.046 | 0.132 | 0.197 |
| cmp-v1-linear-linear-adequate | scova-cf | -0.001 | 0.076 | 0.051 | 0.148 | 0.249 |
| cmp-v1-linear-linear-poor | econml-drlearner | -0.361 | 60.789 | 0.063 | 0.200 | 1457.287 |
| cmp-v1-linear-linear-poor | econml-drlearner-conservative | -0.509 | 0.721 | 0.521 | 1.370 | 2.306 |
| cmp-v1-linear-linear-poor | independent-aipw | -1.928 | 2.714 | 1.845 | 5.246 | 11.398 |
| cmp-v1-linear-linear-poor | linear-ancova | 0.003 | 0.080 | 0.055 | 0.159 | 0.248 |
| cmp-v1-linear-linear-poor | scova-cf | 0.007 | 0.125 | 0.082 | 0.251 | 0.425 |
| cmp-v1-linear-nonlinear-adequate | econml-drlearner | -1.085 | 34.421 | 0.050 | 0.150 | 1088.489 |
| cmp-v1-linear-nonlinear-adequate | econml-drlearner-conservative | 0.158 | 0.315 | 0.206 | 0.634 | 1.152 |
| cmp-v1-linear-nonlinear-adequate | independent-aipw | 0.014 | 0.387 | 0.207 | 0.696 | 3.182 |
| cmp-v1-linear-nonlinear-adequate | linear-ancova | 0.001 | 0.066 | 0.044 | 0.129 | 0.243 |
| cmp-v1-linear-nonlinear-adequate | scova-cf | 0.002 | 0.076 | 0.051 | 0.149 | 0.244 |
| cmp-v1-linear-nonlinear-poor | econml-drlearner | -32.566 | 368.164 | 0.085 | 794.288 | 2687.545 |
| cmp-v1-linear-nonlinear-poor | econml-drlearner-conservative | 0.346 | 0.574 | 0.400 | 1.135 | 1.816 |
| cmp-v1-linear-nonlinear-poor | independent-aipw | 0.033 | 1.466 | 0.905 | 3.006 | 7.120 |
| cmp-v1-linear-nonlinear-poor | linear-ancova | 0.000 | 0.068 | 0.045 | 0.132 | 0.244 |
| cmp-v1-linear-nonlinear-poor | scova-cf | 0.001 | 0.108 | 0.069 | 0.210 | 0.411 |

## Cell-level ATT diagnostics

| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cmp-v1-interaction-linear-adequate | psm-att | 0.017 | 0.094 | 0.063 | 0.187 | 0.311 |
| cmp-v1-interaction-linear-poor | psm-att | 0.064 | 0.134 | 0.093 | 0.260 | 0.477 |
| cmp-v1-interaction-nonlinear-adequate | psm-att | 0.033 | 0.106 | 0.073 | 0.206 | 0.356 |
| cmp-v1-interaction-nonlinear-poor | psm-att | 0.119 | 0.164 | 0.125 | 0.307 | 0.523 |
| cmp-v1-linear-linear-adequate | psm-att | -0.129 | 0.150 | 0.127 | 0.253 | 0.383 |
| cmp-v1-linear-linear-poor | psm-att | -0.177 | 0.203 | 0.176 | 0.337 | 0.499 |
| cmp-v1-linear-nonlinear-adequate | psm-att | -0.086 | 0.117 | 0.090 | 0.215 | 0.351 |
| cmp-v1-linear-nonlinear-poor | psm-att | -0.126 | 0.152 | 0.128 | 0.259 | 0.465 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
