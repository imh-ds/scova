# SCOVA-CF two-group comparative methods study

Descriptive methods evidence within frozen simulated DGPs only. It does not qualify SCOVA-CF, certify observational causal identification, or create a profile.

- Protocol checksum: `2c5d0646be4008dc9914451b60b786ac81348616abd6815037207552ad1b198f`
- Dependency-lock checksum: `2131bd968e061c12847a100a896903ee1748baa4d5dcdd66d8bcd18a20559f09`
- Frozen commit: `3a614242e7f7b836f2b1fbb0df34dc2c003643bf`
- Execution completeness: `True`

## Study-population ATE estimators

| Method | Bias | RMSE | Coverage | Failure rate | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| econml-drlearner | -18.071 | 114.613 | 0.075 | 0.000 | — |
| independent-aipw | -1.114 | 1.730 | 1.000 | 0.000 | — |
| linear-ancova | 0.059 | 0.117 | 0.900 | 0.000 | — |
| scova-cf | 0.007 | 0.110 | 0.975 | 0.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Bias | RMSE | Coverage | Failure rate | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| psm-att | -0.040 | 0.135 | 0.850 | 0.000 | 0.668 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
