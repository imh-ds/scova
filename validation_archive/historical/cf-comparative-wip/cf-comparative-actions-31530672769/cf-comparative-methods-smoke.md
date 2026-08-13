# SCOVA-CF two-group comparative methods study

Descriptive methods evidence within frozen simulated DGPs only. It does not qualify SCOVA-CF, certify observational causal identification, or create a profile.

- Protocol checksum: `2c5d0646be4008dc9914451b60b786ac81348616abd6815037207552ad1b198f`
- Dependency-lock checksum: `2131bd968e061c12847a100a896903ee1748baa4d5dcdd66d8bcd18a20559f09`
- Frozen commit: `4771d9e05616c2490de6970425490e820a0fa349`
- Execution completeness: `False`

## Study-population ATE estimators

| Method | Bias | RMSE | Coverage | Failure rate | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| econml-drlearner | -0.378 | 286.909 | 0.138 | 0.000 | — |
| independent-aipw | -0.606 | 1.802 | 0.943 | 0.000 | — |
| linear-ancova | 0.044 | 0.115 | 0.858 | 0.000 | — |
| scova-cf | -0.003 | 0.105 | 0.980 | 0.000 | — |

## Matched-treated ATT estimator

PSM estimates the ATT among retained matched treated units; it is not ranked against the ATE estimators.

| Method | Bias | RMSE | Coverage | Failure rate | Retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| psm-att | -0.043 | 0.150 | 0.790 | 0.000 | 0.665 |

Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.
