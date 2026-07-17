# SCOVA-CF reference protocol v2 blocking report

## Decision

Protocol `cf-randomized-continuous-aipw-unnormalized-v2` is blocked and must not be used for
calibration, validation, support-profile packaging, or promotion. No v2 held-out validation
namespace was evaluated, and no calibrated profile was issued.

## Failure discovered

The protocol declared the validation environment's scikit-learn version as 1.6.1 while its
plasmode source checksums had been generated using scikit-learn 1.9.0. Installing the declared
version in GitHub Actions exposed the inconsistency:

| Dataset | Frozen v2 checksum | Checksum under declared 1.6.1 |
|---|---|---|
| Diabetes | `e9f4db8517bc1152ebec45b3276b3c577d840920cb8c7047d1ca775ca0a6bfda` | `28128d0ec207a1c0ac5e23a5fdbad720215a35b5f18741fef26c4ecd254dc278` |
| Breast cancer | `762833f63f3b53f6308f01f8248592c137c0ea6a392a407e52c163af82589d2c` | `ba4d19cf8137014a7cbcc9f1d625891c176da5a3dbf2af07333081d81f57e90f` |

This was not only a description-text difference. The diabetes covariate array itself has a
different byte identity between the two scikit-learn versions. Replacing the dependency pin or
the checksums in place would therefore misstate or alter the frozen v2 protocol.

## Disposition

Protocol v3 pins the originally declared scientific environment, adopts the source identities
actually produced by that environment, and assigns new pilot, calibration, validation,
external-comparison, and simultaneous-inference seed namespaces. All workflows, freeze tags,
tests, and release checks bind to v3. The v2 diagnosis remains auditable through this report and
repository history; v2 evidence is inadmissible for v3 promotion.
