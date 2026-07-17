# SCOVA-CF randomized reference validation

## Current release state

The randomized continuous unnormalized-AIPW implementation is a validation candidate. It is
not yet promoted and all ordinary results remain `unstable` and nonconfirmatory. The existence
of a campaign runner or a pilot result is not validation evidence.

The frozen protocol is `benchmarks/specs/cf_reference_v1.json`. Its checksum identifies the
reference estimand, retained simulation cells, learner policies, numerical gates, and three
non-overlapping seed namespaces:

- pilot: 20 replications per cell;
- calibration: 1,000 replications per cell;
- held-out validation: 2,000 replications per cell.

Thresholds may be selected only from the calibration lane. The held-out lane is evaluated once
against the checksum-bound candidate profile. A smoke override always records
`complete_frozen_lane: false` and cannot promote a profile.

## Locked-seed diagnostic

`SCOVACFDeclaration.stability_seeds` declares additional fold seeds. Five successful full
cross-fit refits are required for promotion eligibility. The primary estimate is unchanged and
the refits are never pooled as extra observations. `SCOVACFResult.seed_stability` reports
estimates, standard errors, intervals, ranges, fold hashes, failures, and the largest departure
from the primary estimate in primary-standard-error units.

Supplied oracle nuisance predictions are labelled `fixed-supplied-nuisance` and cannot satisfy
the refit gate. During implementation, this audit exposed and fixed a CF fold bug: the prior
low-bit XOR salt generally left row ordering unchanged across seeds. The replacement uses a
full 64-bit avalanche mix. This changes CF fold assignments but does not affect base SCOVA.

## Evidence commands

```text
python benchmarks/cf_reference_campaign.py --spec benchmarks/specs/cf_reference_v1.json --lane pilot --output release/artifacts/cf-reference-pilot.json
python benchmarks/cf_reference_campaign.py --spec benchmarks/specs/cf_reference_v1.json --lane calibration --output release/artifacts/cf-reference-calibration-campaign.json
python scripts/calibrate_cf_support.py --spec benchmarks/specs/cf_reference_v1.json --calibration-evidence release/artifacts/cf-reference-calibration-campaign.json --output release/artifacts/cf-reference-calibration.json
python benchmarks/cf_reference_campaign.py --spec benchmarks/specs/cf_reference_v1.json --lane validation --output release/artifacts/cf-reference-validation-campaign.json
python scripts/validate_cf_support.py --spec benchmarks/specs/cf_reference_v1.json --validation-evidence release/artifacts/cf-reference-validation-campaign.json --candidate-profile release/artifacts/cf-reference-candidate-profile.json --output release/artifacts/cf-reference-validation.json --profile-output release/artifacts/cf-reference-support-profile.json
python scripts/check_cf_reference_release.py --spec benchmarks/specs/cf_reference_v1.json --evidence-root release/artifacts
```

Install the optional `cf-validation` extra for DoubleMLAPOS and EconML DRLearner comparisons.
Generate their checksum-bound evidence with
`python benchmarks/cf_external_agreement.py --output
release/artifacts/cf-reference-external-agreement.json`.
The release checker fails closed on missing artifacts, checksum mismatches, incomplete or failed
external comparisons, an unpromoted profile, or failed operating-characteristic gates.

## Scientific boundary

This campaign validates population counterfactual means and their contrasts. It does not
validate respondent-specific missing outcomes, paired tests on predictions, within-person
covariance, or individual treatment effects.
