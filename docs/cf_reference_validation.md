# SCOVA-CF randomized reference validation v2

## Release state

The randomized continuous unnormalized-AIPW profile remains a validation candidate. No
calibrated profile is bundled in this source tree, so ordinary SCOVA-CF analyses remain
`unstable` and nonconfirmatory. Passing unit tests or a pilot does not promote the method.
`SupportPolicy.packaged(profile_id)` is the sole calibrated activation route and currently
rejects every identifier.

The frozen manifest is `benchmarks/specs/cf_reference_v1.json` (the historical filename is
retained for compatibility; its schema and protocol ID are v2). Its checksum locks:

- 48 pairwise-covering simulation cells and 12 plasmode cells;
- six focused simultaneous-inference cells and eight external-comparison cells;
- pilot, calibration, held-out validation, inference, and external seed namespaces;
- learner policies, operating gates, exact software versions, dependency-lock checksum, and
  diabetes and breast-cancer source-array checksums.

Calibration uses 1,000 replications per cell. Only its first 60% may fit thresholds; its final
40% is an internal audit. Held-out validation uses a separate 2,000-replication namespace and
may be evaluated only once. A failure requires a v3 protocol and fresh validation seeds.

## What is validated

The profile targets randomized, independent units, mutually exclusive groups, one materially
complete continuous outcome, known assignment probabilities, study-population standardized
counterfactual means, and reference-group contrasts. It retains all declared contrasts, joint
covariance, guarded omnibus inference, pointwise intervals, and unadjusted and Lin-interacted
benchmarks.

The plasmodes use unique rows sampled without replacement from scikit-learn's bundled diabetes
and Wisconsin breast-cancer covariates. Their original targets are standardized prognostic
baselines; randomized treatment effects and fresh continuous noise are injected. They test
realistic covariate dependence and make no observational or clinical claim about either source.

This campaign does not validate respondent-specific missing outcomes, paired tests on predicted
outcomes, within-person covariance, individual treatment effects, associational analyses, or
observational-causal analyses.

## Evidence workflow

The manual GitHub Actions workflow `.github/workflows/cf-reference-validation.yml` exposes the
tiers `pilot`, `calibration`, `calibrate_support`, `validation`, `simultaneous_inference`,
`external_agreement`, `aggregate`, and `release_audit`. Calibration and validation use 128
record-level shards; focused inference uses 64. Shards checkpoint every 25 completed records,
carry checksummed compressed records and provenance sidecars, and can be rerun selectively.

Aggregation rejects missing or duplicate shards and records, invalid seeds, mixed commits or
environments, reduced smoke configurations, bad source data, wrong dependency locks, and invalid
checksums. Validation cannot start until the calibration candidate is frozen and checksum-bound.

The external track has two parts:

1. DoubleMLAPOS receives SCOVA-CF's folds and nuisance predictions. Means, centered influence
   rows, covariance, contrasts, and aligned standard errors must agree at the declared numerical
   tolerances. DoubleML's raw standard errors use an `n` denominator; the artifact records this
   and compares covariance after alignment to SCOVA-CF's `n-1` convention.
2. Fifty datasets in each of eight cells compare fitted DoubleML means and EconML DRLearner
   contrasts. Both receive the frozen folds, treatment ordering, learner classes, parameters,
   and preprocessing policy. DoubleML/EconML are pinned validation-only dependencies, never
   runtime SCOVA dependencies.

Install those dependencies with:

```text
python -m pip install -r benchmarks/requirements-cf-validation.txt
python -m pip install -e . --no-deps
```

## Promotion and packaging

Promotion requires every cell-level coverage, type-I error, bias, standard-error calibration,
structural refusal, usefulness, unstable-enrichment, seed-visibility, simultaneous-inference,
external-agreement, deterministic-replay, and checksum gate to pass. Benchmark efficiency is
reported but is not a promotion gate.

If all evidence passes, `scripts/package_cf_support_profile.py` creates a proposed package
manifest. That exact promoted profile must be committed to
`src/scova/cf/data/support_profiles.json`; the separate release audit then confirms byte-level
profile identity and the entire evidence chain. Until that happens, the empty manifest is an
intentional fail-closed release state.

If validation fails, thresholds must not be revised using held-out results. The blocking report
becomes the release artifact, and work moves to a new protocol and new validation namespace.
