# V3 Comparative Stress Study Design

**Purpose:** Test whether SCOVA-CF's flexible outcome standardization adds value when
both treatment assignment and outcomes have shapes that an interacting linear ANCOVA
cannot represent directly. This is a methods study, not a qualification or causal
certification program.

## Scope

V3 is a separate, frozen evidence set from v1 and v2. It retains two naturally
occurring groups, five measured baseline covariates, continuous outcomes, 1,000 units
per replication, and 1,000 replications per cell. It has eight cells:

| Outcome surface | Assignment surface | Overlap |
| --- | --- | --- |
| Smooth nonlinear | Smooth nonlinear | Adequate / poor |
| Smooth nonlinear | Threshold | Adequate / poor |
| Threshold | Smooth nonlinear | Adequate / poor |
| Threshold | Threshold | Adequate / poor |

Keeping sample size fixed makes the new evidence answer one question cleanly: how do
the methods behave when functional form, rather than sample size, is the source of
stress? A later study may vary sample size, but it must be a separate protocol.

## Data-generating process

The exact formulas, overlap scales, noise distribution, seed base, and factor grid are
in [cf_two_group_comparative_methods_v3.json](../../../benchmarks/specs/cf_two_group_comparative_methods_v3.json).
Every DGP must generate all potential-outcome truth before drawing group membership;
the reported ATE is the mean individual treatment effect over every simulated unit.

The threshold cells deliberately use observable covariate rules, rather than hidden
labels or post-hoc exclusions. They are stress conditions for estimator performance,
not claims about any applied dataset.

## Comparators and reporting

SCOVA-CF, interacting linear ANCOVA, independent AIPW, and both DRLearner recipes
estimate the simulated study-population ATE. PSM remains an ATT among retained matched
treated units and is reported separately.

Primary comparisons are bias, RMSE, median absolute error, 95th-percentile absolute
error, maximum absolute error, numerical-failure rate, and warning rate. Coverage is
reported only for SCOVA-CF, ANCOVA, and independent AIPW. The two DRLearner variants
and PSM remain point-estimation-only in v3 unless a future separately frozen protocol
adds valid sampling-uncertainty methods for their targets.

## Guardrails

- V3 must not pool records or summaries with v1 or v2.
- The existing SCOVA-CF estimator and its observational, assumption-dependent product
  interpretation remain unchanged.
- A preliminary pilot must prove the new DGP implementation, artifact provenance, and
  reporting separation before the final 1,000-replication run is dispatched.
- No result from v3 can create an observational support profile, a promoted regime, or
  a causal-validity claim.
