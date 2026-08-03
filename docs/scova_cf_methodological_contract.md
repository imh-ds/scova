# SCOVA-CF methodological contract

**Version:** 1.0.0  
**Effective date:** 2026-08-03  
**Human owner:** Hohjin (Henry) Im  
**Independent human reviewer:** unassigned — required before this contract governs a freeze

## Status and governance

This is the normative methodological contract for SCOVA-CF. It describes what
the current implementation computes, the assumptions required for each
interpretation, and the boundary between assumption-conditional analysis and a
future evidence-qualified operating regime. It does **not** itself establish
that the observational v11 protocol is valid, promotable, or complete.

The owner and an independent human reviewer must record approval before this
version governs a campaign freeze. A change to the estimand, an identification
assumption, the qualification of a nuisance strategy, or an allowed
interpretation requires a new contract version and a new protocol freeze before
new evidence may support that changed claim.

## Estimand and reported contrasts

For each declared group \(g\), SCOVA-CF estimates the standardized conditional
mean

\[
\psi_g = E_{P_X}\!\left[E(Y \mid G=g, X)\right],
\]

where \(P_X\) is the covariate distribution of the declared eligible study
population. The implemented reference target is therefore the common declared
eligible population; it is not an implicitly trimmed, overlap-weighted, or
externally transported population.

Every reported contrast is a predeclared linear combination
\(\sum_g c_g\psi_g\) of these group-specific standardized means. SCOVA-CF does
not estimate a person's unobserved outcome, an individual treatment effect,
within-person covariance, or an effect for a population outside the declared
eligible study population.

## Interpretation by analysis mode

### Randomized

`randomized` analyses may use causal language when the declared assignment
mechanism is correct, consistency holds, every group has positive assignment
probability in the target population, and the design has been implemented and
declared correctly. Known constant and known stratified assignment mechanisms
are part of that declaration. A causal interpretation does not by itself make
an analysis confirmatory; confirmatory support additionally requires an
eligible promoted support policy.

### Observational-causal

`observational-causal` analyses are assumption-dependent. In addition to
consistency and positivity, their causal interpretation requires conditional
exchangeability given the declared, pre-assignment covariates: within covariate
levels, group assignment must carry no residual information about the relevant
potential outcomes. The software cannot test this condition or rule out
unmeasured confounding.

### Standardized-associational

`standardized-associational` analyses estimate the standardized conditional
mean parameter above, but make no causal-effect claim. The mode does not become
causal because diagnostics, predictive performance, or a simulation campaign
look favorable.

## Assumptions, estimability, and qualification

Identification assumptions describe the scientific and design conditions under
which a mode's estimand has its stated causal interpretation. They include
correctly declared groups, eligibility population, outcome timing, and
pre-assignment covariates; consistency; the mode-specific assignment condition;
and positivity in the target population.

Estimability conditions concern whether the fitted analysis can reliably
estimate that target. They include independent analysis units (unless a future
clustered extension is explicitly used), finite outcome variance and applicable
regularity conditions, adequate overlap, valid fold construction, successful
cross-fitting, and nuisance estimates sufficiently accurate for the AIPW
remainder to be negligible at the intended precision. In the usual AIPW
formulation, this requires the product of outcome-regression and propensity
estimation errors to be sufficiently small. These are not consequences of
declaring a causal mode.

Support, balance, effective-sample-size, calibration, and predictive-loss
diagnostics assess observable warning signs. They neither verify
exchangeability nor prove positivity, structural-model correctness, or the
AIPW nuisance-error condition in an applied dataset. Simulations establish
performance only for their frozen data-generating regimes; they cannot broaden
this contract after the fact.

## Nuisance-model policy

The default `adaptive` strategy is the only observational nuisance strategy
eligible for a future **validated operating regime** label. That label remains
unavailable until an approved, frozen protocol supplies the required evidence.

`linear` and `custom` strategies remain supported interfaces. Their results are
assumption-conditional and are not eligible for that future observational
validated-regime label under this contract. Selecting a linear model does not
establish that the true nuisance functions are linear; providing a custom model
does not establish its adequacy. Predictive loss, balance, propensity
calibration, and support diagnostics cannot upgrade either strategy into a
validated structural model.

## Explicit non-claims

SCOVA-CF does not:

- test for unmeasured confounding or empirically prove exchangeability;
- empirically prove positivity throughout an applied target population;
- estimate individual counterfactual outcomes or individual effects;
- silently trim observations, clip propensities, retarget the population, or
  select a more favorable estimator; or
- extend a validated-operating-regime claim beyond conditions directly covered
  by an approved qualification protocol.

## Documentation review checklist

Review this contract before approving a pull request that changes the SCOVA-CF
estimator, declaration semantics, support-policy scope, or validation protocol.
The review record must answer all of the following:

1. Does the change alter the estimand, target population, identification
   assumptions, nuisance-strategy qualification, or allowed interpretation?
2. If yes, has the contract version been updated and has a new protocol freeze
   been planned before evidence is interpreted under the changed claim?
3. Do `docs/scova_cf.md` and the SCOVA-CF overview in `README.md` still agree
   with this contract?
4. Has the human owner and an independent human reviewer recorded approval?

Until all applicable answers are recorded, the change may be exploratory but
must not be presented as evidence for a governing or promoted SCOVA-CF claim.
