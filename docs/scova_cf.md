# SCOVA-CF

<!-- CF_REFERENCE_PROFILE_STATUS_START -->
The randomized continuous unnormalized-AIPW profile `cf-randomized-continuous-aipw-unnormalized-v9-promoted` is promoted. Its packaged profile checksum is `cc52d5e0fe3b8470d101e6572bbeafeb2ec6752f4545961f505c3d53351b1991` and confirmatory support is available only when this profile is explicitly selected.
<!-- CF_REFERENCE_PROFILE_STATUS_END -->

SCOVA-CF is an opt-in feature of the SCOVA package for governed estimation of
population counterfactual means. It builds on SCOVA's nonlinear nuisance
learning and multi-group AIPW machinery without changing base SCOVA's role as
an alternative to ordinary ANOVA/ANCOVA mean-comparison workflows.

The normative definition of this target, the assumptions needed for each
analysis mode, and the limits of diagnostics and validation evidence are in the
[SCOVA-CF methodological contract](scova_cf_methodological_contract.md).

For group (g), the first SCOVA-CF estimand is

\[
\psi_g = E_{P_X}[E(Y\mid G=g,X)].
\]

Every group mean is therefore standardized to the same declared eligible
population. SCOVA-CF does not recover a person's unobserved potential outcome,
identify individual treatment effects, recreate within-person covariance, or
turn model predictions into repeated measurements. No paired-test or
individual-potential-outcome API is provided.

The promoted v9 randomized profile remains distinct from the observational v11
campaign. The latter is not promoted and is not evidence about the randomized
profile. The validation protocol, locked-seed diagnostic, and fail-closed
release process are described in `docs/cf_reference_validation.md`.

## Current reference slice

The current implementation provides:

- a separate `scova.cf` namespace and artifact schema;
- randomized, observational-causal, and standardized-associational modes with
  mode-derived claim labels;
- known constant or design-stratum-specific randomization probabilities;
- cross-fitted propensity estimates for nonrandomized modes;
- the unnormalized multi-group AIPW reference estimator for continuous outcomes;
- deterministic outcome-free design locks and group/stratum-aware folds;
- outcome-blind support, weight, ESS, balance, and calibration diagnostics;
- typed limitations and refusals rather than best-effort causal results;
- guarded omnibus, pointwise Wald, and finite-family max-t inference;
- mandatory unadjusted and fully interacted linear benchmarks; and
- label-preserving, non-pickle replay artifacts.

The default support policy is deliberately provisional. It can emit `unstable`
or `unsupported`, but it cannot emit a confirmatory `supported` result. An exact
promoted packaged policy may do so only when explicitly selected. Hájek AIPW, TMLE,
repeated-cross-fit aggregation, ratio/odds scales, clustered inference,
missing-outcome scores, overlap targets, and individual-effect bounds remain
gated future modules.

## Example

```python
from scova import ContrastSpec
from scova.cf import (
    AnalysisMode,
    KnownAssignment,
    SCOVACF,
    SCOVACFDeclaration,
)

declaration = SCOVACFDeclaration(
    outcome="outcome",
    group="group",
    covariates=("x1", "x2", "x3"),
    mode=AnalysisMode.RANDOMIZED,
    scientific_question="What would the population mean be under each group?",
    eligibility="All eligible study units",
    target_population="The eligible study-unit population",
    group_definitions=(("g0", "group zero"), ("g1", "group one"), ("g2", "group two")),
    outcome_time="end of follow-up",
    outcome_units="points",
    covariate_rationales=(
        ("x1", "baseline prognostic factor"),
        ("x2", "baseline prognostic factor"),
        ("x3", "baseline prognostic factor"),
    ),
    assignment=KnownAssignment(
        probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))
    ),
    contrasts=(
        ContrastSpec("g0 - g1", (("g0", 1.0), ("g1", -1.0))),
    ),
    n_splits=5,
    random_state=42,
)

analysis = SCOVACF().analyze(data, declaration)
if hasattr(analysis, "group_means"):
    print(analysis.evidence_card)
    print(analysis.contrasts["g0 - g1"].to_dict())
else:
    print(analysis.to_dict())
```

## Interpretation rules

- `randomized` results receive the `randomization-supported` claim class, but
  remain nonconfirmatory unless an eligible promoted packaged policy is selected.
- `observational-causal` results are assumption-dependent and require a
  prespecified quantitative unmeasured-confounding sensitivity analysis before
  eventual promotion; the software cannot test conditional exchangeability.
- `standardized-associational` results never use causal-effect language.
- Prediction loss, balance, propensity calibration, and support diagnostics
  cannot upgrade the declared mode, claim class, or nuisance-model assumptions.
- For observational analyses, only the default `adaptive` nuisance strategy is
  eligible for any future validated-operating-regime label. `linear` and
  `custom` strategies remain assumption-conditional.
- SCOVA-CF never silently trims rows, clips weights, changes targets, selects a
  more favorable estimator, or uses a generalized inverse to hide a singular
  omnibus problem.

The expected performance advantage over ANOVA/ANCOVA is limited to validated
regimes such as nonlinear baseline-outcome relationships, heterogeneous effects,
or the need to standardize all group means to one population. It is not an
unconditional dominance claim.
