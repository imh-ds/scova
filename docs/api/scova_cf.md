# SCOVA-CF public API

SCOVA-CF is an opt-in estimator for cross-fitted, standardized group means
over one declared eligible population. It is useful when a linear ANCOVA model
may be too restrictive and matching would discard data. It does not create
person-specific counterfactuals, individual treatment effects, or a new target
population by matching, trimming, or clipping.

The [SCOVA-CF methodological contract](../contracts/scova_cf_methodological_contract.md)
defines the estimand, assumptions, diagnostics, and qualification boundary.
This page is the practical API guide.

<!-- CF_REFERENCE_PROFILE_STATUS_START -->
The randomized continuous unnormalized-AIPW profile
`cf-randomized-continuous-aipw-unnormalized-v9-promoted` is retained as the
promoted v9 evidence profile. Its packaged profile checksum is
`cc52d5e0fe3b8470d101e6572bbeafeb2ec6752f4545961f505c3d53351b1991`; it can
yield `qualified` output only when explicitly selected and its gates pass.
Because the v9 evidence predates later numerical changes on the current
checkout, it must be refreshed against the exact release source before a
current package release makes a qualification claim.
<!-- CF_REFERENCE_PROFILE_STATUS_END -->

## Quick start

The declaration is intentionally detailed: it records the scientific question,
eligible population, group definitions, covariate rationale, assignment
mechanism, and prespecified contrasts. The repository contains the same
workflow in [`examples/counterfactual_means.py`](../../examples/counterfactual_means.py).

```python
from scova import ContrastSpec
from scova.cf import (
    AnalysisMode,
    KnownAssignment,
    SCOVACF,
    SCOVACFDeclaration,
    SupportPolicy,
)
from scova.simulate import generate_data

simulation = generate_data("randomized", n=600, seed=42)
declaration = SCOVACFDeclaration(
    outcome="outcome",
    group="group",
    covariates=("x1", "x2", "x3"),
    mode=AnalysisMode.RANDOMIZED,
    scientific_question="What would the population mean be under each group?",
    eligibility="All generated study units",
    target_population="Eligible generated study-unit population",
    group_definitions=(
        ("g0", "randomized group zero"),
        ("g1", "randomized group one"),
        ("g2", "randomized group two"),
    ),
    outcome_time="end of follow-up",
    outcome_units="points",
    covariate_rationales=tuple(
        (name, "baseline prognostic factor") for name in ("x1", "x2", "x3")
    ),
    assignment=KnownAssignment(
        probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))
    ),
    support_policy=SupportPolicy.packaged(
        "cf-randomized-continuous-aipw-unnormalized-v9-promoted"
    ),
    stability_seeds=(1, 2, 3, 4, 5),
    contrasts=(
        ContrastSpec("g0 - g1", (("g0", 1.0), ("g1", -1.0))),
    ),
)

analysis = SCOVACF().analyze(simulation.data, declaration)
if hasattr(analysis, "group_means"):
    print(analysis.status.qualification_status.value)
    print(dict(zip(analysis.group_labels, analysis.group_means, strict=True)))
    print(analysis.contrasts["g0 - g1"].to_dict())
    print(analysis.infer(n_bootstrap=499).to_dict())
else:
    print(analysis.to_dict())
```

`SCOVACF().analyze(data, declaration, nuisance_predictions=None)` accepts a
pandas `DataFrame` containing the declared outcome, group, and covariate
columns. It returns an `SCOVACFResult` when numerical analysis completes or an
`SCOVACFRefusal` with machine-readable details when the declared analysis
cannot be completed. A custom propensity and outcome model must be supplied
together when using a custom nuisance strategy.

The quick start explicitly selects the only packaged promoted profile and
declares five stability refits because those are prerequisites for its
qualification gate. Omitting the packaged policy produces a valid numerical
result, but it remains `unqualified`.

## What the declaration controls

`SCOVACFDeclaration` is immutable and requires:

- outcome, group, and baseline covariate column names;
- one of `randomized`, `observational-causal`, or
  `standardized-associational` analysis modes;
- the scientific question, eligibility rule, target population, outcome time,
  and outcome units;
- operational definitions for at least two groups and a rationale for every
  declared covariate;
- `KnownAssignment` for randomized mode, or `EstimatedAssignment` for either
  nonrandomized mode;
- at least one prespecified `ContrastSpec`; and
- cross-fitting, nuisance, support, interval, missingness, and optional
  sensitivity settings.

Post-treatment covariates must not be used as baseline adjustment variables.
The declaration records the estimand as
`study-population-standardized-means`, uses unnormalized AIPW, and defaults to
Wald plus multiplier-bootstrap inference for continuous outcomes.

## Modes and result status

- `randomized` uses the declared known assignment mechanism. Causal
  interpretation still depends on correct implementation, consistency, and
  positivity.
- `observational-causal` is an assumption-dependent causal comparison. It
  requires temporal ordering, pre-assignment covariates, consistency, overlap,
  and conditional exchangeability. New results are always `unqualified`;
  SCOVA-CF does not certify observational causal validity.
- `standardized-associational` estimates the same standardized conditional
  mean without a causal claim and is `ineligible` for causal qualification.

Read `analysis.status.qualification_status` and
`analysis.status.qualification_reason` (or the equivalent fields in
`to_dict()`) before interpreting an estimate. `qualified` means that the
explicitly selected promoted randomized profile matched the declaration and
passed its gates; it is operating-regime coverage, not proof of causal
assumptions in an individual dataset. A refusal returns no best-effort number.

`SCOVACFResult` exposes group means, declared contrast estimates, diagnostics,
influence-based covariance, benchmarks, an evidence card, and serialization
helpers. Call `infer()` for the declared finite contrast family. Use
`to_dict()`, `save()`, and `load()` for versioned, non-pickle result artifacts.

## Diagnostics and limits

Overlap, balance, weights, effective sample size, arm size, units per
covariate, influence concentration, and stability diagnostics reveal potential
problems. They do not verify exchangeability, prove positivity, or establish
causal validity. SCOVA-CF never silently clips propensity scores, trims rows,
or retargets the population.

The former observational qualification program is retired. Its calibration,
workflow, and comparative-study sources are summarized in the
[methodology decision log](../methodology_decision_log.md) and retained in the
[local archive index](../archive/README.md); they are not a current
qualification route.
