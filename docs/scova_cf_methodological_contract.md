# SCOVA-CF estimator contract

**Version:** 2.0
**Effective date:** 2026-08-10
**Owner:** package maintainer

SCOVA-CF is a flexible, cross-fitted estimator for standardized group means. This contract describes what its software computes; it does not certify an observational causal conclusion.

## Target and output

For each declared group `g`, SCOVA-CF estimates \(\psi_g = E_{P_X}[E(Y \mid G=g, X)]\), where `P_X` is the declared eligible study population. Every reported contrast is a declared linear combination of these standardized group means. The estimator uses group-specific outcome models, propensity models where assignment is estimated, cross-fitting, and the unnormalized multi-group AIPW correction.

It does not produce person-specific counterfactuals or individual treatment effects. It does not match and discard rows, silently trim or clip weights, or change the target population.

## Analysis modes

| Mode | What the result means |
| --- | --- |
| `randomized` | A causal interpretation relies on the declared assignment mechanism, consistency, positivity, and correct implementation of the design. The existing promoted randomized profile may yield `qualified` output when explicitly selected and all of its support gates pass. |
| `observational-causal` | An assumption-dependent causal standardized comparison. Its interpretation additionally requires conditional exchangeability given the declared, pre-assignment covariates. New observational results are always `unqualified`; software does not certify this assumption. |
| `standardized-associational` | The standardized conditional-mean parameter without a causal claim. It is `ineligible` for causal qualification. |

## Assumptions and diagnostics

The user, not the package, supplies the scientific and identification claims: independent analysis units, temporal ordering, pre-assignment covariates, consistency, the eligible population, and—when using observational-causal mode—conditional exchangeability and overlap.

Finite-sample operation also needs adequate overlap, regularity and variance, successful cross-fitting, and sufficiently accurate nuisance estimation. AIPW's useful robustness properties depend on both outcome and propensity nuisance errors being adequately controlled, including a sufficiently small product of those errors. Predictive loss, balance, propensity calibration, weights, effective sample size, stability, and simulations can reveal warning signs; they cannot prove exchangeability, positivity, nuisance adequacy in an applied dataset, or causal validity.

Covariate count, units per covariate, arm sizes, weights, balance, and influence concentration remain visible diagnostics. They are warnings, not a universal observational scope boundary.

## Qualification status

| Status | Meaning |
| --- | --- |
| `qualified` | Only a matching explicitly selected promoted randomized profile passed its gates. This is operating-regime coverage, not proof of causal assumptions in a dataset. |
| `unqualified` | A numerical causal-capable result whose support or profile is not qualified. Every new observational-causal result has this status and states that observational qualification is retired. |
| `ineligible` | A successful standardized-associational result, which does not make a causal claim. |
| `unavailable` | A refusal; no numerical estimate was produced. |

`confirmatory` remains a backward-compatible alias for `qualified`.

## Historical qualification record

Observational qualification and promotion are retired product objectives. The historical v1/v2 protocols, decision records, workflows, and calibration result remain available for provenance only. The completed v2 calibration selected no candidate profile; see [the historical calibration result](scova_cf_observational_calibration_v2.md) and [archived workflows](historical/workflows/). They must not be used to relabel a new observational result as software-certified.

A future comparative methods program may report performance against ANCOVA, matching, standard AIPW, and TMLE/DR learners; it cannot itself create a universal observational safety certificate.
