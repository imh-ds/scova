# SCOVA-CF

SCOVA-CF is an opt-in SCOVA feature for flexible, cross-fitted AIPW standardization of group means to one declared eligible population. It is designed for naturally occurring groups when a purely linear ANCOVA model may be too restrictive and matching would discard data. It does not create individual counterfactuals or change the target population by matching, trimming, or clipping.

The formal target, assumptions, and limits are in the [SCOVA-CF estimator contract](scova_cf_methodological_contract.md).

<!-- CF_REFERENCE_PROFILE_STATUS_START -->
The randomized continuous unnormalized-AIPW profile `cf-randomized-continuous-aipw-unnormalized-v9-promoted` remains promoted. Its packaged profile checksum is `cc52d5e0fe3b8470d101e6572bbeafeb2ec6752f4545961f505c3d53351b1991`; it can yield `qualified` output only when explicitly selected and its gates pass.
<!-- CF_REFERENCE_PROFILE_STATUS_END -->

For group `g`, SCOVA-CF estimates \(\psi_g = E_{P_X}[E(Y \mid G=g, X)]\). It reports declared linear contrasts of those common-population means, cross-fitted nuisance diagnostics, influence-based intervals, and unadjusted and fully interacted-linear benchmarks.

## Interpreting results

- `randomized` uses the declared known assignment mechanism. Its causal interpretation still depends on correct design implementation, consistency, and positivity.
- `observational-causal` is an assumption-dependent causal comparison. It requires user-supplied temporal ordering, pre-assignment covariates, consistency, overlap, and conditional exchangeability. Every new result is `unqualified`: the package does not certify observational causal validity.
- `standardized-associational` estimates the same standardized conditional-mean parameter without a causal claim and is `ineligible` for causal qualification.

Diagnostics for overlap, balance, weights, effective sample size, arm size, units per covariate, and stability indicate potential problems. They do not verify exchangeability or prove positivity. Covariate count is displayed as a diagnostic, not an observational eligibility ceiling.

The artifact's `qualification_status` and `qualification_reason` make this interpretation explicit. `qualified` is reserved for the existing matching randomized profile; it means profile coverage, not proof of causal assumptions in an individual dataset.

## Historical record

The former observational qualification program is retired from active GitHub Actions. Its v1/v2 artifacts—including the completed v2 no-candidate result—are historical evidence, not a certification route. See [the historical calibration result](scova_cf_observational_calibration_v2.md).
