# Stage 5B promotion audit

## Endpoint derivation

For a locked, graph-supported pair `(a, b)`, B2 keeps the Stage 5A overlap component and replaces the unsupported remainder with clipped, smooth reference predictions. The endpoint influence construction has five channels:

1. the Stage 5A estimated-propensity overlap/tilt influence contribution;
2. the empirical query-distribution contribution under the frozen soft-k geometry;
3. the Gaussian-mixture transport tilt, whose bandwidth is the design-locked soft-min temperature;
4. the cross-fitted augmented residual contribution for each group outcome surface; and
5. the Imbens--Manski endpoint-pair critical value.

Inference is conditional on the locked design geometry and reference sample. It requires finite declared `[L, U]`, positive finite transport tilts, the configured transport ESS gate, cross-fitted outcome and propensity nuisances satisfying the stated product-rate condition, and no empirical mass at a clipping boundary. A failed condition yields a typed experimental refusal or a boundary-blocked endpoint.

## Interpretation

`Gamma` is measured in outcome units per locked shrinkage-Mahalanobis distance. Results are conditional on the Stage 4 graph: only its supported pairwise edges can appear. Conditional means are clipped to the declared `[L, U]` range before aggregation. The reported Imbens--Manski interval is an interval for the B2 identified-set endpoints, not a causal certificate and not evidence that the Lipschitz assumption is true.

Passing the promotion-audit campaign only establishes reproducible experimental evidence. Stage 5B B2 remains `experimental`; it cannot emit `certified`, `certified-overlap-only`, or a Gamma certificate. B3 and Holder variants are not included.

## Artifact compatibility

Lipschitz artifacts at schema v1 load as legacy results with inference unavailable. Schema v2 adds endpoint influence arrays, standard errors, Imbens--Manski intervals, transport diagnostics, and boundary diagnostics. Stage 5A/B1 artifacts are unchanged.
