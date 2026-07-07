# Stage 3 stabilization status

Stage 3 remains experimental. Stable promotion is controlled by
`release/stage3_promotion.json` and `scripts/check_stage3_release.py`; version
numbers or manually edited booleans cannot bypass the artifact checks.

Implemented infrastructure:

- theorem appendix and automated numerical theory gates;
- frozen `stage3-directional-v1` protocol with 24 calibration, 48 validation,
  and 24 robustness cells;
- disjoint calibration, validation, and publication seed namespaces;
- shardable campaign execution, checksums, aggregation, and directional criteria;
- scientific-target versus pseudo-target reporting under GPS misspecification;
- production learner profiles, inner-CV convex stacking, and nested calibration;
- versioned per-grid reliability diagnostics with warning/refusal behavior;
- locked-threshold calibration with anti-trivial-refusal requirements;
- pinned JAX matrices and bounded-batch memory benchmarking;
- schema-4 persistence and schema-1 through schema-3 migration support;
- checksummed, artifact-backed release evidence.

Remaining directional release work:

1. run the calibration campaign and lock `stage3-directional-v1` thresholds;
2. run untouched directional validation and robustness campaigns;
3. complete JAX 0.4.38 and 0.10.2 matrices;
4. generate coverage, build, memory, and evidence artifacts;
5. pass the artifact-backed promotion checker.

This initial validation is directional engineering evidence, not publication-ready
validation. The 288-cell primary, 96-cell robustness, 2,000-repetition, and
10,000-case JAX campaigns remain available as a later publication protocol.

Until the directional artifacts exist, provisional thresholds can warn or refuse
but can never produce a stable pass. `certified` verdicts remain unavailable even
after the directional release.
