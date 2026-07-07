# Stage 3 stabilization status

Stage 3 remains experimental. Stable promotion is controlled by
`release/stage3_promotion.json` and `scripts/check_stage3_release.py`; version
numbers or code coverage alone cannot bypass the gate.

Implemented infrastructure:

- theorem appendix, notation map, and two-review protocol;
- exact 288-cell primary and 96-cell robustness specifications;
- disjoint calibration and validation seed namespaces;
- shardable campaign execution, checksums, aggregation, and pass criteria;
- scientific-target versus pseudo-target reporting under GPS misspecification;
- production learner profiles, inner-CV convex stacking, and nested calibration wrappers;
- versioned per-grid reliability diagnostics with provisional warn/refuse behavior;
- threshold candidate profiles and held-out calibration tooling;
- randomized JAX matrices for pull-request and nightly validation;
- bounded-batch large-array memory benchmarking without nuisance refitting;
- schema-4 persistence and schema-1 through schema-3 migration support;
- checksummed release-evidence generation.

Outstanding external/release work:

1. two independent theory reviews;
2. calibration campaign and locked threshold artifact;
3. held-out release and robustness campaigns;
4. completed minimum/latest JAX artifacts;
5. signed release dossier and final promotion approval.

Until those exist, provisional thresholds can warn or refuse but can never
produce a stable certification.
