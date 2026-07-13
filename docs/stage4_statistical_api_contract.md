# Stage 4 statistical and API contract

## Status and scope

Stage 4 implements **design-certified multi-group comparability** and an
outcome-blind design/analysis firewall.  It depends on the Stage 3 finite-grid
smooth-overlap-path implementation.  Until the Stage 3 directional validation
artifact is locked and verified, all Stage 4 outputs are experimental and
cannot emit a confirmatory `certified` verdict.

This contract fixes the Stage 4 release boundary.  It covers the
theorem-backed outer design/estimation split only.  Same-sample graph selection
with cross-fitted outcome estimation may be investigated later, but is not a
Stage 4 confirmatory mode.

Stage 4 does not implement partial identification, confounding sensitivity,
TMLE, learned contrasts, ranking, or DR-CRT.

## Statistical object

Let the observed data be \(O_i=(X_i,A_i,Y_i)\), with groups
\(A_i\in\{1,\ldots,K\}\).  The design stage observes only \((X,A)\) on a
design split \(D\); its complementary estimation split \(E\) is used for
outcome estimation and inference.  The split assignment is deterministic from
the declaration seed and is recorded in the result.

For each declared subset \(S\subseteq\{1,\ldots,K\}\), \(|S|\ge2\), and
each declared finite-grid tilt \(\lambda\), Stage 4 uses the Stage 3 subset
overlap target

\[
h_{S,\lambda}(x)=h_0(x)^{1-\lambda}
  \left\{\left(\sum_{k\in S}e_k(x)^{-1}\right)^{-1}\right\}^{\lambda}.
\]

The design stage evaluates the declared reliability gates for this target.  A
subset is *supported* at a grid point only if every required gate passes.  An
edge \((a,b)\) is present when the subset \(\{a,b\}\) is supported at at
least one declared grid point.  A hyperedge is defined identically for a
declared subset of size greater than two.  The result stores the complete set
of supported grid points; it never fills gaps or assumes monotonicity.

The selected confirmatory contrast family is the intersection of the
pre-declared family with graph-supported contrasts.  A pairwise contrast is
supported by its edge.  A contrast involving more than two groups is supported
only by a declared, supported hyperedge containing every nonzero-weight group.
An omnibus claim is permitted only for a supported maximal clique/hyperedge;
the package never converts several disconnected pairwise results into a global
claim.

Conditional on the outcome-free design selection, simultaneous max-t
inference on \(E\) controls FWER over the locked selected family at the
declared level.  This is the Stage 4 operational version of T3.  The coverage
claim is conditional on the realized design split and graph, and relies on the
same smooth-tilt, nuisance-rate, and multiplier-bootstrap conditions as Stage
3 for the estimation split.

## Required design gates

The initial release uses Stage 3's versioned gate metrics at every candidate
subset and lambda:

- per-group effective sample size;
- target effective-sample-size ratio;
- top-one-percent weight concentration;
- influence concentration is an outcome-stage reliability diagnostic and is
  deliberately excluded from design-stage support certification;
- lower propensity quantiles and propensity calibration error;
- maximum standardized weighted covariate imbalance;
- cross-fit stability; and
- finite target normalization and finite variance.

The exact thresholds, their calibration status, and their artifact checksum
come from the supplied Stage 3 threshold object.  A `warning` does not create
a confirmatory edge.  A `refuse` produces no edge and records all failing
metrics.  A calibrated passing threshold artifact is necessary, but not by
itself sufficient, for a confirmatory Stage 4 result.

## Firewall and locking semantics

The firewall is an API and dataflow guarantee, not a cryptographic claim about
callers who already possess a raw dataframe.

1. `prepare_design` accepts covariates and group labels only.  It does not
   accept a dataframe containing an outcome column, an outcome array, or an
   outcome learner.
2. It serializes the complete design declaration, split assignment, graph
   selection, GPS learner metadata, threshold provenance, and supported family
   into a canonical payload.  The SHA-256 hash of this payload is the design
   lock.
3. `analyze_outcomes` accepts a locked design result and an outcome vector for
   precisely its estimation rows.  It verifies the lock, row identifiers, and
   supplied declaration before fitting outcome regressions.
4. Altering any target, active subset, lambda grid, contrast family, gate,
   confidence level, seed, split policy, or learner profile invalidates the
   lock.  The caller must rerun design selection.
5. A post-lock addition or modification to a contrast is allowed only as a
   separately labeled `exploratory-only` analysis.  It cannot change the
   stored confirmatory family or its multiplicity correction.

The legacy `SCOVA.fit(data, declaration)` interface remains the Stage 0--2
convenience interface and is not a Stage 4 firewall path.

## Public API

Names below are the required public contract; internal module organization may
evolve without changing these semantics.

```python
design = SCOVADesign(
    propensity_model=...,  # scikit-learn compatible
    design_fraction=0.50,
)

locked = design.prepare_design(
    covariates=X,
    groups=A,
    declaration=design_declaration,
    row_ids=row_ids,
)

result = design.analyze_outcomes(
    locked,
    outcomes=Y_estimation,
    row_ids=row_ids_estimation,
    outcome_model=...,  # scikit-learn compatible
)
```

`DesignDeclaration` is frozen and includes: base/path target, lambda grid,
declared contrasts and candidate subsets, gate thresholds, FWER confidence
level, seed, design fraction, and learner-profile identifiers.  It has a
canonical `to_dict()` and `declaration_hash`.

`SCOVADesignResult` exposes `design_lock`, `graph`, `supported_family`,
`refusals`, `diagnostics`, `split_assignments`, and `design_report()`.  It is
serializable without pickle.  `ComparabilityGraphResult` exposes nodes, edges,
hyperedges, maximal cliques, supported lambdas, and gate evidence.

`SCOVAGraphResult` (the return from `analyze_outcomes`) exposes point estimates
and graph-conditional simultaneous inference only for `supported_family`.
It retains all requested-but-refused contrasts and their typed reasons in its
report payload.

## Verdicts and reporting

Stage 4 returns one verdict per requested contrast:

- `certified-overlap-only` only when Stage 3 thresholds are calibrated, the
  design lock is valid, a subset is graph-supported, and graph-conditional
  inference completes.  This says nothing about the study-population target.
- `exploratory-only` for any post-lock target or contrast change, or for a
  same-sample experimental selection mode.
- `refused(reason)` when no supporting subset/grid point passes, the lock is
  invalid, or outcome data fail alignment checks.
- existing `descriptive-only` interpretation language continues to apply when
  the analyst has not declared causal assumptions.

The report must show the graph, each selected subset's lambda support, gate
table, selected family, conditional error scope, design lock, and all
unsupported requests.  It must not describe an edge as absent without showing
which gate failed.

## Persistence and compatibility

Stage 4 uses a new, explicitly versioned result schema.  It persists the
declaration and design hashes; split-row identifiers or their digest; threshold
artifact hash; GPS metadata; complete graph/hypergraph; requested and selected
families; refusal reasons; and graph-conditional inference settings.  Loading
an older result must not fabricate a Stage 4 lock or graph.  Older schemas stay
readable under their existing semantics.

`SCOVADesignResult.save()` persists no raw covariates or outcomes.
`SCOVADesignResult.load(..., data=...)` requires the original outcome-free
design data and verifies its declaration hash, row IDs, and data digest before
the artifact can be used for outcome analysis.  Stage 4 remains experimental
in the 0.4 line even when its directional release gate passes.

## Acceptance criteria

Stage 4 is complete only when all of the following are automated or backed by
a versioned benchmark artifact:

1. The design API cannot receive outcomes, and outcome access during design is
   rejected by structural tests.
2. Locks are deterministic, survive serialization, and invalidate for every
   declared design change.
3. Graph construction is invariant to group-label order and agrees with
   directly evaluated Stage 3 pairwise/subset path diagnostics.
4. Pairwise-without-K-way simulations retain valid pairwise edges while
   refusing unsupported global claims.
5. Strong-overlap simulations recover the full declared graph.
6. The frozen null campaign meets its conditional-FWER criterion over the
   selected family, and the coverage campaign meets its simultaneous-coverage
   criterion.
7. Results from post-lock requests cannot enter confirmatory inference.

Any statistical, API, gate, or DGP change after freezing the accompanying
protocol requires a new protocol identifier and new validation seed namespace.

The `stage4-graph-firewall-v3` campaign evaluates FWER, coverage, and graph
selection only among accepted records from cells declared inferential. Its
strong-overlap DGP is balanced and independent of covariates; recovery requires
the exact declared complete pairwise graph in every inferential strong-overlap
cell. Expected rare-group refusals are evaluated separately. The v2 evidence
remains an immutable failed protocol and cannot be rerun or retuned.

The `stage4-graph-firewall-v4` protocol replaces every non-rare 500-row,
eight-group catalog cell with its 2,000-row counterpart before any seed is
used. Its artifacts bind the frozen protocol, rendered catalog, catalog-source
digest, Stage 3 threshold digest, and v4 metric-contract schema. Every
inferential cell reports its accepted-repetition count, so an aggregate names
any cell that fails the minimum-acceptance requirement.
