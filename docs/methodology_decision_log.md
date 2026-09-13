# SCOVA methodology decision log

This document consolidates the methodological, evidence, and release-boundary
decisions recorded across `docs/` and the repository history. It is a
navigation and decision record, not a replacement for the normative contracts,
theory appendices, runbooks, or machine-readable manifests linked below.

The historical decision records were consolidated from the repository through
commit `761c577` (2026-08-13). A short commit reference identifies the history
entry where the decision was made or materially revised. The current public
path layout is indexed by [`docs/README.md`](README.md); detailed numerical
values, schemas, and execution instructions remain authoritative in the linked
source documents and artifacts.

## Current operating position

| Area | Current decision | Status | Primary authority |
| --- | --- | --- | --- |
| Core SCOVA target | Estimate standardized group means over the declared eligible study population; retain declared contrasts and fixed finite-family simultaneous inference. | Active | [`contracts/statistical_contract.md`](contracts/statistical_contract.md) |
| Stage 3 overlap path | Use an estimated-target, finite-grid smooth-overlap path with its own influence-function correction and reliability gates. | Experimental; no stable or certified verdict | [`reproducibility/stage3_stabilization.md`](reproducibility/stage3_stabilization.md), [`theory/stage3_appendix.md`](theory/stage3_appendix.md) |
| Stage 4 comparability | Select supported pairwise/subset graph structure from outcome-free design data, lock it, then analyze outcomes only on the complementary split. | Experimental; graph support is not a causal certificate | [`api/stage4_statistical_api_contract.md`](api/stage4_statistical_api_contract.md) |
| Stage 5B bounds | Report anchored endpoint experiments only under locked graph geometry and declared bounds; do not issue a Gamma certificate. | Experimental | [`archive/README.md`](archive/README.md) |
| SCOVA-CF randomized profile | The explicitly selected `cf-randomized-continuous-aipw-unnormalized-v9-promoted` profile may produce `qualified` output when its gates pass. | Promoted, regime-limited | [`api/scova_cf.md`](api/scova_cf.md), [`../src/scova/cf/data/support_profiles.json`](../src/scova/cf/data/support_profiles.json) |
| SCOVA-CF observational use | Observational-causal results remain assumption-dependent and `unqualified`; the observational qualification program is retired. | Active product boundary; qualification route retired | [`contracts/scova_cf_methodological_contract.md`](contracts/scova_cf_methodological_contract.md), [`archive/README.md`](archive/README.md) |
| Comparative methods evidence | v2 and v3 are completed descriptive simulation studies with separate evidence sets. They do not create a support profile or causal certification. | Completed methods evidence | [`../evidence/presentation/scova-cf/README.md`](../evidence/presentation/scova-cf/README.md) |

## Status vocabulary

- **Active** — the current contract or product boundary.
- **Promoted** — backed by a packaged profile or release artifact, subject to
  explicit compatibility and gate checks.
- **Experimental** — implemented or evidenced, but not eligible for the stable
  or confirmatory claim named in the source.
- **Historical** — retained for audit or learning; it cannot authorize a new
  claim.
- **Superseded** — replaced by a later protocol or contract. The record remains
  useful for understanding why the replacement exists.

## Chronological decisions

### D-01 — Fix the common study target and finite-family inference

**Date:** 2026-07-06
**Status:** Active

**Decision.** The initial SCOVA estimand is the standardized conditional mean
for each group over the observed, declared eligible study population. Reported
contrasts are declared linear combinations of those means. Simultaneous
inference is defined over the complete finite contrast family using influence
values and a centered Gaussian multiplier maximum statistic; it does not repair
weak support, nuisance failure, or unmeasured confounding.

**Why.** This creates a fixed, population-aware comparison target while keeping
interpretation separate from computation. The contract explicitly rejects
implicit trimming, clipping, retargeting, or hidden extrapolation.

**Consequence.** Causal language remains conditional on analyst-supplied
consistency, exchangeability, positivity, and design assumptions. The package
reports computational inference status separately from causal interpretation.

**Sources:** [`contracts/statistical_contract.md`](contracts/statistical_contract.md); commits
`4ebba6e`, `5f16e31`.

### D-02 — Make the estimated-target overlap path finite-grid and experimental

**Date:** 2026-07-06 to 2026-07-07
**Status:** Experimental

**Decision.** Stage 3 may move from the fixed study target toward a smooth
overlap target on a declared finite grid. Because the target depends on the
estimated propensity, its influence function includes an explicit assignment
channel and target-drift correction. Joint bands, sign sets, and drift bounds
are valid only for the predeclared finite grid and contrast family.

**Why.** The target itself changes with the propensity law, so treating the
propensity as fixed would omit a first-order contribution. Numerical identity,
orthogonality, simplex-invariance, endpoint-reduction, and held-out simulation
gates were made prerequisites for any stable promotion.

**Consequence.** Stage 3 remains under `scova.experimental`. Provisional
thresholds may warn or refuse, but cannot emit a stable pass; a directional
engineering campaign is not publication-ready validation. No continuum-wide
confidence-band, adaptive-grid, or causal-certification claim is allowed.

**Sources:** [`theory/stage3_appendix.md`](theory/stage3_appendix.md),
[`theory/stage3_notation_map.md`](theory/stage3_notation_map.md),
[`theory/estimated_tilt_eif.md`](theory/estimated_tilt_eif.md),
[`reproducibility/stage3_directional_runbook.md`](reproducibility/stage3_directional_runbook.md),
[`reproducibility/stage3_stabilization.md`](reproducibility/stage3_stabilization.md); commits `cbcbada`,
`d8ef0bf`, `abb0a67`, `cb0b51c`.

### D-03 — Enforce outcome-free Stage 4 design selection and graph claims

**Date:** 2026-07-09 to 2026-07-13
**Status:** Experimental

**Decision.** Stage 4 separates design selection from outcome analysis. The
design API receives covariates and group labels only, builds supported
pairwise/subset graph structure on a design split, and emits a checksum-bound
lock. Outcomes are analyzed only on the complementary estimation split. A
confirmatory contrast must be supported by the locked graph; disconnected
pairwise results cannot be promoted into a global omnibus claim.

**Why.** Outcome-free selection is the operational firewall needed for the
conditional FWER claim. Explicit graph and hypergraph support prevents the
package from silently filling target-path gaps or inferring a K-way claim from
several unrelated edges.

**Consequence.** Post-lock target or contrast changes are exploratory-only.
Invalid locks, unsupported edges, and alignment failures produce explicit
refusal reasons or errors. Even if directional gates pass, Stage 4 remains
experimental and its graph support is not a causal certificate.

**Sources:** [`api/stage4_statistical_api_contract.md`](api/stage4_statistical_api_contract.md),
the archived `stage4_campaign_runbook.md` (see [`archive/README.md`](archive/README.md)); commits `8bd4822`,
`0297c9f`, `2a16a46`, `87f64b8`, `e5b858a`, `371dac1`, `bba4a1a`.

### D-04 — Keep Stage 5B anchored bounds conditional and experimental

**Date:** 2026-07-13
**Status:** Experimental

**Decision.** Stage 5B B2 may replace unsupported portions of a locked graph
with clipped, smooth reference predictions and report endpoint-pair inference
under declared outcome bounds. Its result is conditional on the locked Stage 4
geometry, reference sample, transport diagnostics, nuisance conditions, and
finite clipping range.

**Why.** The endpoint construction combines multiple influence channels and an
Imbens--Manski critical value, but the Lipschitz assumption and extrapolation
remain scientific assumptions rather than empirically verified facts.

**Consequence.** Passing the promotion-audit campaign establishes reproducible
experimental evidence only. Stage 5B cannot emit `certified`,
`certified-overlap-only`, or a Gamma certificate; B3 and Holder variants are
outside the release boundary.

**Sources:** the archived `stage5b_promotion_audit.md` (see [`archive/README.md`](archive/README.md)); commits
`d237500`, `e79c54d`, `74b4e22`, `5580376`, `28ed7d0`.

### D-05 — Add SCOVA-CF as an opt-in standardized-mean estimator

**Date:** 2026-07-17
**Status:** Active, with regime-specific qualification

**Decision.** SCOVA-CF estimates standardized group means using cross-fitted,
unnormalized multi-group AIPW. It does not estimate person-specific
counterfactual outcomes or individual effects, recreate within-person
covariance, match away rows, silently trim or clip, or change the declared
target population.

**Why.** The feature is intended for naturally occurring groups where flexible
nuisance learning can be useful without discarding observations through
matching, while preserving an explicit estimand and nonclaim boundary.

**Consequence.** Randomized, observational-causal, and
standardized-associational modes are distinct. The mode controls interpretation;
diagnostics do not upgrade an associational result into a causal result.

**Sources:** [`api/scova_cf.md`](api/scova_cf.md),
[`contracts/scova_cf_methodological_contract.md`](contracts/scova_cf_methodological_contract.md);
commit `2b8c9b8`.

### D-06 — Treat protocol identity, source identity, and freeze state as evidence

**Date:** 2026-07-17
**Status:** Active governance; early protocols historical

**Decision.** Reference-profile campaigns must run from an immutable freeze,
bind their protocol, dependency lock, source arrays, code identity, seeds, and
artifacts by checksum, and fail closed on any mismatch. A statistical or
numerical change requires a new protocol and untouched validation namespace.

**Why.** Protocol v2 declared scikit-learn 1.6.1 while its source checksums had
been generated under 1.9.0; the diabetes bytes actually differed. Correcting
the pin or hashes in place would have changed the frozen experiment rather than
repaired it.

**Consequence.** v2 was blocked before accepted pilot or held-out evidence.
Protocol v3 adopted the pinned environment's source identities and new seed
namespaces. Failed tags and evidence remain auditable but are inadmissible for
v3 promotion.

**Sources:** archived `cf_reference_v2_blocking_report.md`,
`cf_reference_validation.md`, and `scova_cf_reference_runbook.md` (see [`archive/README.md`](archive/README.md)); commits
`5ccb189`, `b020267`, `025c822`, `b68f65c`.

### D-07 — Narrow the randomized reference profile rather than weaken gates

**Date:** 2026-07-18 to 2026-07-19
**Status:** Superseded by the final v9 profile; rationale retained

**Decision.** When calibration evidence showed that observable support
diagnostics could not distinguish specific failing regimes, the profile scope
was narrowed to the regimes directly supported by evidence rather than
weakening coverage or type-I-error gates. V5 therefore required two or three
groups and at least 50 observed units per group, and reused only the pinned v4
calibration source with a fresh validation namespace.

**Why.** The v4 calibration result could not separate the failing small-arm
two-group interaction and five-group heteroskedastic settings using the
available diagnostics. A scope restriction was the defensible prospective
response; post-hoc threshold relaxation was not.

**Consequence.** V4 calibration could not simply become v5 validation. V5
focused inference and later amendments had to honor the candidate's declared
scope; evidence outside it was not allowed to authorize promotion.

**Sources:** archived `scova_cf_v4_reference_runbook.md`,
`scova_cf_v5_reference_runbook.md`, and `cf_reference_v5_inference_blocking_report.md` (see [`archive/README.md`](archive/README.md));
commits `2abca27`, `0d226ea`, `858f91a`.

### D-08 — Correct inference-gate semantics without changing the estimator

**Date:** 2026-07-19
**Status:** Incorporated into later reference evidence

**Decision.** A simultaneous-inference gate checks an upper error bound only for
families containing at least one true null. A family with no true null has
structural FWER zero, and conservative control is valid when it remains below
the calibrated upper bound. The completed v6 shards could be reaggregated
under the corrected semantics without recomputation.

**Why.** V5 failed four of six focused cells, including cells outside the
candidate profile. The first v6 aggregation also applied an invalid two-sided
equality check to all families, which could reject a valid conservative
procedure.

**Consequence.** V6 was an inference-only amendment: it reused only explicitly
bound upstream evidence, introduced fresh inference seeds, retained untouched
validation seeds, and made no estimator, target, support-policy, or threshold
change.

**Sources:** archived `cf_reference_v5_inference_blocking_report.md`,
`cf_reference_v6_inference_gate_erratum.md`, and `scova_cf_v6_inference_runbook.md` (see [`archive/README.md`](archive/README.md)); commits
`4f1cb38`, `8410504`.

### D-09 — Use calibration-side robustness and family-wise error control for v9

**Date:** 2026-07-21 to 2026-07-22
**Status:** Promoted as the randomized v9 profile

**Decision.** The final v9 protocol combined two script-level fixes without
changing the estimator's numerical fingerprint:

1. select the calibration candidate by the one-sided lower confidence bound of
   its instability-enrichment risk ratio, rather than by the tightest
   thresholds; and
2. control the held-out coverage/type-I gate's family-wise spurious-failure
   rate with a Sidak correction over the gated cells.

**Why.** The earlier v7/v8 route suffered from winner's-curse selection and an
uncorrected per-cell gate that could fail a well-calibrated campaign more than
half the time across the cell family. Both fixes use only calibration or gate
semantics; held-out validation remains untouched and decisive.

**Consequence.** Held-out v9 evidence passed end to end, and the exact support
profile was packaged and promoted. This does not extend the profile to
observational assignment, unknown regimes, other outcomes, or unsupported
group/sample configurations.

**Sources:** archived `scova_cf_v7_recalibration_runbook.md` and `cf_reference_validation.md` (see [`archive/README.md`](archive/README.md)),
[`api/scova_cf.md`](api/scova_cf.md);
commits `73e6760`, `33492c4`, `d094a0f`, `2cc8a3a`.

### D-10 — Package the v9 profile and preserve its evidence

**Date:** 2026-07-22 to 2026-07-25
**Status:** Active release state

**Decision.** The promoted profile is available only through explicit packaged
profile selection and compatibility checks. The package records its profile,
protocol, calibration-evidence, validation-evidence, and profile checksums.
The complete release evidence is retained in the repository rather than left
only in expiring Actions artifacts.

**Why.** Qualification means coverage of a tested operating regime, not proof
that an individual applied dataset satisfies causal assumptions. Byte-exact
provenance is necessary to keep the packaged profile tied to the evidence that
promoted it.

**Consequence.** A randomized known-constant profile cannot vouch for an
observational analysis. The package remains fail-closed if the selected profile
or its evidence identity does not match.

**Sources:** [`api/scova_cf.md`](api/scova_cf.md),
[`../src/scova/cf/data/support_profiles.json`](../src/scova/cf/data/support_profiles.json),
[`../validation_archive/cf-v9/README.md`](../validation_archive/cf-v9/README.md);
commits `2cc8a3a`, `d0f1946`.

### D-11 — Separate assignment regimes; do not transfer randomized evidence

**Date:** 2026-07-26 to 2026-08-02
**Status:** Historical development boundary

**Decision.** A support profile must declare the assignment regime it governs.
Randomized known-constant evidence cannot qualify an observational-causal
declaration, and observational evidence cannot qualify the randomized profile.
The observational campaign therefore had to generate its own calibration,
external-agreement, simultaneous-inference, and validation evidence rather than
inheriting randomized lanes.

**Why.** Estimated assignment changes the scientific and numerical problem.
Passing randomized evidence for a known propensity would otherwise be
misleading evidence for a confounded assignment regime.

**Consequence.** The v10/v11 observational workflows and their evidence are
historical development records. They do not change the current randomized
profile and cannot be treated as a second qualification route.

**Sources:** archived `cf-reference-v10-validation.yml`,
`cf-reference-v11-validation.yml`, and `scova_cf_methodological_contract_v1.md` (see [`archive/README.md`](archive/README.md));
commits `30c07d8`, `0b71286`, `776b4a8`, `1b87bbd`, `b4894af`.

### D-12 — Make qualification status and methods evidence explicit

**Date:** 2026-08-03
**Status:** Active

**Decision.** Every SCOVA-CF result carries a qualification status and reason.
`qualified` is reserved for a matching promoted operating regime;
`unqualified` denotes a numerical causal-capable result without that coverage;
`ineligible` denotes a noncausal or contract-ineligible result; and
`unavailable` denotes a typed refusal. Qualification simulations and factorial
methods studies are separate program types.

**Why.** A numerical estimate, a useful diagnostic, a simulation result, and a
validated operating regime answer different questions. The prior documentation
repeatedly risked making those categories look interchangeable.

**Consequence.** Methods artifacts cannot enter qualification consumers.
Observational `linear` and `custom` nuisance strategies remain
assumption-conditional, and diagnostics cannot upgrade their structural
assumptions or claim class.

**Sources:** [`contracts/scova_cf_methodological_contract.md`](contracts/scova_cf_methodological_contract.md),
archived `scova_cf_methodological_contract_v1.md` (see [`archive/README.md`](archive/README.md)),
[`api/scova_cf.md`](api/scova_cf.md); commits `11759c5`, `f0c4372`, `8776d2e`,
`ed9c504`.

### D-13 — Govern scope changes through checksum-bound decisions and approval

**Date:** 2026-08-03 to 2026-08-08
**Status:** Active governance

**Decision.** Material qualification-scope failures or contradictions must be
recorded with evidence, competing explanations, uncertainty, one prospective
path, consequences for prior evidence, and named owner approval. The
machine-readable registry at
[`../src/scova/cf/data/scope_decisions.json`](../src/scova/cf/data/scope_decisions.json)
is the authoritative registry; its checksum-bound manifest is required before
qualification dispatch. Independent review is a promotion safeguard, not a
development blocker.

**Why.** A simulation result cannot become a post-hoc exclusion rule merely
because excluding a cell makes a profile pass. Exclusions are permitted only
when their predicates are pre-outcome, runtime-checkable, and independent of
the simulation result.

**Consequence.** A resolved decision records governance but does not prove
exchangeability, positivity, causal validity, or applied-data nuisance
adequacy. Existing evidence retains its original status and cannot be
relabeled after a scope decision.

**Sources:** archived `scova_cf_scope_decisions.md` (see [`archive/README.md`](archive/README.md)),
[`contracts/scova_cf_methodological_contract.md`](contracts/scova_cf_methodological_contract.md),
[`../src/scova/cf/data/scope_decisions.json`](../src/scova/cf/data/scope_decisions.json);
commits `7b6e907`, `c4095b8`, `b2e00d3`.

### D-14 — Record the observational calibration failure as a valid negative result

**Date:** 2026-08-09
**Status:** Historical; qualification route retired

**Decision.** The frozen observational adaptive-only v2 calibration completed
successfully as a measurement but selected no candidate support profile. All
480 candidates were evaluated; none met the preregistered enrichment gates.
The closest candidate's risk ratio and absolute bad-result-rate difference were
below their required thresholds.

**Why.** A negative calibration result is evidence about the frozen program,
not an execution failure and not a license to round, relax, or tune a rule
after seeing the result.

**Consequence.** The calibration artifact remains available for diagnosis and
methods study only. It cannot create, promote, or imply an observational
support profile.

**Sources:** archived `observational_calibration_v2.md` and
`cf-observational-qualification.yml` (see [`archive/README.md`](archive/README.md));
commits `934f97b`, `2efbd0e`.

### D-15 — Retire observational qualification and preserve it as provenance

**Date:** 2026-08-10
**Status:** Active product boundary

**Decision.** The observational qualification program is retired from active
GitHub Actions. Its v1/v2 contracts, workflows, decision records, and
calibration result are moved or retained under historical documentation for
audit, but no new observational result can be labeled software-qualified.

**Why.** The completed prospective calibration produced no candidate, while
the untestable causal assumptions and non-identifiable applied-data boundaries
cannot be repaired by diagnostics or by a more favorable interpretation of
the simulation evidence.

**Consequence.** New observational-causal outputs are `unqualified`; new
standardized-associational outputs are `ineligible` for causal qualification.
Future comparative methods work may describe performance but cannot reopen the
retired qualification route.

**Sources:** [`contracts/scova_cf_methodological_contract.md`](contracts/scova_cf_methodological_contract.md),
[`api/scova_cf.md`](api/scova_cf.md), and archived qualification material (see [`archive/README.md`](archive/README.md));
commit `3611469`.

### D-16 — Freeze comparative methods studies with separate estimand panels

**Date:** 2026-08-10 to 2026-08-11
**Status:** Completed descriptive evidence; no qualification effect

**Decision.** The comparative program studies performance within frozen
two-group, five-covariate simulated DGPs. SCOVA-CF, ANCOVA, independent AIPW,
and DRLearner target the eligible-population ATE. One-to-one propensity-score
matching targets a matched-treated ATT and is reported separately; it must not
be ranked as an ATE estimator. Pilot outputs are explicitly incomplete, and
v1/v2/v3 evidence sets are never pooled.

**Why.** A common table would imply a common estimand where matching does not
have one. Separating methods evidence from qualification evidence prevents a
performance ranking in simulated data from becoming a product-support claim.

**Consequence.** Reports use separate ATE and ATT summaries, retain Monte Carlo
uncertainty, and label smoke/pilot denominators as incomplete. The studies do
not alter the SCOVA-CF estimator, support policy, result taxonomy, or
observational interpretation.

**Sources:** archived `scova_cf_comparative_methods.md` and
`2026-08-10-two-group-comparative-methods-study.md` (see [`archive/README.md`](archive/README.md));
commits `e256c69`, `f97746f`, `ec1b86b`, `db12c50`, `43cf437`, `15aa29d`,
`4771d9e`, `fe978e4`, `8e140d8`, `d55158b`.

### D-17 — Restrict comparative coverage claims to valid interval bases

**Date:** 2026-08-12
**Status:** Active reporting rule

**Decision.** Comparative coverage is reported only for SCOVA-CF, independent
AIPW, and model-based ANCOVA, whose interval constructions target sampling
uncertainty for the declared ATE. PSM pair-difference dispersion and DRLearner
predicted-effect dispersion are retained for reproducibility but excluded from
coverage comparisons.

**Why.** The latter quantities do not provide valid sampling standard errors for
the stated ATE/ATT targets. Rendering them as coverage would create an invalid
inferential ranking without changing the underlying artifact.

**Consequence.** The v2 artifact remains immutable, while the audited renderer
suppresses the invalid coverage comparisons. A corrected DRLearner inferential
implementation requires a new protocol and separate evidence set.

**Sources:** archived `scova_cf_comparative_inference_audit.md` and
`scova_cf_comparative_methods.md` (see [`archive/README.md`](archive/README.md)); commit
`3072c66`.

### D-18 — Freeze v3 as a separate functional-form stress study

**Date:** 2026-08-12 to 2026-08-13
**Status:** Completed descriptive evidence; no qualification effect

**Decision.** V3 keeps the v2 sample size, group count, covariate count, and
replication denominator, but replaces the functional-form factors with
smooth-nonlinear and threshold outcome/assignment surfaces. It retains target
separation and includes coverage only for estimators with valid sampling
uncertainty. V3 records and summaries must not be pooled with v1 or v2.

**Why.** The study is intended to isolate functional-form stress rather than
change sample size or silently broaden the applied claim. Threshold cells are
observable DGP stressors, not applied-data exclusion rules.

**Consequence.** The current evidence index records a complete v3 final
artifact with 1,000 replications per cell. It remains methods evidence only;
no result creates an observational support profile, promoted regime, or causal
validity claim.

**Sources:** archived `2026-08-12-v3-comparative-stress-study-design.md` and
`scova_cf_comparative_methods.md` (see [`archive/README.md`](archive/README.md)),
[`../evidence/presentation/scova-cf/README.md`](../evidence/presentation/scova-cf/README.md);
commits `7b85731`, `e066558`, `84d15b5`, `761c577`.

### D-19 — Separate presentable evidence from historical development material

**Date:** 2026-08-13
**Status:** Active evidence organization

**Decision.** Completed final evidence is indexed under `evidence/presentation/`;
pilots, shard rehearsals, retired qualification work, and other non-presentable
material are retained under historical archives with explicit labels. Source
artifacts are preserved byte-for-byte, and separate study versions are not
pooled.

**Why.** The repository contains valuable development evidence, but its
presence should not make a smoke run look like a final denominator or make
observational simulation results look like applied causal validation.

**Consequence.** The presentation index is the preferred entry point for
completed simulation evidence. Historical archives remain available for audit
without serving as release or manuscript evidence.

**Sources:** [`../evidence/README.md`](../evidence/README.md),
archived `2026-08-13-evidence-organization.md` (see [`archive/README.md`](archive/README.md)),
[`../validation_archive/historical/README.md`](../validation_archive/historical/README.md);
commit `761c577`.

## Reconciliation notes for older documents

The log intentionally records the current state when older documents and later
artifacts use different release language:

1. The archived `cf_reference_validation.md` describes the earlier v3
   empty-manifest state. The current packaged profile and
   [`api/scova_cf.md`](api/scova_cf.md) supersede that release-state paragraph for the
   repository snapshot recorded here; v3 remains historical protocol evidence.
2. Archived `scova_cf_methodological_contract_v1.md` is retained for
   provenance. The normative contract is the current
   [`contracts/scova_cf_methodological_contract.md`](contracts/scova_cf_methodological_contract.md),
   version 2.0, effective 2026-08-10.
3. Archived `scova_cf_comparative_methods.md` contains
   wording written while v3 was still the next study. The current completion
   state and final checksums are recorded in
   [`evidence/presentation/scova-cf/README.md`](../evidence/presentation/scova-cf/README.md).
4. The archived scope-decision Markdown file describes the registry; the registry at
   [`../src/scova/cf/data/scope_decisions.json`](../src/scova/cf/data/scope_decisions.json)
   is the machine-readable authority for resolved prospective scope decisions.

## Non-claims that remain in force

Across all stages and evidence programs, the package does not claim that:

- diagnostics empirically prove exchangeability, positivity, nuisance adequacy,
  or causal validity in an applied dataset;
- simulation performance establishes universal estimator dominance or creates a
  support profile by itself;
- a standardized mean is an individual counterfactual or individual effect;
- a finite-grid, graph, transport, or endpoint experiment is a continuum or
  extrapolation certificate; or
- a pilot, smoke run, failed protocol, or superseded artifact can substitute
  for the frozen final evidence required by its governing protocol.

For new changes, update the relevant normative contract or machine-readable
manifest first, then add a decision record here when the change affects an
estimand, assumption, evidence boundary, protocol, release rule, or
interpretation.
