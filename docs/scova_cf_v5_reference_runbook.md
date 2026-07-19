# SCOVA-CF v5 amended-profile promotion runbook

This runbook executes protocol `cf-randomized-continuous-aipw-unnormalized-v5`. It may
promote only randomized, independent-unit, continuous-outcome analyses with
known constant assignment probabilities, unnormalized cross-fitted AIPW, two or three
groups, and at least 50 observed units in every group.

## Immutable baseline

1. Commit all campaign and workflow code on one clean commit.
2. Create the annotated tag `scova-cf-reference-v5-freeze-r1` at that commit and
   push both the commit and tag.
3. Dispatch `freeze_check` using the freeze tag. This inexpensive preflight must
   verify the tag, pinned environment, repository imports, dependency lock, protocol
   checksum, dataset identities, and deterministic freeze manifest before any shard runs.
4. Dispatch every tier through `simultaneous_inference` using the freeze tag as
   the workflow ref. The workflow rejects a ref that is not exactly tagged.

V5 is a documented scope amendment, not a rerun of calibration. Its only admissible
calibration input is the checksum-pinned v4 campaign artifact. The v4 result showed that
the support diagnostics cannot distinguish the failing 30-per-arm two-group interaction
and five-group heteroskedastic regimes. V5 therefore excludes those regimes from the
promotable profile rather than weakening coverage or type-I-error gates.

The workflow deliberately exposes no `calibration` tier. Run `calibrate_support` with
the archived v4 calibration run ID. It verifies the source protocol, evidence checksum,
and source commit before creating a v5 candidate. A new validation namespace is required;
v4 calibration evidence cannot be used as v5 validation evidence.

Any numerical, estimator, adapter, protocol, metric, cell, threshold-family, or
seed change requires a new campaign commit. V4 is a statistical protocol change,
so its calibration and validation evidence must never be combined with v3/r5 records.

## Ordered dispatch

Record each successful GitHub Actions run ID and dispatch in this order:

1. `freeze_check`: verify the immutable v5 tag and the v4 calibration-source lock.
2. `calibrate_support`, passing v4 calibration run ID `29660145080`. This is a fast,
   artifact-only operation; it must be the first v5 evidence-producing tier.
3. `external_agreement`, passing the v5 calibration-support run ID.
4. `simultaneous_inference`, passing the v5 calibration-support and external run IDs.
5. `validation`, passing the v5 calibration-support, external, and inference run IDs.

An optional `pilot` may be run first: 16 shards, all 60 cells, 20 replications, and all five stability
   seeds. Do not proceed unless the strict aggregate, checkpoint replay, zero
   error gate, and projected-runtime margin pass.
6. `aggregate`, passing all four evidence run IDs. A failure produces a blocking
   report and no proposed manifest.

Recovery uses `recovery_run_id` and reruns only absent record indices. Recovery
does not permit a different commit, protocol, environment, seed, candidate, or
shard layout.

## Promotion and release

After every gate passes, dispatch `promotion_patch` with the aggregate run ID.
Review and apply the generated patch in a new release commit. The patch contains
the exact promoted manifest, version `0.5.0`, and the checksum-bound status text.

Dispatch `release_audit` on that release commit with the aggregate run ID. It
reruns Ruff, mypy, the complete test and coverage suites, the evidence audit,
wheel build/inspection, and deterministic artifact checks. It also creates a
GitHub provenance attestation for the wheel and final evidence bundle.

Finally dispatch `release` on the audited commit with the audit run ID. The
repository must provide `SCOVA_RELEASE_GPG_PRIVATE_KEY` and
`SCOVA_RELEASE_GPG_PASSPHRASE` secrets. This tier fails closed without them,
creates signed tag `v0.5.0`, and attaches the attested evidence permanently to
the GitHub release.

If any statistical gate fails, do not apply the promotion patch. Archive the
evidence, publish the blocking report, keep the package manifest empty, and
define a new protocol with untouched validation seeds before another attempt.
