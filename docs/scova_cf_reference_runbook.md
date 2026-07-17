# SCOVA-CF randomized reference promotion runbook

This runbook executes protocol `scova-cf-randomized-continuous-aipw-v2`. It may
promote only randomized, independent-unit, continuous-outcome analyses with
known constant assignment probabilities and unnormalized cross-fitted AIPW.

## Immutable baseline

1. Commit all campaign and workflow code on one clean commit.
2. Create the annotated tag `scova-cf-reference-v2-freeze` at that commit and
   push both the commit and tag.
3. Dispatch every tier through `simultaneous_inference` using the freeze tag as
   the workflow ref. The workflow rejects a ref that is not exactly tagged.

Any numerical, estimator, adapter, protocol, metric, cell, threshold-family, or
seed change requires a new campaign commit. A statistical change requires a v3
protocol and new validation namespace. Engineering-only corrections require a
new commit and a complete pilot and calibration rerun.

## Ordered dispatch

Record each successful GitHub Actions run ID and dispatch in this order:

1. `pilot`: 16 shards, all 60 cells, 20 replications, and all five stability
   seeds. Do not proceed unless the strict aggregate, checkpoint replay, zero
   error gate, and projected-runtime margin pass.
2. `calibration`: 128 shards and 1,000 replications per cell.
3. `calibrate_support`, passing the calibration run ID. This freezes the only
   admissible candidate checksum.
4. `external_agreement`, passing the candidate run ID.
5. `simultaneous_inference`, passing the candidate and external run IDs.
6. `validation`, passing the candidate, external, and inference run IDs. Do not
   inspect shard results while this tier is executing. The 2,000-replication
   namespace is consumed after its single aggregation, whether it passes or
   fails.
7. `aggregate`, passing all four evidence run IDs. A failure produces a blocking
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
define a v3 protocol with untouched validation seeds before another attempt.
