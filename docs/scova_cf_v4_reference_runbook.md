# SCOVA-CF v4 randomized reference promotion runbook

This runbook executes protocol `cf-randomized-continuous-aipw-unnormalized-v4`. It may
promote only randomized, independent-unit, continuous-outcome analyses with
known constant assignment probabilities and unnormalized cross-fitted AIPW.

## Immutable baseline

1. Commit all campaign and workflow code on one clean commit.
2. Create the annotated tag `scova-cf-reference-v4-freeze-r1` at that commit and
   push both the commit and tag.
3. Dispatch `freeze_check` using the freeze tag. This inexpensive preflight must
   verify the tag, pinned environment, repository imports, dependency lock, protocol
   checksum, dataset identities, and deterministic freeze manifest before any shard runs.
4. Dispatch every tier through `simultaneous_inference` using the freeze tag as
   the workflow ref. The workflow rejects a ref that is not exactly tagged.

The v3/r5 calibration campaign is archived as a protocol-design result: no candidate
support profile passed its internal calibration gates. It must not be reused as v4
evidence. V4 uses disjoint seed namespaces and a separate immutable freeze tag.

Any numerical, estimator, adapter, protocol, metric, cell, threshold-family, or
seed change requires a new campaign commit. V4 is a statistical protocol change,
so its calibration and validation evidence must never be combined with v3/r5 records.

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
define a new protocol with untouched validation seeds before another attempt.
