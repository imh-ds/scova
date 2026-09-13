# SCOVA-CF v9 current-source revalidation runbook

This runbook describes the evidence gate for making the retained randomized
SCOVA-CF profile current for a release. The historical v9 archive remains the
record of the original promotion; it is not silently relabeled as evidence for
a later checkout.

## Frozen runner

Use the exact Python and numerical stack declared in the
[validation requirements](../../benchmarks/requirements-cf-validation.txt):

- Python `3.12.13`
- NumPy `2.2.6`
- pandas `2.2.3`
- SciPy `1.15.3`
- scikit-learn `1.6.1`
- DoubleML `0.11.3`
- EconML `0.16.0`

The workflow checks package versions, the dependency-lock checksum, and the
scikit-learn dataset source checksums before any campaign starts. Every campaign
shard records the same identities, and aggregation rejects mixed identities.

## Current-source sequence

Create a new freeze tag on the exact release source checkout, then dispatch
`.github/workflows/cf-reference-v9-validation.yml` in this order:

1. `freeze_check` — verify the tag, lock, data sources, and campaign entry points.
2. `calibrate_support` — use the tracked frozen v6 development source in
   `validation_archive/cf-v9/release-evidence/` to produce a new v9 candidate
   under the current implementation.
3. `external_agreement` — rerun all v9 external-comparison cells on the current
   implementation.
4. `simultaneous_inference` — rerun all 64 inference shards on the current
   implementation.
5. `validation_preflight` — execute a one-cell held-out smoke check with the
   new candidate and current external/inference evidence.
6. `validation` — run all 128 held-out shards and aggregate them.
7. `aggregate` — bind calibration, external agreement, inference, and held-out
   validation into a new self-verifying release-evidence bundle.

The dispatch inputs must refer to artifacts from the corresponding current
freeze-tagged run. Do not substitute the historical v5/v6 external or inference
artifacts when the current numerical implementation fingerprint has changed.

## Promotion gate

Inspect the aggregate report and run:

```bash
python scripts/check_cf_reference_release.py \
  --spec benchmarks/specs/cf_reference_v9.json \
  --evidence-root validation_archive/cf-v9/release-evidence \
  --packaged-manifest src/scova/cf/data/support_profiles.json
```

Only a passing current-source bundle should be used for `promotion_patch` and
`release_audit`. A failed or incomplete campaign remains evidence that the
current checkout is not yet authorized to make a `qualified` claim.
