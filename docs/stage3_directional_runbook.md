# Stage 3 directional validation runbook

This runbook produces engineering-directional evidence for `0.3.0`. It does not
produce publication-ready validation.

## Recommended GitHub Actions path

Run **Stage 3 directional release validation** manually from the Actions tab.
The workflow performs the complete sequence on GitHub-hosted Linux runners:

1. eight calibration shards;
2. checksum/provenance verification and threshold locking;
3. eight held-out validation and eight robustness shards using that exact lock;
4. pinned JAX, coverage, build, typing, lint, and memory jobs;
5. shard aggregation, evidence generation, and the promotion checker.

The final `stage3-directional-release-bundle` artifact contains the complete
`release/artifacts` directory and the package threshold file. A failed promotion
check remains a failed workflow, but the diagnostic bundle is uploaded whenever
assembly reached the final job.

The workflow never pushes commits or modifies the release branch. After a passing
run, review and copy the bundle into the repository before performing the version
and public-API promotion changes. GitHub artifact retention is 90 days for the
final bundle and locked thresholds, and 30 days for intermediate shards.

The commands below are the manual recovery path and are also useful for auditing
individual workflow phases.

## 1. Calibration

Run all eight calibration shards from the frozen specification:

```text
python benchmarks/stage3_campaign.py --tier calibration --seed-set calibration \
  --shard-index <0-7> --shard-count 8 --output <calibration-shard.json>
```

Lock the least restrictive passing profile and write the identical package copy:

```text
python benchmarks/calibrate_stage3_gates.py <all calibration shards> \
  --output release/artifacts/stage3-directional-thresholds.json \
  --package-output src/scova/experimental/data/stage3_thresholds.json
```

Do not edit either threshold file after calibration.

## 2. Untouched validation

Run all eight shards for `directional_validation` and `directional_robustness`
with `--seed-set validation` and
`--thresholds release/artifacts/stage3-directional-thresholds.json`. Aggregate
each tier independently:

```text
python benchmarks/summarize_stage3.py <validation shards> \
  --output release/artifacts/stage3-directional-validation.json
python benchmarks/summarize_stage3.py <robustness shards> \
  --output release/artifacts/stage3-directional-robustness.json
```

Thresholds must not be changed in response to these results. A statistical or
implementation change starts a new protocol and seed namespace.

## 3. Engineering artifacts

Produce the two 2,000-case JAX matrices using exactly JAX 0.4.38 and 0.10.2,
the memory artifact, branch-aware `coverage.json`, and wheel/sdist build record.
Place them at the paths declared in `release/stage3_promotion.json`.

Every aggregation must first run `scripts/verify_stage3_shards.py` against the
complete shard set. Generate the evidence report with every required artifact
argument, then run:

```text
python scripts/check_stage3_release.py
```

Only a zero exit status authorizes version promotion and top-level API exports.

## 4. Publication campaign

The `publication_release` and `publication_robustness` tiers retain the original
large campaign. They are deliberately nonblocking for `0.3.0` and must use the
publication seed namespace. Their results must be described separately from
the directional evidence.
