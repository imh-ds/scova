# Stage 3 directional validation runbook

This runbook produces engineering-directional evidence for `0.3.0`. It does not
produce publication-ready validation.

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

Generate the evidence report with every required artifact argument, then run:

```text
python scripts/check_stage3_release.py
```

Only a zero exit status authorizes version promotion and top-level API exports.

## 4. Publication campaign

The `publication_release` and `publication_robustness` tiers retain the original
large campaign. They are deliberately nonblocking for `0.3.0` and must use the
publication seed namespace. Their results must be described separately from
the directional evidence.
