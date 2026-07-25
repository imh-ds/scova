# SCOVA-CF v9 evidence archive

Permanent copy of the frozen evidence behind the promoted SCOVA-CF reference
support profile `cf-randomized-continuous-aipw-unnormalized-v9-promoted`.

Unlike `stage4-v4/`, this archive **is** tracked in Git.  The v9 evidence
existed only as a GitHub Actions artifact due to expire on 2026-10-20, and it
is the reporting input for the planned SCOVA technical article, so it is
version-controlled rather than kept on one machine.

## What is here

`release-evidence/` is the complete `cf-release-evidence` bundle from aggregate
run [29963775591](https://github.com/imh-ds/scova/actions/runs/29963775591),
downloaded and re-verified locally.  It is self-contained — every artifact the
release gate requires is inside it, including the external-agreement and
simultaneous-inference evidence reused from the frozen v5/v6 runs.

Per-record campaign data (the material an analysis would re-open):

- `cf-reference-validation-campaign.json.gz` — the held-out draw.  Carries
  SCOVA-CF alongside the `unadjusted` (≈ one-way ANOVA) and `lin_interacted`
  (≈ robust ANCOVA) comparators on identical paired datasets.
- `cf-reference-calibration-campaign.json.gz` — the development draw.

Adjudication and identity:

- `cf-reference-calibration.json`, `cf-reference-validation.json` — gate audits.
- `cf-reference-inference.json`, `cf-reference-external-agreement.json`.
- `cf-reference-candidate-profile.json`, `cf-reference-support-profile.json`,
  `proposed-support-profiles.json`.
- `cf-reference-validation-report.md` — carries every evidence checksum.

`provenance.json` records the frozen spec hash, the freeze tag and commit, both
CF numerical fingerprints at that commit, the source run IDs, and a SHA-256 for
every retained file.

## Verifying

```bash
python scripts/check_cf_reference_release.py \
  --spec benchmarks/specs/cf_reference_v9.json \
  --evidence-root validation_archive/cf-v9/release-evidence \
  --packaged-manifest src/scova/cf/data/support_profiles.json
```

This passes against the copy in this directory: the bundle is internally
consistent and binds to the profile shipped in
`src/scova/cf/data/support_profiles.json`.

## Scope of the claim

The promoted profile covers the **randomized** regime only — its compatibility
lock is `mode: randomized`, `assignment: known-constant`, continuous outcome,
unnormalized AIPW, 2–3 groups.  Nothing here validates the observational path.

## Fingerprint note

The fingerprints in `provenance.json` are those of the frozen commit
`d094a0f`.  Current `main` differs, because promoting the profile rewrote
`src/scova/cf/data/support_profiles.json`, which is itself inside the
fingerprint.  A future campaign therefore needs fresh external/inference
evidence regardless of any estimator change — that cost is already sunk.
