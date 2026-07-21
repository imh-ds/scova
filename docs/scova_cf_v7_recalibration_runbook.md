# SCOVA-CF v7 support-profile recalibration runbook

V7 treats the rejected v6 held-out campaign as development evidence and never
reuses it for promotion. It recalibrates the existing six-feature threshold
family, adds an explicit instability-enrichment screen during calibration, and
reserves a fresh held-out validation seed namespace beginning at `4100000000`.

The v7 enrichment contract requires unsupported results to have at least twice
the bad-inference risk and at least `0.05` greater absolute bad-inference risk
than supported results. The v6 development split contains two preregistered
threshold candidates satisfying both requirements while retaining more than
the required 85% of useful supported results. The deterministically selected
candidate retains the maximum 19,200 eligible fit-split results. Final
promotion still depends on one untouched v7 validation campaign.

## Immutable source evidence

- Rejected v6 aggregate/development run: `29792674086`
- v6 simultaneous-inference evidence run: `29720065229`
- v5 external-agreement evidence run: `29676748117`

The protocol binds each reused artifact by protocol checksum, evidence
checksum, and source commit. The v6 development campaign is accepted only with
its original `validation` lane identity; it is not relabeled inside the signed
payload.

## Dispatch order

Run every tier from tag `scova-cf-reference-v7-freeze-r1`.

1. Dispatch `freeze_check` with every run-ID field blank.
2. Dispatch `calibrate_support` with development run `29792674086`. Leave all
   other run-ID fields blank. Record the successful candidate run ID.
3. Dispatch `validation_preflight` with the candidate run ID, inference run
   `29720065229`, and external run `29676748117`.
4. After preflight passes, dispatch `validation` with those same three IDs.
   This is the only expensive new campaign and uses the untouched v7 seeds.
5. Dispatch `aggregate` with the candidate run ID, successful v7 validation
   run ID, inference run `29720065229`, and external run `29676748117`.
6. Run `promotion_patch`, `release_audit`, and `release` only if aggregate
   promotes a profile and uploads a complete release-evidence bundle.

Never substitute run `29792674086` for the v7 validation run. It is a frozen
development source and is no longer held out.
