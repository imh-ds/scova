# Prospective observational calibration v2 — completed result

**Status:** completed historical calibration; no candidate profile selected
**Run:** [GitHub Actions 31302416116](https://github.com/imh-ds/scova/actions/runs/31302416116)  
**Commit:** `934f97b789bee1df6cc645e6f4fe546007387fcf`  
**Protocol:** `cf-observational-adaptive-qualification-v2`  
**Protocol checksum:** `317af04c514d8aadfaf47fd116709f4e0eeedfa169cdda291982349ecd408df5`  
**Calibration evidence checksum:** `9b6dbe1f5593857069104e013cce9914fa8d7b7b3ae151349eb15f816c2e4d84`

## What ran

The frozen, adaptive-only observational qualification calibration evaluated 48
simulation cells with 2,000 replications per cell (96,000 records), distributed
over 128 completed Actions shards. The calibration aggregator completed with
zero execution failures. It reserved 1,200 replications per cell for fitting
candidate rules and 800 per cell for the audit.

## Result

The calibration evaluated 480 predeclared candidate support rules. None passed
all frozen gates, so it wrote no candidate support profile:

```json
{
  "all_calibration_gates_passed": false,
  "candidate_count": 480,
  "evaluated_top_candidates": 480,
  "candidate_profile": null,
  "execution_failure_count": 0
}
```

The closest candidate did not clear the preregistered enrichment screen. Its
unsafe-versus-supported bad-result risk ratio was 1.632 (required at least
2.0), and its absolute bad-result-rate difference was 0.0204 (required at
least 0.05). It therefore cannot be treated as a candidate by rounding or by
relaxing the rule after the result is known.

## Consequence

This was a successful measurement with a negative historical qualification
outcome, not a software failure. The v2 program did not produce a support
policy. Observational qualification is now retired, so this result must not be
used to create, promote, or imply a new observational support profile.

The complete compressed evidence and per-shard records are retained as the
`cf-observational-calibration-aggregate` Actions artifact on the linked run.
They may be used for diagnosis and methods-study analysis, but not to create a
post-hoc qualification profile.
