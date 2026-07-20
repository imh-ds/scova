# SCOVA-CF v6 inference-only amendment runbook

V6 remediates the v5 focused simultaneous-inference design. It does not rerun
calibration or external numerical agreement, and it does not alter the
randomized continuous unnormalized-AIPW estimator.

Its six focused cells are all strong-support randomized designs with two or
three groups and at least 80 nominal units per group. V6 uses the fresh
simultaneous-inference seed namespace beginning at `3900000000`.

## Required upstream evidence

- v5 candidate profile run: `29668464451`
- v5 external-agreement evidence run: `29676748117`

The v6 protocol verifies their protocol, artifact, commit, and numerical
implementation identities before accepting them as sources.

## Dispatch order

1. Dispatch `freeze_check` from tag `scova-cf-reference-v6-freeze-r3`.
2. Dispatch `simultaneous_inference` from the same tag, entering the two
   upstream run IDs above in the candidate-profile and external-evidence fields.
   If all 64 shards finish but only their aggregate refuses a host-platform
   mismatch, dispatch `inference_reaggregate` from the current freeze tag and
   enter the failed shard run in the optional recovery-run field. This reuses
   those exact shards; it never recomputes simulation records.
3. Only after the inference evidence passes, dispatch `validation` from the
   same tag with those two IDs and the successful v6 inference run ID.

Do not use v5 inference evidence for validation. The v5 validation namespace is
untouched and remains the one allowed held-out validation lane for V6.
