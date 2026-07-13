# Stage 4 graph-firewall campaign runbook

Stage 4 is experimental `0.4`. Its graph-supported result is not a causal
certificate, and no campaign result permits changing Stage 3 thresholds.

Run the `Stage 4 graph-firewall validation` workflow by tier. For every new
run, select protocol `v4`; v1, v2, and v3 are retained only to inspect their
immutable evidence. V2 and v3 failed their promotion gates and must never be rerun.

1. `engineering_smoke` uses its dedicated smoke namespace and does not require
   Stage 3 artifacts. Leave every other input at its default and leave `shard`
   blank. Its verifier requires every fixed record to complete, either exercise
   held-out inference or make its declared safe refusal, and reject post-lock
   tampering for inferential records.
2. `calibration` downloads the Stage 3 reaggregated release bundle and verifies
   its calibrated threshold artifact. Its evidence is diagnostic only. Run
   `verify_calibration` with the completed calibration run ID before validation.
3. `directional_validation` and then `directional_robustness` use the untouched
   validation namespace. Each has 64 canonical shards and a 180-minute limit.
4. `aggregate` receives the two completed run IDs and writes the evidence
   bundle. If a selected shard was rerun, supply its recovery-run ID as well.

Do not dispatch calibration until the v4 engineering-smoke workflow is green.
Do not dispatch validation or robustness until the v4 calibration verifier is
green. The workflow defaults to v4 so a normal dispatch cannot accidentally
reuse an earlier protocol or seed namespace.

V4 scores all inferential metrics only over accepted, declared inferential
records. It replaces all non-rare 500-row/eight-group cells with 2,000-row
eight-group cells before freezing. Strong-complete-graph recovery requires exact
recovery of every true pairwise edge in every inferential strong-overlap cell;
expected rare-group refusals are scored separately.

The aggregate job rejects missing, duplicate, mixed-protocol, mixed-threshold,
or invalid-checksum shard records. Promotion requires every frozen directional
criterion and zero accepted post-lock mutations. A failure starts a new
protocol; it does not authorize changing this one.
