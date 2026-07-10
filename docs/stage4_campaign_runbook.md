# Stage 4 graph-firewall campaign runbook

Stage 4 is experimental `0.4`. Its graph-supported result is not a causal
certificate, and no campaign result permits changing Stage 3 thresholds.

Run the `Stage 4 graph-firewall validation` workflow by tier:

1. `engineering_smoke` uses its dedicated smoke namespace and does not require
   Stage 3 artifacts. It is an implementation check only.
2. `calibration` downloads the Stage 3 reaggregated release bundle and verifies
   its calibrated threshold artifact. Its evidence is diagnostic only.
3. `directional_validation` and then `directional_robustness` use the untouched
   validation namespace. Each has 64 canonical shards and a 180-minute limit.
4. `aggregate` receives the two completed run IDs and writes the evidence
   bundle. If a selected shard was rerun, supply its recovery-run ID as well.

The aggregate job rejects missing, duplicate, mixed-protocol, mixed-threshold,
or invalid-checksum shard records. Promotion requires every frozen directional
criterion and zero accepted post-lock mutations. A failure starts a new
protocol; it does not authorize changing this one.
