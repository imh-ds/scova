# SCOVA-CF comparative WIP inventory

These files are retained for reproducibility and workflow debugging only. They are not a substitute for either frozen final aggregate in [`evidence/presentation/scova-cf/`](../../../evidence/presentation/scova-cf/).

| Directory | Source | Why archived |
| --- | --- | --- |
| `cf-comparative-actions-31424306687/` | Early v1 Actions output | Pilot-scale precursor to the later frozen v2/v3 designs. |
| `cf-comparative-actions-31438171984/` | Early v1 Actions output | Pilot-scale precursor to the later frozen v2/v3 designs. |
| `cf-comparative-actions-31530672769/` | v1 Actions output | Historical protocol; do not pool with v2 or v3. |
| `github-run-31563232342/` | v2, 50 replications/cell pilot | Incomplete relative to the v2 1,000-replication final denominator. |
| `github-run-31622240305/` | v3, 5 replications/cell smoke | Workflow canary only. |
| `github-run-31624747323/` | v3, 50 replications/cell pilot | Incomplete relative to the v3 1,000-replication final denominator. |
| `shard-rehearsal*/` | Local v2 shard rehearsal | Shard/aggregation mechanics, not methods evidence. |
| `v3-shard-rehearsal/` | Local v3 shard rehearsal | Shard/aggregation mechanics, not methods evidence. |

`pytest-full/` remains excluded because it contains transient test fixtures rather than simulation-study evidence.
