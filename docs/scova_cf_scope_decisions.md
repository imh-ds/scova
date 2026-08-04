# SCOVA-CF scope-decision log

The authoritative registry is `src/scova/cf/data/scope_decisions.json`. Render
it for review with:

```powershell
python -m scripts.render_cf_scope_decisions `
  --registry src/scova/cf/data/scope_decisions.json `
  --output docs/scova_cf_scope_decision_log.generated.md
```

Each record must identify the evidence and uncertainty, choose exactly one
prospective path, state its effect on existing evidence, and contain separate
owner and independent-reviewer approvals before it is `resolved`. A scope
exclusion is allowed only when its predicate is pre-outcome, runtime-checkable,
and independent of a simulation result.

The registry currently includes unresolved historical blockers. Consequently,
the prospective observational qualification workflow is expected to refuse
dispatch until those records are resolved through recorded review.
