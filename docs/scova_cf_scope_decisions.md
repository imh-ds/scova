# SCOVA-CF scope-decision log

The authoritative registry is `src/scova/cf/data/scope_decisions.json`. Render
it for review with:

```powershell
python -m scripts.render_cf_scope_decisions `
  --registry src/scova/cf/data/scope_decisions.json `
  --output docs/scova_cf_scope_decision_log.generated.md
```

Each record must identify the evidence and uncertainty, choose exactly one
prospective path, state its effect on existing evidence, and contain the named
owner's approval before it is `resolved`. Independent review is a later
promotion safeguard; it does not block development or calibration work. A scope
exclusion is allowed only when its predicate is pre-outcome, runtime-checkable,
and independent of a simulation result.

The registry records the decisions governing the current prospective
observational qualification workflow. A future profile cannot be promoted or
presented as `qualified` without independent review.
