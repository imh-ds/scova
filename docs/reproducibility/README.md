# Reproducibility and release evidence

These runbooks explain how to reproduce the experimental Stage 3 directional
campaign and how to read its release gate. Run commands from the repository
root and preserve the generated artifacts; local outputs are ignored by Git.

- [`stage3_directional_runbook.md`](stage3_directional_runbook.md) — Actions
  workflow and manual recovery commands.
- [`stage3_stabilization.md`](stage3_stabilization.md) — current blockers and
  promotion semantics.

The decision log records why the campaign is directional and what remains
required before any stable or confirmatory claim.

The retained SCOVA-CF v9 evidence is bound to its frozen source commit. A
release built from a later checkout must rerun or formally rebind that campaign
against the exact release source and pinned dependency stack before making a
current `qualified` claim.
