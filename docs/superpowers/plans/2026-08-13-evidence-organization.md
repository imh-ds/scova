# Simulation Evidence Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make completed SCOVA and SCOVA-CF simulation evidence easy to locate while retaining non-presentable comparative-study outputs in a clearly labeled historical archive.

**Architecture:** `evidence/presentation/` holds small indexes plus the complete final comparative artifacts and the existing release JSON evidence. `validation_archive/historical/` holds prior pilots and local rehearsal artifacts, separately labeled so they cannot be mistaken for manuscript evidence.

**Tech Stack:** Git-tracked Markdown and JSON simulation artifacts.

## Global Constraints

- Preserve source artifacts byte-for-byte; copying does not replace the existing release or local-artifact paths.
- Do not archive test scratch files or `.claude/`.
- Present only the frozen SCOVA-CF v2/v3 final aggregates; do not pool simulation versions.
- Describe all observational findings as simulation evidence, not applied-data causal validation.

---

### Task 1: Create the presentation evidence index

**Files:**
- Create: `evidence/README.md`
- Create: `evidence/presentation/scova/README.md`
- Create: `evidence/presentation/scova-cf/README.md`

- [ ] Add a top-level index that distinguishes presentation evidence from historical material.
- [ ] Document the SCOVA release evidence and its experimental boundaries.
- [ ] Document the SCOVA-CF v2/v3 final runs, checksums, target estimands, and interval-interpretation restriction.

### Task 2: Preserve completed evidence

**Files:**
- Create: `evidence/presentation/scova/*.json`
- Create: `evidence/presentation/scova-cf/v2-final/*`
- Create: `evidence/presentation/scova-cf/v3-final/*`

- [ ] Copy the six existing regular-SCOVA release artifacts.
- [ ] Copy the aggregated v2 and v3 SCOVA-CF final JSON and Markdown reports from their local Actions downloads.
- [ ] Verify the copied SCOVA-CF artifacts retain their recorded checksums and complete denominators.

### Task 3: Archive non-presentable comparative work

**Files:**
- Create: `validation_archive/historical/README.md`
- Create: `validation_archive/historical/cf-comparative-wip/*`
- Modify: `.gitignore`

- [ ] Copy the v1/v2/v3 pilot and rehearsal outputs into a named historical folder.
- [ ] Exclude pytest scratch outputs and final v2/v3 aggregates from the historical copy.
- [ ] Make the historical archive tracked and label its artifacts as non-presentable.

### Task 4: Verify integrity and scope

**Files:**
- Verify: `evidence/`
- Verify: `validation_archive/historical/`

- [ ] Check the expected file inventory and Git status.
- [ ] Parse each retained final JSON and assert the frozen run metadata, complete status, cell count, and record count.
- [ ] Confirm no `.claude/` files are staged.
