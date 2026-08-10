# Two-Group Comparative Methods Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare SCOVA-CF with linear ANCOVA, propensity-score matching, independent AIPW, and EconML DRLearner in a reproducible two-group observational simulation study.

**Architecture:** Add a frozen methods-only protocol, a DGP module returning all potential-outcome truth, independent comparator adapters, and a checksum-bound artifact/report pipeline. The comparison never creates qualification, calibration, profile, or promotion evidence.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn 1.6.1, `scova.cf`, optional `econml==0.16.0`, pytest, GitHub Actions.

## Global Constraints

- The initial study has two groups, five baseline covariates, continuous outcomes, and eight observational DGP cells.
- SCOVA-CF, ANCOVA, independent AIPW, and DRLearner estimate the eligible study-population ATE `E[Y(1)-Y(0)]`.
- One-to-one propensity-score matching estimates the matched treated-population ATT. Its retention and results must be a separate panel; it must never be ranked as an ATE estimator.
- Primary factors are outcome surface (linear/interaction), confounding surface (linear/nonlinear), and overlap (adequate/poor): `2 × 2 × 2 = 8` cells.
- Final execution uses 1,000 units and 1,000 replications per cell. GitHub Actions smoke runs use at most 25 replications and are explicitly incomplete evidence.
- Every artifact carries cell IDs, seeds, protocol checksum, dependency-lock checksum, frozen commit, planned/completed replication counts, and Monte-Carlo uncertainty.
- Preserve SCOVA-CF numerical behavior, retired observational qualification status, and the randomized v9 profile.

---

### Task 1: Freeze the design and DGP contract

**Files:**
- Create: `benchmarks/specs/cf_two_group_comparative_methods_v1.json`
- Create: `benchmarks/cf_comparative_simulation.py`
- Create: `tests/test_cf_comparative_methods.py`

**Interfaces:** `comparative_cells() -> tuple[dict[str, object], ...]` and `simulate_comparative_cell(cell: Mapping[str, object], seed: int) -> ComparativeData`, where `ComparativeData` includes observed data, `mu0`, `mu1`, propensity, ATE, ATT, and source metadata.

- [ ] **Step 1: Write the failing tests**

```python
def test_design_has_eight_two_group_cells() -> None:
    cells = comparative_cells()
    assert len(cells) == 8
    assert {cell["n_groups"] for cell in cells} == {2}

def test_returned_truth_matches_potential_outcomes() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=17)
    assert dgp.ate == pytest.approx(np.mean(dgp.mu1 - dgp.mu0))
```

- [ ] **Step 2: Run the tests to confirm RED**

Run: `python -m pytest tests/test_cf_comparative_methods.py -k 'design or truth' -q`

Expected: import failure for `cf_comparative_simulation`.

- [ ] **Step 3: Implement the deterministic eight-cell DGP**

```python
def comparative_cells() -> tuple[dict[str, object], ...]:
    return tuple({"cell_id": f"cmp-v1-{outcome}-{confounding}-{overlap}",
                  "n_groups": 2, "n_covariates": 5, "n": 1000,
                  "outcome_surface": outcome, "confounding_surface": confounding,
                  "overlap": overlap}
                 for outcome in ("linear", "interaction")
                 for confounding in ("linear", "nonlinear")
                 for overlap in ("adequate", "poor"))
```

Generate five Gaussian baseline covariates, treatment propensity from the declared confounding/overlap factors, heterogeneous `mu0`/`mu1`, and observed outcome. Record the formulas and seed in source metadata.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest tests/test_cf_comparative_methods.py -k 'design or truth' -q`

Expected: PASS.

```powershell
git add benchmarks/specs/cf_two_group_comparative_methods_v1.json benchmarks/cf_comparative_simulation.py tests/test_cf_comparative_methods.py
git commit -m "feat: add two-group comparative study protocol"
```

### Task 2: Add independent estimator adapters

**Files:**
- Create: `benchmarks/cf_comparative_estimators.py`
- Modify: `tests/test_cf_comparative_methods.py`

**Interfaces:** `MethodEstimate(name, estimand, estimate, standard_error, status, details)` plus `fit_scova_cf`, `fit_linear_ancova`, `fit_independent_aipw`, `fit_matching_att`, and `fit_econml_drlearner`.

- [ ] **Step 1: Write the failing tests**

```python
def test_standardization_methods_identify_the_ate_target() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=21)
    for fit in (fit_scova_cf, fit_linear_ancova, fit_independent_aipw):
        assert fit(dgp, seed=21).estimand == "ate"

def test_matching_reports_its_att_target_and_retention() -> None:
    result = fit_matching_att(simulate_comparative_cell(comparative_cells()[0], seed=22), seed=22)
    assert result.estimand == "att"
    assert 0 < result.details["treated_retained_fraction"] <= 1
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_cf_comparative_methods.py -k 'standardization_methods or matching_reports' -q`

Expected: missing-adapter import failure.

- [ ] **Step 3: Implement the adapters**

Use SCOVA-CF in observational-causal mode and retain its unqualified status. Implement ANCOVA as treatment, covariates, and treatment-by-covariate interactions standardized over all eligible rows. Implement independent five-fold AIPW using held-out histogram-gradient-boosting propensity and treatment-specific outcome fits; do not call SCOVA-CF’s score assembly. Implement 1:1 no-replacement logistic-propensity matching on logit score with a 0.2-SD caliper and return `limited/no-matches` instead of changing the caliper. Use EconML DRLearner when installed, otherwise return `blocked/missing-econml`.

- [ ] **Step 4: Enforce target separation, verify, and commit**

```python
def test_matching_never_enters_ate_rows() -> None:
    rows = score_replication(simulate_comparative_cell(comparative_cells()[0], seed=23), seed=23)
    assert {row["method"] for row in rows if row["estimand"] == "att"} == {"psm-att"}
```

Run: `python -m pytest tests/test_cf_comparative_methods.py -k 'adapter or matching or ate_rows' -q`

Expected: PASS, with a documented blocked DRLearner status if EconML is absent.

```powershell
git add benchmarks/cf_comparative_estimators.py tests/test_cf_comparative_methods.py
git commit -m "feat: add comparative estimator adapters"
```

### Task 3: Build the methods-only artifact and report

**Files:**
- Create: `benchmarks/cf_comparative_methods.py`
- Create: `scripts/render_cf_comparative_methods_report.py`
- Create: `docs/scova_cf_comparative_methods.md`
- Modify: `tests/test_cf_comparative_methods.py`

**Interfaces:** `run_comparative_study(replications: int, max_cells: int | None = None) -> dict[str, object]` and `comparative_artifact(records, replications) -> dict[str, object]`.

- [ ] **Step 1: Write the failing artifact tests**

```python
def test_artifact_separates_ate_and_att_summaries() -> None:
    artifact = comparative_artifact(records=small_complete_records(), replications=25)
    assert artifact["program_type"] == "methods"
    assert "psm-att" not in artifact["ate_summaries"]
    assert set(artifact["att_summaries"]) == {"psm-att"}

def test_smoke_artifact_is_explicitly_incomplete() -> None:
    artifact = run_comparative_study(replications=5, max_cells=2)
    assert artifact["complete"] is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_cf_comparative_methods.py -k 'artifact or smoke_artifact' -q`

Expected: missing-artifact-function failure.

- [ ] **Step 3: Implement summaries and Markdown rendering**

Calculate bias, RMSE, empirical SD, mean reported SE, SE/SD ratio, 95% coverage, failure rate, and retention. Use percentile bootstrap intervals over replications for mean metrics and Wilson intervals for proportions. Reject input rows that carry profile, calibration, promotion, or qualification fields. Render separate ATE and ATT tables, state the DGP-only scope, and label incomplete runs as incomplete methods evidence.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest tests/test_cf_comparative_methods.py -q`

Expected: PASS.

```powershell
git add benchmarks/cf_comparative_methods.py scripts/render_cf_comparative_methods_report.py docs/scova_cf_comparative_methods.md tests/test_cf_comparative_methods.py
git commit -m "feat: add comparative methods artifact and report"
```

### Task 4: Add only a bounded methods smoke workflow

**Files:**
- Create: `.github/workflows/cf-comparative-methods.yml`
- Modify: `README.md`
- Modify: `docs/scova_cf_comparative_methods.md`

- [ ] **Step 1: Write the failing workflow-bound test**

```python
def test_run_rejects_more_than_frozen_final_replications() -> None:
    with pytest.raises(ValueError, match="1 through 1000"):
        run_comparative_study(replications=1001)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_cf_comparative_methods.py -k frozen_final_replications -q`

Expected: missing validation failure.

- [ ] **Step 3: Implement the bounded manual workflow**

Validate `1 <= replications <= 1000`; the workflow accepts at most 25 and defaults to 5. Use `workflow_dispatch`, install the frozen study stack including `econml==0.16.0`, render JSON and Markdown outputs, and upload them as `cf-comparative-methods-smoke`. State in README that it is descriptive methods evidence only and cannot dispatch any qualification lane.

- [ ] **Step 4: Run final verification and dispatch smoke**

Run: `python -m pytest tests/test_cf_comparative_methods.py -q`

Run: `ruff check src tests examples benchmarks scripts`

Run: `mypy src/scova`

Run: `gh workflow run cf-comparative-methods.yml --ref <branch> -f replications=5`

Expected: all local checks pass; uploaded smoke artifact has `program_type: "methods"`, `complete: false`, and separate ATE/ATT summaries.

- [ ] **Step 5: Commit**

```powershell
git add .github/workflows/cf-comparative-methods.yml README.md docs/scova_cf_comparative_methods.md tests/test_cf_comparative_methods.py
git commit -m "ci: add comparative methods smoke workflow"
```

## Self-review

- All requested comparators are present and the first study remains two-group and eight-cell.
- Matching’s different target is explicit, preventing a misleading common-target ranking.
- The plan reports uncertainty and failures but makes no observational qualification or causal-certification claim.
- The plan changes no SCOVA-CF estimator behavior, support thresholds, result taxonomy, or randomized profile.
