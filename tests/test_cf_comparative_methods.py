"""Tests for the two-group descriptive comparative methods study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.cf_comparative_estimators import (
    fit_econml_drlearner,
    fit_econml_drlearner_conservative,
    fit_independent_aipw,
    fit_linear_ancova,
    fit_matching_att,
    fit_scova_cf,
    score_replication,
)
from benchmarks.cf_comparative_methods import comparative_artifact, run_comparative_study
from benchmarks.cf_comparative_simulation import comparative_cells, simulate_comparative_cell
from scripts.render_cf_comparative_methods_report import render


def test_design_has_eight_two_group_cells() -> None:
    cells = comparative_cells()

    assert len(cells) == 8
    assert {cell["n_groups"] for cell in cells} == {2}
    assert {cell["n_covariates"] for cell in cells} == {5}
    assert len({cell["cell_id"] for cell in cells}) == 8


def test_current_design_is_v2_with_two_distinct_drlearner_recipes() -> None:
    protocol = Path("benchmarks/specs/cf_two_group_comparative_methods_v2.json").read_text(
        encoding="utf-8"
    )

    assert "cf-two-group-comparative-methods-v2" in protocol
    assert "econml-drlearner-conservative" in protocol


def test_new_artifacts_identify_the_v2_protocol() -> None:
    artifact = comparative_artifact(records=[], replications=1)

    assert artifact["protocol_id"] == "cf-two-group-comparative-methods-v2"


def test_returned_truth_matches_potential_outcomes() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=17)

    assert dgp.ate == pytest.approx(np.mean(dgp.mu1 - dgp.mu0))
    assert dgp.att == pytest.approx(np.mean((dgp.mu1 - dgp.mu0)[dgp.group == 1]))
    assert np.all((dgp.propensity > 0.0) & (dgp.propensity < 1.0))


def test_standardization_methods_identify_the_ate_target() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=21)

    for fit in (fit_scova_cf, fit_linear_ancova, fit_independent_aipw):
        assert fit(dgp, seed=21).estimand == "ate"


def test_matching_reports_its_att_target_and_retention() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=22)
    result = fit_matching_att(dgp, seed=22)

    assert result.estimand == "att"
    assert 0 < result.details["treated_retained_fraction"] <= 1


def test_matching_never_enters_ate_rows() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=23)
    rows = score_replication(dgp, seed=23)

    assert {row["method"] for row in rows if row["estimand"] == "att"} == {"psm-att"}


def test_drlearner_records_its_frozen_recipe() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=24)
    result = fit_econml_drlearner(dgp, seed=24)

    assert result.details["recipe"] == {
        "model_propensity": "auto",
        "model_regression": "auto",
        "cv": 2,
        "min_propensity": 1e-6,
    }


def test_conservative_drlearner_is_a_distinct_explicit_comparator() -> None:
    dgp = simulate_comparative_cell(comparative_cells()[0], seed=25)
    result = fit_econml_drlearner_conservative(dgp, seed=25)

    assert result.name == "econml-drlearner-conservative"
    assert result.estimand == "ate"
    assert result.details["recipe"] == {
        "model_propensity": "hist-gradient-boosting-classifier",
        "model_regression": "hist-gradient-boosting-regressor",
        "cv": 5,
        "min_propensity": 0.01,
    }


def test_artifact_separates_ate_and_att_summaries() -> None:
    records = [
        {
            "cell_id": comparative_cells()[0]["cell_id"],
            "method": "linear-ancova",
            "estimand": "ate",
            "estimate": 1.1,
            "standard_error": 0.2,
            "truth": 1.0,
            "status": "ok",
            "details": {},
        },
        {
            "cell_id": comparative_cells()[0]["cell_id"],
            "method": "psm-att",
            "estimand": "att",
            "estimate": 1.2,
            "standard_error": 0.3,
            "truth": 1.0,
            "status": "ok",
            "details": {"treated_retained_fraction": 0.8},
        },
    ]
    artifact = comparative_artifact(records=records, replications=25)

    assert artifact["program_type"] == "methods"
    assert "psm-att" not in artifact["ate_summaries"]
    assert set(artifact["att_summaries"]) == {"psm-att"}


def test_artifact_includes_cell_level_tail_error_summaries() -> None:
    first, second = comparative_cells()[:2]
    records = [
        {
            "cell_id": first["cell_id"],
            "method": "scova-cf",
            "estimand": "ate",
            "estimate": 1.0,
            "standard_error": 0.2,
            "truth": 1.0,
            "status": "ok",
            "details": {},
        },
        {
            "cell_id": second["cell_id"],
            "method": "scova-cf",
            "estimand": "ate",
            "estimate": 3.0,
            "standard_error": 0.2,
            "truth": 1.0,
            "status": "ok",
            "details": {},
        },
    ]
    artifact = comparative_artifact(records=records, replications=1)

    first_summary = artifact["cell_ate_summaries"][first["cell_id"]]["scova-cf"]
    second_summary = artifact["cell_ate_summaries"][second["cell_id"]]["scova-cf"]
    assert first_summary["maximum_absolute_error"] == 0.0
    assert second_summary["maximum_absolute_error"] == 2.0
    assert second_summary["absolute_error_p95"] == 2.0
    assert "Cell-level ATE diagnostics" in render(artifact)


def test_smoke_artifact_is_explicitly_incomplete() -> None:
    artifact = run_comparative_study(replications=1, max_cells=1)

    assert artifact["complete"] is False


def test_full_five_rep_smoke_is_incomplete_against_frozen_final_denominator() -> None:
    records = []
    for cell in comparative_cells():
        for replication in range(5):
            for method, estimand in (
                ("scova-cf", "ate"),
                ("linear-ancova", "ate"),
                ("independent-aipw", "ate"),
                ("econml-drlearner", "ate"),
                ("econml-drlearner-conservative", "ate"),
                ("psm-att", "att"),
            ):
                records.append(
                    {
                        "cell_id": cell["cell_id"],
                        "seed": replication,
                        "method": method,
                        "estimand": estimand,
                        "estimate": 1.0,
                        "standard_error": 0.2,
                        "truth": 1.0,
                        "status": "ok",
                        "details": {},
                    }
                )

    artifact = comparative_artifact(records=records, replications=5)

    assert artifact["completed_records"] == 240
    assert artifact["complete"] is False


def test_numerical_support_warning_remains_in_ate_summary() -> None:
    record = {
        "cell_id": comparative_cells()[0]["cell_id"],
        "method": "scova-cf",
        "estimand": "ate",
        "estimate": 1.1,
        "standard_error": 0.2,
        "truth": 1.0,
        "status": "limited/unstable-support",
        "details": {},
    }
    artifact = comparative_artifact(records=[record], replications=1)

    assert artifact["ate_summaries"]["scova-cf"]["bias"] == pytest.approx(0.1)
    assert artifact["ate_summaries"]["scova-cf"]["failure_rate"] == 0.0


def test_run_rejects_more_than_frozen_final_replications() -> None:
    with pytest.raises(ValueError, match="1 through 1000"):
        run_comparative_study(replications=1001)


def test_methods_workflow_allows_the_descriptive_pilot_limit() -> None:
    workflow = Path(".github/workflows/cf-comparative-methods.yml").read_text(encoding="utf-8")

    assert "1 through 50" in workflow
    assert '"$REPLICATIONS" -gt 50' in workflow
