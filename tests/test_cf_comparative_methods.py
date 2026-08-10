"""Tests for the two-group descriptive comparative methods study."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.cf_comparative_simulation import comparative_cells, simulate_comparative_cell
from benchmarks.cf_comparative_estimators import (
    fit_independent_aipw,
    fit_linear_ancova,
    fit_matching_att,
    fit_scova_cf,
    score_replication,
)


def test_design_has_eight_two_group_cells() -> None:
    cells = comparative_cells()

    assert len(cells) == 8
    assert {cell["n_groups"] for cell in cells} == {2}
    assert {cell["n_covariates"] for cell in cells} == {5}
    assert len({cell["cell_id"] for cell in cells}) == 8


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
