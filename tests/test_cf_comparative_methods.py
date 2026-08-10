"""Tests for the two-group descriptive comparative methods study."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.cf_comparative_simulation import comparative_cells, simulate_comparative_cell


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
