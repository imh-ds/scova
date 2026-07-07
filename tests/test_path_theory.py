import numpy as np
import pytest

from scova.experimental.validation import (
    assignment_eif_inner_product,
    population_one_step_estimate,
    target_directional_derivative,
    target_value,
)


def population_setup():
    propensity = np.array(
        [[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.25, 0.6, 0.15], [0.4, 0.35, 0.25]]
    )
    outcomes = np.array(
        [[-1.0, 0.2, 1.4], [-0.3, 0.6, 1.1], [0.4, -0.2, 1.8], [0.8, 0.3, 0.9]]
    )
    direction = np.array(
        [[0.02, -0.01, -0.01], [-0.01, 0.025, -0.015], [0.015, -0.02, 0.005], [-0.02, 0.01, 0.01]]
    )
    return propensity, outcomes, direction


def test_assignment_submodel_eif_identity() -> None:
    propensity, outcomes, direction = population_setup()
    analytic = target_directional_derivative(
        propensity, outcomes, direction, 1, 0.7, (0, 1, 2)
    )
    inner_product = assignment_eif_inner_product(
        propensity, outcomes, direction, 1, 0.7, (0, 1, 2)
    )
    assert inner_product == pytest.approx(analytic, rel=1e-12, abs=1e-12)
    step = 1e-6
    finite_difference = (
        target_value(propensity + step * direction, outcomes, 1, 0.7, (0, 1, 2))
        - target_value(propensity - step * direction, outcomes, 1, 0.7, (0, 1, 2))
    ) / (2 * step)
    assert finite_difference == pytest.approx(analytic, rel=2e-7, abs=2e-9)


def test_correction_removes_first_order_nuisance_error() -> None:
    propensity, outcomes, direction = population_setup()
    outcome_direction = np.array(
        [[0.2, -0.1, 0.05], [-0.1, 0.2, -0.05], [0.15, -0.2, 0.1], [-0.05, 0.1, -0.15]]
    )
    truth = target_value(propensity, outcomes, 0, 0.8, (0, 1, 2))
    errors = {"corrected": [], "naive": []}
    for step in (0.04, 0.02, 0.01):
        candidate_e = propensity + step * direction
        candidate_m = outcomes + step * outcome_direction
        for corrected, name in ((True, "corrected"), (False, "naive")):
            estimate = population_one_step_estimate(
                propensity,
                outcomes,
                candidate_e,
                candidate_m,
                0,
                0.8,
                (0, 1, 2),
                corrected=corrected,
            )
            errors[name].append(abs(estimate - truth))
    corrected_ratio = errors["corrected"][-1] / errors["corrected"][-2]
    naive_ratio = errors["naive"][-1] / errors["naive"][-2]
    assert corrected_ratio < 0.35
    assert naive_ratio > 0.40

