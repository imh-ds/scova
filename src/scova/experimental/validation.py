"""Population-level numerical validation helpers for the estimated-tilt EIF."""

from __future__ import annotations

import numpy as np

from .tilts import geometric_tilt_and_gradient


def target_value(
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
    group: int,
    lam: float,
    active_codes: tuple[int, ...],
) -> float:
    tilt, _ = geometric_tilt_and_gradient(propensity, np.array([lam]), active_codes)
    return float(np.sum(tilt[:, 0] * outcome_regression[:, group]) / tilt[:, 0].sum())


def target_directional_derivative(
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
    propensity_direction: np.ndarray,
    group: int,
    lam: float,
    active_codes: tuple[int, ...],
) -> float:
    """Analytic derivative of the target through the propensity channel."""
    tilt, gradient = geometric_tilt_and_gradient(propensity, np.array([lam]), active_codes)
    psi = target_value(propensity, outcome_regression, group, lam, active_codes)
    derivative_tilt = np.einsum("nk,nk->n", gradient[:, 0, :], propensity_direction)
    return float(
        np.mean((outcome_regression[:, group] - psi) * derivative_tilt) / np.mean(tilt[:, 0])
    )


def assignment_eif_inner_product(
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
    propensity_direction: np.ndarray,
    group: int,
    lam: float,
    active_codes: tuple[int, ...],
) -> float:
    """Exact E[EIF * score] for a multinomial assignment submodel."""
    if not np.allclose(propensity_direction.sum(axis=1), 0, atol=1e-12):
        raise ValueError("multinomial propensity directions must sum to zero")
    tilt, gradient = geometric_tilt_and_gradient(propensity, np.array([lam]), active_codes)
    psi = target_value(propensity, outcome_regression, group, lam, active_codes)
    eta = np.mean(tilt[:, 0])
    total = 0.0
    for row in range(len(propensity)):
        for assigned in range(propensity.shape[1]):
            residual = np.zeros(propensity.shape[1])
            residual[assigned] = 1
            residual -= propensity[row]
            q = float(gradient[row, 0] @ residual)
            influence = (outcome_regression[row, group] - psi) * (tilt[row, 0] + q) / eta
            score = propensity_direction[row, assigned] / propensity[row, assigned]
            total += propensity[row, assigned] * influence * score
    return total / len(propensity)


def population_one_step_estimate(
    true_propensity: np.ndarray,
    true_outcome_regression: np.ndarray,
    candidate_propensity: np.ndarray,
    candidate_outcome_regression: np.ndarray,
    group: int,
    lam: float,
    active_codes: tuple[int, ...],
    *,
    corrected: bool,
) -> float:
    """Exact conditional expectation of the candidate one-step estimator."""
    tilt, gradient = geometric_tilt_and_gradient(
        candidate_propensity, np.array([lam]), active_codes
    )
    h = tilt[:, 0]
    plug_in = float(np.sum(h * candidate_outcome_regression[:, group]) / h.sum())
    residual_expectation = (
        true_propensity[:, group]
        / candidate_propensity[:, group]
        * (true_outcome_regression[:, group] - candidate_outcome_regression[:, group])
    )
    augmentation = h * residual_expectation
    if corrected:
        q_expectation = np.einsum(
            "nk,nk->n", gradient[:, 0, :], true_propensity - candidate_propensity
        )
        augmentation += (candidate_outcome_regression[:, group] - plug_in) * q_expectation
    return float(plug_in + augmentation.mean() / h.mean())
