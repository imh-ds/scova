"""Validated built-in smooth tilts and analytic simplex gradients."""

from __future__ import annotations

import numpy as np


def validate_active_groups(n_groups: int, active_codes: tuple[int, ...]) -> tuple[int, ...]:
    active = tuple(active_codes)
    if len(active) < 2:
        raise ValueError("An overlap target requires at least two active groups")
    if len(set(active)) != len(active):
        raise ValueError("active groups cannot contain duplicates")
    if any(code < 0 or code >= n_groups for code in active):
        raise ValueError("active group code is outside the propensity matrix")
    return active


def harmonic_overlap(propensity: np.ndarray, active_codes: tuple[int, ...]) -> np.ndarray:
    """Harmonic-mean overlap tilt for a declared group subset."""
    probability = np.asarray(propensity, dtype=float)
    if probability.ndim != 2 or not np.all(np.isfinite(probability)):
        raise ValueError("propensity must be a finite two-dimensional array")
    if np.any(probability <= 0):
        raise ValueError("propensities must be strictly positive")
    active = validate_active_groups(probability.shape[1], active_codes)
    log_probability = np.log(probability[:, active])
    maximum = np.max(-log_probability, axis=1, keepdims=True)
    log_inverse_sum = maximum[:, 0] + np.log(
        np.exp(-log_probability - maximum).sum(axis=1)
    )
    overlap = np.exp(-log_inverse_sum)
    if not np.all(np.isfinite(overlap)) or np.any(overlap <= 0):
        raise ValueError("overlap tilt underflowed or became non-finite")
    return overlap


def geometric_tilt_and_gradient(
    propensity: np.ndarray,
    lambdas: np.ndarray,
    active_codes: tuple[int, ...],
    *,
    target: str = "kway",
) -> tuple[np.ndarray, np.ndarray]:
    """Return h_lambda and its ambient-simplex analytic gradient."""
    probability = np.asarray(propensity, dtype=float)
    grid = np.asarray(lambdas, dtype=float)
    if grid.ndim != 1 or not np.all(np.isfinite(grid)):
        raise ValueError("lambdas must be a finite one-dimensional array")
    if target == "study":
        return (
            np.ones((len(probability), len(grid))),
            np.zeros((len(probability), len(grid), probability.shape[1])),
        )
    active = validate_active_groups(probability.shape[1], active_codes)
    overlap = harmonic_overlap(probability, active)
    # Multiplying a tilt by a lambda-specific constant does not change Q_h.
    log_scaled_overlap = np.log(len(active)) + np.log(overlap)
    tilt = np.exp(log_scaled_overlap[:, None] * grid[None, :])
    gradient = np.zeros((len(probability), len(grid), probability.shape[1]))
    for code in active:
        gradient[:, :, code] = (
            grid[None, :]
            * tilt
            * overlap[:, None]
            / np.square(probability[:, code, None])
        )
    if not np.all(np.isfinite(tilt)) or not np.all(np.isfinite(gradient)):
        raise ValueError("tilt or gradient became non-finite")
    if np.any(tilt <= 0):
        raise ValueError("tilt values must remain strictly positive")
    return tilt, gradient


def finite_difference_gradient(
    propensity_row: np.ndarray,
    lambdas: np.ndarray,
    active_codes: tuple[int, ...],
    *,
    step: float = 1e-6,
) -> np.ndarray:
    """Ambient central finite-difference gradient for validation only."""
    row = np.asarray(propensity_row, dtype=float)
    result = np.empty((len(lambdas), len(row)))
    for code in range(len(row)):
        plus = row.copy()
        minus = row.copy()
        plus[code] += step
        minus[code] -= step
        if minus[code] <= 0:
            raise ValueError("finite-difference step crosses the simplex boundary")
        plus_tilt, _ = geometric_tilt_and_gradient(
            plus[None, :], lambdas, active_codes
        )
        minus_tilt, _ = geometric_tilt_and_gradient(
            minus[None, :], lambdas, active_codes
        )
        result[:, code] = (plus_tilt[0] - minus_tilt[0]) / (2 * step)
    return result

