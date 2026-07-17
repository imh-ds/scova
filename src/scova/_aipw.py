"""Private AIPW primitives shared by SCOVA feature layers."""

from __future__ import annotations

import numpy as np


def validate_probability_matrix(probability: np.ndarray, n: int, k: int) -> np.ndarray:
    """Validate and return a coherent, strictly positive probability matrix."""
    values = np.asarray(probability, dtype=float)
    if values.shape != (n, k):
        raise ValueError(f"Propensity predictions must have shape {(n, k)}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Propensity predictions must be finite")
    if np.any(values <= 0) or np.any(values > 1):
        raise ValueError("Propensity predictions must be strictly positive and at most one")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("Each propensity prediction row must sum to one")
    return values


def assemble_aipw(
    outcome: np.ndarray,
    group_codes: np.ndarray,
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-target means, influence rows, and estimator covariance."""
    n, n_groups = propensity.shape
    if outcome.shape != (n,):
        raise ValueError(f"Outcome must have shape {(n,)}")
    if group_codes.shape != (n,):
        raise ValueError(f"Group codes must have shape {(n,)}")
    if outcome_regression.shape != (n, n_groups):
        raise ValueError(f"Outcome predictions must have shape {(n, n_groups)}")
    if not np.all(np.isfinite(outcome)):
        raise ValueError("Outcome must be finite")
    if not np.all(np.isfinite(outcome_regression)):
        raise ValueError("Outcome predictions must be finite")
    if np.any(group_codes < 0) or np.any(group_codes >= n_groups):
        raise ValueError("Group codes must index propensity columns")
    observed = np.eye(n_groups, dtype=float)[group_codes]
    signal = outcome_regression + observed / propensity * (outcome[:, None] - outcome_regression)
    means = signal.mean(axis=0)
    influence = signal - means
    covariance = np.cov(influence, rowvar=False, ddof=1) / n
    covariance = np.atleast_2d(covariance)
    covariance = (covariance + covariance.T) / 2
    return means, influence, covariance
