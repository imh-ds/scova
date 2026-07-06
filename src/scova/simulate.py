"""Deterministic data-generating processes with oracle nuisance values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Scenario = Literal["randomized", "observational", "nonlinear", "imbalanced", "weak_overlap"]


@dataclass(frozen=True, slots=True)
class SimulationData:
    data: pd.DataFrame
    propensity: np.ndarray
    outcome_regression: np.ndarray
    true_group_means: np.ndarray
    group_labels: tuple[str, ...]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def generate_data(
    scenario: Scenario = "randomized",
    *,
    n: int = 1000,
    n_groups: int = 3,
    seed: int = 0,
) -> SimulationData:
    """Generate observed data and finite-sample oracle target quantities."""
    if n < 2 or n_groups < 2:
        raise ValueError("n and n_groups must both be at least 2")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    labels = tuple(f"g{index}" for index in range(n_groups))
    group_effect = np.linspace(-0.75, 0.75, n_groups)

    if scenario == "randomized":
        propensity = np.full((n, n_groups), 1 / n_groups)
    else:
        slopes = np.linspace(-0.8, 0.8, n_groups)
        scale = 3.0 if scenario == "weak_overlap" else 1.0
        logits = scale * (
            x[:, [0]] * slopes[None, :] + 0.35 * x[:, [1]] * slopes[::-1][None, :]
        )
        if scenario == "imbalanced":
            logits[:, -1] -= 2.0
        propensity = _softmax(logits)

    outcome_regression = np.empty((n, n_groups))
    for code, effect in enumerate(group_effect):
        base = 0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2]
        if scenario == "nonlinear":
            base = np.sin(x[:, 0]) + 0.4 * np.square(x[:, 1]) - 0.25 * x[:, 2]
        outcome_regression[:, code] = base + effect + 0.3 * code * x[:, 0]

    uniforms = rng.random(n)
    group_codes = (uniforms[:, None] > np.cumsum(propensity, axis=1)).sum(axis=1)
    outcome = outcome_regression[np.arange(n), group_codes] + rng.normal(scale=1.0, size=n)
    data = pd.DataFrame(x, columns=["x1", "x2", "x3"])
    data["group"] = [labels[code] for code in group_codes]
    data["outcome"] = outcome
    return SimulationData(
        data=data,
        propensity=propensity,
        outcome_regression=outcome_regression,
        true_group_means=outcome_regression.mean(axis=0),
        group_labels=labels,
    )


def eif_perturbation_check(
    signal: np.ndarray, score: np.ndarray, *, eps: float = 1e-6
) -> tuple[float, float]:
    """Compare a finite-mixture derivative with the empirical EIF inner product."""
    signal = np.asarray(signal, dtype=float)
    score = np.asarray(score, dtype=float)
    if signal.ndim != 1 or score.shape != signal.shape:
        raise ValueError("signal and score must be one-dimensional arrays of equal length")
    centered_score = score - score.mean()
    if np.max(np.abs(eps * centered_score)) >= 1:
        raise ValueError("eps is too large for positive perturbation weights")
    plus = np.average(signal, weights=1 + eps * centered_score)
    minus = np.average(signal, weights=1 - eps * centered_score)
    finite_difference = float((plus - minus) / (2 * eps))
    influence_inner_product = float(np.mean((signal - signal.mean()) * centered_score))
    return finite_difference, influence_inner_product
