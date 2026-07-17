"""Mandatory benchmark analyses for SCOVA-CF reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..declaration import JsonLabel


def _summary(means: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    return {
        "means": means.tolist(),
        "standard_errors": np.sqrt(np.maximum(np.diag(covariance), 0)).tolist(),
        "covariance": covariance.tolist(),
    }


def unadjusted_benchmark(
    outcome: np.ndarray,
    group_codes: np.ndarray,
    group_labels: tuple[JsonLabel, ...],
) -> dict[str, Any]:
    """Unadjusted group means with independent-unit heteroskedastic variance."""
    means = np.empty(len(group_labels))
    covariance = np.zeros((len(group_labels), len(group_labels)))
    counts: dict[str, int] = {}
    for code, label in enumerate(group_labels):
        values = outcome[group_codes == code]
        means[code] = float(values.mean())
        covariance[code, code] = float(values.var(ddof=1) / len(values))
        counts[str(label)] = len(values)
    return {
        "name": "unadjusted-group-means",
        "group_labels": list(group_labels),
        "counts": counts,
        **_summary(means, covariance),
    }


def lin_interacted_benchmark(
    outcome: np.ndarray,
    x: np.ndarray,
    group_codes: np.ndarray,
    group_labels: tuple[JsonLabel, ...],
) -> dict[str, Any]:
    """Fully interacted centered linear benchmark with HC1 covariance."""
    n, p = x.shape
    k = len(group_labels)
    centered = x - x.mean(axis=0)
    treatment = np.eye(k, dtype=float)[group_codes][:, 1:]
    interactions = np.column_stack(
        [treatment[:, code, None] * centered for code in range(k - 1)]
    )
    design = np.column_stack((np.ones(n), treatment, centered, interactions))
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1] or n <= design.shape[1]:
        return {
            "name": "lin-fully-interacted",
            "status": "limited/rank-deficient-benchmark",
            "reason": "The fully interacted benchmark design is rank deficient or saturated",
            "group_labels": list(group_labels),
        }
    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residual = outcome - design @ coefficients
    meat = design.T @ (np.square(residual)[:, None] * design)
    coefficient_covariance = bread @ meat @ bread
    coefficient_covariance *= n / (n - design.shape[1])
    mean_design = np.zeros((k, design.shape[1]))
    mean_design[:, 0] = 1.0
    for code in range(1, k):
        mean_design[code, code] = 1.0
    means = mean_design @ coefficients
    covariance = mean_design @ coefficient_covariance @ mean_design.T
    covariance = (covariance + covariance.T) / 2
    return {
        "name": "lin-fully-interacted",
        "status": "complete",
        "group_labels": list(group_labels),
        "covariance_type": "HC1",
        **_summary(means, covariance),
    }
