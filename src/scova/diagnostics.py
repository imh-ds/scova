"""Initial diagnostics for fixed-target estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _multinomial_calibration(
    group_codes: np.ndarray, propensity: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    n, n_groups = propensity.shape
    observed = np.eye(n_groups)[group_codes]
    clipped = np.maximum(propensity[np.arange(n), group_codes], np.finfo(float).tiny)
    log_loss = float(-np.mean(np.log(clipped)))
    brier = float(np.mean(np.sum(np.square(propensity - observed), axis=1)))
    classwise: list[float] = []
    edges = np.linspace(0, 1, n_bins + 1)
    for code in range(n_groups):
        error = 0.0
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            in_bin = (propensity[:, code] >= left) & (
                (propensity[:, code] < right) | ((right == 1) & (propensity[:, code] <= right))
            )
            if np.any(in_bin):
                error += float(np.mean(in_bin)) * abs(
                    float(np.mean(observed[in_bin, code]) - np.mean(propensity[in_bin, code]))
                )
        classwise.append(error)
    return {
        "log_loss": log_loss,
        "brier_score": brier,
        "classwise_expected_calibration_error": classwise,
        "worst_class_expected_calibration_error": float(max(classwise)),
    }


def compute_diagnostics(
    x: np.ndarray,
    group_codes: np.ndarray,
    propensity: np.ndarray,
    influence: np.ndarray,
    folds: np.ndarray,
    covariate_names: tuple[str, ...],
    group_labels: tuple[str | int | float | bool, ...],
) -> dict[str, Any]:
    """Compute design and influence diagnostics without changing the estimand."""
    overall_mean = x.mean(axis=0)
    overall_sd = x.std(axis=0, ddof=1)
    safe_sd = np.where(overall_sd > 0, overall_sd, 1.0)
    propensity_ranges: dict[str, list[float]] = {}
    effective_sample_sizes: dict[str, float] = {}
    maximum_balance_difference: dict[str, float] = {}
    balance_by_covariate: dict[str, dict[str, float]] = {}

    for code, label in enumerate(group_labels):
        key = str(label)
        probabilities = propensity[:, code]
        propensity_ranges[key] = [float(probabilities.min()), float(probabilities.max())]
        observed = group_codes == code
        weights = observed.astype(float) / probabilities
        weight_sum = weights.sum()
        ess = weight_sum**2 / np.square(weights).sum()
        effective_sample_sizes[key] = float(ess)
        weighted_mean = np.average(x, axis=0, weights=weights)
        standardized = (weighted_mean - overall_mean) / safe_sd
        balance_by_covariate[key] = {
            name: float(value) for name, value in zip(covariate_names, standardized, strict=True)
        }
        maximum_balance_difference[key] = float(np.max(np.abs(standardized)))

    squared = np.square(influence)
    top_count = max(1, int(np.ceil(0.01 * influence.shape[0])))
    concentration: dict[str, float] = {}
    for code, label in enumerate(group_labels):
        denominator = squared[:, code].sum()
        top_share = np.sort(squared[:, code])[-top_count:].sum()
        concentration[str(label)] = float(top_share / denominator) if denominator > 0 else 0.0

    fold_group_counts: dict[str, dict[str, int]] = {}
    for fold in sorted(np.unique(folds)):
        fold_group_counts[str(int(fold))] = {
            str(label): int(np.sum((folds == fold) & (group_codes == code)))
            for code, label in enumerate(group_labels)
        }

    return {
        "propensity_ranges": propensity_ranges,
        "effective_sample_sizes": effective_sample_sizes,
        "maximum_absolute_weighted_balance_difference": maximum_balance_difference,
        "weighted_balance": balance_by_covariate,
        "top_one_percent_influence_variance_share": concentration,
        "fold_group_counts": fold_group_counts,
        "propensity_quantiles": {
            str(label): {
                "q01": float(np.quantile(propensity[:, code], 0.01)),
                "q05": float(np.quantile(propensity[:, code], 0.05)),
            }
            for code, label in enumerate(group_labels)
        },
        "propensity_calibration": _multinomial_calibration(group_codes, propensity),
    }
