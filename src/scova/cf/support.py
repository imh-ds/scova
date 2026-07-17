"""Outcome-blind support assessment for SCOVA-CF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..declaration import JsonLabel
from ..diagnostics import _multinomial_calibration
from .declaration import SupportPolicy
from .status import SCOVACFStatus, SupportStatus


@dataclass(frozen=True, slots=True)
class SupportAssessment:
    status: SCOVACFStatus
    diagnostics: dict[str, Any]


def _share_of_largest(values: np.ndarray, fraction: float) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return 1.0
    count = max(1, int(np.ceil(fraction * len(values))))
    return float(np.sort(positive)[-count:].sum() / positive.sum())


def assess_support(
    *,
    x: np.ndarray,
    group_codes: np.ndarray,
    propensity: np.ndarray,
    folds: np.ndarray,
    covariate_names: tuple[str, ...],
    group_labels: tuple[JsonLabel, ...],
    policy: SupportPolicy,
    assignment_source: str,
) -> SupportAssessment:
    """Compute outcome-blind diagnostics and apply the declared support policy."""
    n, n_groups = propensity.shape
    overall_mean = x.mean(axis=0)
    overall_sd = x.std(axis=0, ddof=1)
    safe_sd = np.where(overall_sd > 0, overall_sd, 1.0)
    warnings: list[str] = []
    group_diagnostics: dict[str, Any] = {}
    for code, label in enumerate(group_labels):
        observed = group_codes == code
        count = int(observed.sum())
        weights = observed.astype(float) / propensity[:, code]
        weight_sum = float(weights.sum())
        squared_sum = float(np.square(weights).sum())
        ess = weight_sum**2 / squared_sum if squared_sum > 0 else 0.0
        ess_ratio = ess / count if count > 0 else 0.0
        normalized = weights / weight_sum if weight_sum > 0 else weights
        weighted_mean = (
            np.average(x, axis=0, weights=weights)
            if weight_sum > 0
            else np.full(x.shape[1], np.nan)
        )
        balance = (weighted_mean - overall_mean) / safe_sd
        max_normalized = float(normalized.max()) if normalized.size else 1.0
        top_one = _share_of_largest(weights, 0.01)
        top_five = _share_of_largest(weights, 0.05)
        if count < policy.min_group_count:
            warnings.append(
                f"group {label!r} count {count} is below {policy.min_group_count}"
            )
        if ess_ratio < policy.min_ess_ratio:
            warnings.append(
                f"group {label!r} ESS ratio {ess_ratio:.3f} is below "
                f"{policy.min_ess_ratio:.3f}"
            )
        if max_normalized > policy.max_normalized_weight:
            warnings.append(
                f"group {label!r} maximum normalized weight {max_normalized:.3f} exceeds "
                f"{policy.max_normalized_weight:.3f}"
            )
        if top_one > policy.max_top_one_percent_weight_share:
            warnings.append(
                f"group {label!r} top-one-percent weight share {top_one:.3f} exceeds "
                f"{policy.max_top_one_percent_weight_share:.3f}"
            )
        maximum_balance = float(np.max(np.abs(balance)))
        if maximum_balance > policy.max_weighted_balance_difference:
            warnings.append(
                f"group {label!r} maximum weighted balance difference "
                f"{maximum_balance:.3f} exceeds "
                f"{policy.max_weighted_balance_difference:.3f}"
            )
        group_diagnostics[str(label)] = {
            "count": count,
            "propensity_target_quantiles": {
                "minimum": float(np.min(propensity[:, code])),
                "q01": float(np.quantile(propensity[:, code], 0.01)),
                "q05": float(np.quantile(propensity[:, code], 0.05)),
                "median": float(np.median(propensity[:, code])),
                "maximum": float(np.max(propensity[:, code])),
            },
            "propensity_observed_quantiles": {
                "minimum": float(np.min(propensity[observed, code])),
                "q05": float(np.quantile(propensity[observed, code], 0.05)),
                "median": float(np.median(propensity[observed, code])),
                "maximum": float(np.max(propensity[observed, code])),
            },
            "effective_sample_size": float(ess),
            "effective_sample_size_ratio": float(ess_ratio),
            "maximum_normalized_weight": max_normalized,
            "top_one_percent_weight_share": top_one,
            "top_five_percent_weight_share": top_five,
            "weighted_balance": {
                name: float(value)
                for name, value in zip(covariate_names, balance, strict=True)
            },
            "maximum_absolute_weighted_balance_difference": maximum_balance,
        }
    fold_counts = {
        str(int(fold)): {
            str(label): int(np.sum((folds == fold) & (group_codes == code)))
            for code, label in enumerate(group_labels)
        }
        for fold in sorted(np.unique(folds))
    }
    if not policy.calibrated:
        warnings.append(
            f"support policy {policy.version!r} is provisional and cannot confirm support"
        )
    if warnings:
        status = SCOVACFStatus(
            support=SupportStatus.UNSTABLE,
            code="limited/unstable-support",
            reason="; ".join(warnings),
            confirmatory=False,
        )
    else:
        status = SCOVACFStatus(
            support=SupportStatus.SUPPORTED,
            code="supported",
            reason=f"All calibrated support checks passed under {policy.version}",
            confirmatory=True,
        )
    diagnostics = {
        "outcome_blind": True,
        "assignment_source": assignment_source,
        "policy": policy.to_dict(),
        "sample_size": n,
        "number_of_groups": n_groups,
        "groups": group_diagnostics,
        "fold_group_counts": fold_counts,
        "propensity_calibration": _multinomial_calibration(group_codes, propensity),
        "warnings": warnings,
    }
    return SupportAssessment(status=status, diagnostics=diagnostics)


def influence_concentration(
    influence: np.ndarray, group_labels: tuple[JsonLabel, ...]
) -> dict[str, Any]:
    """Post-estimation influence diagnostics, kept separate from support decisions."""
    squared = np.square(influence)
    result: dict[str, Any] = {}
    for code, label in enumerate(group_labels):
        denominator = float(squared[:, code].sum())
        result[str(label)] = {
            "top_one_percent_variance_share": (
                _share_of_largest(squared[:, code], 0.01) if denominator > 0 else 0.0
            ),
            "top_five_percent_variance_share": (
                _share_of_largest(squared[:, code], 0.05) if denominator > 0 else 0.0
            ),
        }
    return result
