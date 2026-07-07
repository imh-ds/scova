"""Simultaneous inference for finite families of fixed-target contrasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

import numpy as np
from scipy.stats import chi2

from ._version import __version__


class InferenceStatus(str, Enum):
    """Computational status, separate from substantive interpretation."""

    COMPLETE = "complete"
    WARNING = "warning"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class SimultaneousContrastResult:
    name: str
    estimate: float
    standard_error: float
    statistic: float
    simultaneous_confidence_interval: tuple[float, float]
    adjusted_p_value: float
    rejected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "statistic": self.statistic,
            "simultaneous_confidence_interval": list(
                self.simultaneous_confidence_interval
            ),
            "adjusted_p_value": self.adjusted_p_value,
            "rejected": self.rejected,
        }


@dataclass(frozen=True, slots=True)
class GlobalTestResult:
    max_t_statistic: float
    max_t_p_value: float
    wald_statistic: float
    wald_degrees_of_freedom: int
    wald_p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_t_statistic": self.max_t_statistic,
            "max_t_p_value": self.max_t_p_value,
            "wald_statistic": self.wald_statistic,
            "wald_degrees_of_freedom": self.wald_degrees_of_freedom,
            "wald_p_value": self.wald_p_value,
        }


@dataclass(frozen=True, slots=True)
class SimultaneousInferenceResult:
    family: tuple[str, ...]
    contrasts: tuple[SimultaneousContrastResult, ...]
    confidence_level: float
    n_bootstrap: int
    random_state: int
    multiplier: str
    critical_value: float
    global_test: GlobalTestResult
    status: InferenceStatus
    reasons: tuple[str, ...]
    package_version: str
    configuration_key: str

    def contrast(self, name: str) -> SimultaneousContrastResult:
        """Retrieve one simultaneous contrast result by name."""
        for contrast in self.contrasts:
            if contrast.name == name:
                return contrast
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": list(self.family),
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "random_state": self.random_state,
            "multiplier": self.multiplier,
            "critical_value": self.critical_value,
            "global_test": self.global_test.to_dict(),
            "status": self.status.value,
            "reasons": list(self.reasons),
            "package_version": self.package_version,
            "configuration_key": self.configuration_key,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SimultaneousInferenceResult:
        contrasts = tuple(
            SimultaneousContrastResult(
                name=item["name"],
                estimate=float(item["estimate"]),
                standard_error=float(item["standard_error"]),
                statistic=float(item["statistic"]),
                simultaneous_confidence_interval=tuple(
                    item["simultaneous_confidence_interval"]
                ),
                adjusted_p_value=float(item["adjusted_p_value"]),
                rejected=bool(item["rejected"]),
            )
            for item in values["contrasts"]
        )
        global_values = values["global_test"]
        global_test = GlobalTestResult(
            max_t_statistic=float(global_values["max_t_statistic"]),
            max_t_p_value=float(global_values["max_t_p_value"]),
            wald_statistic=float(global_values["wald_statistic"]),
            wald_degrees_of_freedom=int(global_values["wald_degrees_of_freedom"]),
            wald_p_value=float(global_values["wald_p_value"]),
        )
        return cls(
            family=tuple(values["family"]),
            contrasts=contrasts,
            confidence_level=float(values["confidence_level"]),
            n_bootstrap=int(values["n_bootstrap"]),
            random_state=int(values["random_state"]),
            multiplier=values["multiplier"],
            critical_value=float(values["critical_value"]),
            global_test=global_test,
            status=InferenceStatus(values["status"]),
            reasons=tuple(values["reasons"]),
            package_version=values["package_version"],
            configuration_key=values["configuration_key"],
        )


def _configuration_key(
    family: tuple[str, ...], confidence_level: float, n_bootstrap: int, random_state: int
) -> str:
    payload = json.dumps(
        {
            "family": family,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "random_state": random_state,
            "multiplier": "gaussian",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def run_simultaneous_inference(
    *,
    family: tuple[str, ...],
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    influence_values: np.ndarray,
    weights: np.ndarray,
    group_covariance: np.ndarray,
    confidence_level: float,
    n_bootstrap: int,
    random_state: int,
    batch_size: int,
    warning_reasons: tuple[str, ...] = (),
) -> SimultaneousInferenceResult:
    """Run deterministic Gaussian multiplier and rank-aware Wald inference."""
    if not family:
        raise ValueError("Simultaneous inference requires at least one contrast")
    if len(set(family)) != len(family):
        raise ValueError("Simultaneous inference family cannot contain duplicate names")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    arrays = (estimates, standard_errors, influence_values, weights, group_covariance)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Inference inputs must all be finite")
    if np.any(standard_errors <= 0):
        raise ValueError("Every contrast must have a strictly positive standard error")
    n, n_contrasts = influence_values.shape
    if estimates.shape != (n_contrasts,) or standard_errors.shape != (n_contrasts,):
        raise ValueError("Contrast estimates, errors, and influence columns must align")
    if weights.shape[0] != n_contrasts:
        raise ValueError("Contrast weights and family must align")
    if n_contrasts != len(family):
        raise ValueError("Family names and contrast arrays must align")

    rng = np.random.default_rng(random_state)
    bootstrap_max = np.empty(n_bootstrap, dtype=float)
    offset = 0
    while offset < n_bootstrap:
        current = min(batch_size, n_bootstrap - offset)
        multipliers = rng.normal(size=(current, n))
        multipliers -= multipliers.mean(axis=1, keepdims=True)
        statistics = (multipliers @ influence_values) / n / standard_errors
        bootstrap_max[offset : offset + current] = np.max(np.abs(statistics), axis=1)
        offset += current

    observed = estimates / standard_errors
    observed_max = float(np.max(np.abs(observed)))
    critical = float(
        np.quantile(bootstrap_max, confidence_level, method="higher")
    )
    max_t_p = float((1 + np.sum(bootstrap_max >= observed_max)) / (n_bootstrap + 1))
    alpha = 1 - confidence_level
    simultaneous: list[SimultaneousContrastResult] = []
    for column, name in enumerate(family):
        adjusted_p = float(
            (1 + np.sum(bootstrap_max >= abs(observed[column]))) / (n_bootstrap + 1)
        )
        margin = critical * standard_errors[column]
        simultaneous.append(
            SimultaneousContrastResult(
                name=name,
                estimate=float(estimates[column]),
                standard_error=float(standard_errors[column]),
                statistic=float(observed[column]),
                simultaneous_confidence_interval=(
                    float(estimates[column] - margin),
                    float(estimates[column] + margin),
                ),
                adjusted_p_value=adjusted_p,
                rejected=adjusted_p <= alpha,
            )
        )

    contrast_covariance = weights @ group_covariance @ weights.T
    contrast_covariance = (contrast_covariance + contrast_covariance.T) / 2
    rank = int(np.linalg.matrix_rank(contrast_covariance, hermitian=True))
    if rank < 1:
        raise ValueError("The contrast covariance has zero numerical rank")
    inverse = np.linalg.pinv(contrast_covariance, hermitian=True)
    wald_statistic = max(float(estimates @ inverse @ estimates), 0.0)
    wald_p = float(chi2.sf(wald_statistic, rank))
    global_test = GlobalTestResult(
        max_t_statistic=observed_max,
        max_t_p_value=max_t_p,
        wald_statistic=wald_statistic,
        wald_degrees_of_freedom=rank,
        wald_p_value=wald_p,
    )
    status = InferenceStatus.WARNING if warning_reasons else InferenceStatus.COMPLETE
    key = _configuration_key(family, confidence_level, n_bootstrap, random_state)
    return SimultaneousInferenceResult(
        family=family,
        contrasts=tuple(simultaneous),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        multiplier="gaussian",
        critical_value=critical,
        global_test=global_test,
        status=status,
        reasons=warning_reasons,
        package_version=__version__,
        configuration_key=key,
    )
