"""Stage 5A graph-conditional bounded-outcome anchored bounds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ._version import __version__
from .declaration import JsonLabel

ANCHOR_SCHEMA_VERSION = 1
LIPSCHITZ_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class AnchoredContrastResult:
    """B1 identified-set endpoints for one graph-supported pair."""

    name: str
    groups: tuple[JsonLabel, JsonLabel]
    identified_component: float
    non_support_mass: float
    lower_endpoint: float
    upper_endpoint: float
    confidence_interval: tuple[float, float]
    lower_influence_values: np.ndarray
    upper_influence_values: np.ndarray

    @property
    def lower_standard_error(self) -> float:
        n = len(self.lower_influence_values)
        return float(np.sqrt(np.sum(np.square(self.lower_influence_values)) / (n * (n - 1))))

    @property
    def upper_standard_error(self) -> float:
        n = len(self.upper_influence_values)
        return float(np.sqrt(np.sum(np.square(self.upper_influence_values)) / (n * (n - 1))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "groups": list(self.groups),
            "identified_component": self.identified_component,
            "non_support_mass": self.non_support_mass,
            "lower_endpoint": self.lower_endpoint,
            "upper_endpoint": self.upper_endpoint,
            "confidence_interval": list(self.confidence_interval),
            "lower_standard_error": self.lower_standard_error,
            "upper_standard_error": self.upper_standard_error,
        }


@dataclass(frozen=True, slots=True)
class AnchoredBoundsResult:
    """Persistable Stage 5A result, deliberately separate from core schemas."""

    design_lock: str
    interpretation: str
    verdict: str
    assumption: str
    support_weight: str
    outcome_lower: float | None
    outcome_upper: float | None
    confidence_level: float
    random_state: int
    contrasts: tuple[AnchoredContrastResult, ...]
    refused: tuple[str, ...]
    package_version: str = __version__
    schema_version: int = ANCHOR_SCHEMA_VERSION

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "design_lock": self.design_lock,
            "interpretation": self.interpretation,
            "verdict": self.verdict,
            "inference_scope": "graph-conditional-outer-split-bounded-anchor",
            "assumption": self.assumption,
            "support_weight": self.support_weight,
            "outcome_lower": self.outcome_lower,
            "outcome_upper": self.outcome_upper,
            "confidence_level": self.confidence_level,
            "random_state": self.random_state,
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
            "refused": list(self.refused),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        metadata = self.report()
        arrays: dict[str, np.ndarray] = {
            "metadata": np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
        }
        for index, contrast in enumerate(self.contrasts):
            arrays[f"lower_influence::{index}"] = contrast.lower_influence_values
            arrays[f"upper_influence::{index}"] = contrast.upper_influence_values
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> AnchoredBoundsResult:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if int(metadata["schema_version"]) != ANCHOR_SCHEMA_VERSION:
                raise ValueError("unsupported anchored-bounds schema")
            contrasts = tuple(
                AnchoredContrastResult(
                    name=str(item["name"]),
                    groups=tuple(item["groups"]),
                    identified_component=float(item["identified_component"]),
                    non_support_mass=float(item["non_support_mass"]),
                    lower_endpoint=float(item["lower_endpoint"]),
                    upper_endpoint=float(item["upper_endpoint"]),
                    confidence_interval=tuple(item["confidence_interval"]),
                    lower_influence_values=archive[f"lower_influence::{index}"].copy(),
                    upper_influence_values=archive[f"upper_influence::{index}"].copy(),
                )
                for index, item in enumerate(metadata["contrasts"])
            )
        return cls(
            design_lock=str(metadata["design_lock"]),
            interpretation=str(metadata["interpretation"]),
            verdict=str(metadata["verdict"]),
            assumption=str(metadata["assumption"]),
            support_weight=str(metadata["support_weight"]),
            outcome_lower=(
                None if metadata["outcome_lower"] is None else float(metadata["outcome_lower"])
            ),
            outcome_upper=(
                None if metadata["outcome_upper"] is None else float(metadata["outcome_upper"])
            ),
            confidence_level=float(metadata["confidence_level"]),
            random_state=int(metadata["random_state"]),
            contrasts=contrasts,
            refused=tuple(metadata["refused"]),
            package_version=str(metadata["package_version"]),
        )


@dataclass(frozen=True, slots=True)
class LipschitzContrastResult:
    """Experimental B2 endpoints on a declared finite Gamma grid."""

    name: str
    groups: tuple[JsonLabel, JsonLabel]
    identified_component: float
    gamma_grid: np.ndarray
    lower_endpoints: np.ndarray
    upper_endpoints: np.ndarray
    smooth_distances: np.ndarray
    reference_predictions: np.ndarray
    exploratory_positive_gamma: float | None
    lower_influence_values: np.ndarray | None = None
    upper_influence_values: np.ndarray | None = None
    endpoint_standard_errors: np.ndarray | None = None
    confidence_intervals: np.ndarray | None = None
    inference_status: str = "unavailable"
    boundary_proximity: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "name": self.name,
            "groups": list(self.groups),
            "identified_component": self.identified_component,
            "gamma_grid": self.gamma_grid.tolist(),
            "lower_endpoints": self.lower_endpoints.tolist(),
            "upper_endpoints": self.upper_endpoints.tolist(),
            "exploratory_positive_gamma": self.exploratory_positive_gamma,
            "inference_status": self.inference_status,
        }
        if self.endpoint_standard_errors is not None:
            report["endpoint_standard_errors"] = self.endpoint_standard_errors.tolist()
        if self.confidence_intervals is not None:
            report["confidence_intervals"] = self.confidence_intervals.tolist()
        if self.boundary_proximity is not None:
            report["boundary_proximity"] = self.boundary_proximity.tolist()
        return report


@dataclass(frozen=True, slots=True)
class LipschitzAnchorResult:
    """Separate, non-confirmatory artifact for Stage 5B B2 analyses."""

    design_lock: str
    geometry_digest: str | None
    verdict: str
    outcome_lower: float | None
    outcome_upper: float | None
    gamma_grid: np.ndarray
    contrasts: tuple[LipschitzContrastResult, ...]
    refused: tuple[str, ...]
    confidence_level: float | None = None
    inference_method: str = "unavailable"
    transport_diagnostics: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LIPSCHITZ_SCHEMA_VERSION

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_lock": self.design_lock,
            "geometry_digest": self.geometry_digest,
            "verdict": self.verdict,
            "inference_scope": "experimental-graph-conditional-lipschitz-anchor",
            "outcome_lower": self.outcome_lower,
            "outcome_upper": self.outcome_upper,
            "gamma_grid": self.gamma_grid.tolist(),
            "confidence_level": self.confidence_level,
            "inference_method": self.inference_method,
            "transport_diagnostics": self.transport_diagnostics,
            "contrasts": [item.to_dict() for item in self.contrasts],
            "refused": list(self.refused),
        }

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "metadata": np.array(json.dumps(self.report(), sort_keys=True, allow_nan=False))
        }
        for index, contrast in enumerate(self.contrasts):
            arrays[f"distances::{index}"] = contrast.smooth_distances
            arrays[f"reference_predictions::{index}"] = contrast.reference_predictions
            if contrast.lower_influence_values is not None:
                arrays[f"lower_influence::{index}"] = contrast.lower_influence_values
            if contrast.upper_influence_values is not None:
                arrays[f"upper_influence::{index}"] = contrast.upper_influence_values
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> LipschitzAnchorResult:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            version = int(metadata["schema_version"])
            if version not in (1, LIPSCHITZ_SCHEMA_VERSION):
                raise ValueError("unsupported Lipschitz-anchor schema")
            contrasts = tuple(
                LipschitzContrastResult(
                    name=str(item["name"]),
                    groups=tuple(item["groups"]),
                    identified_component=float(item["identified_component"]),
                    gamma_grid=np.asarray(item["gamma_grid"], dtype=float),
                    lower_endpoints=np.asarray(item["lower_endpoints"], dtype=float),
                    upper_endpoints=np.asarray(item["upper_endpoints"], dtype=float),
                    smooth_distances=archive[f"distances::{index}"].copy(),
                    reference_predictions=archive[f"reference_predictions::{index}"].copy(),
                    exploratory_positive_gamma=item["exploratory_positive_gamma"],
                    lower_influence_values=(
                        archive[f"lower_influence::{index}"].copy()
                        if version >= 2 and f"lower_influence::{index}" in archive.files
                        else None
                    ),
                    upper_influence_values=(
                        archive[f"upper_influence::{index}"].copy()
                        if version >= 2 and f"upper_influence::{index}" in archive.files
                        else None
                    ),
                    endpoint_standard_errors=(
                        np.asarray(item["endpoint_standard_errors"], dtype=float)
                        if version >= 2 and item.get("endpoint_standard_errors") is not None
                        else None
                    ),
                    confidence_intervals=(
                        np.asarray(item["confidence_intervals"], dtype=float)
                        if version >= 2 and item.get("confidence_intervals") is not None
                        else None
                    ),
                    inference_status=(
                        str(item.get("inference_status", "unavailable"))
                        if version >= 2 else "legacy-unavailable"
                    ),
                    boundary_proximity=(
                        np.asarray(item["boundary_proximity"], dtype=bool)
                        if version >= 2 and item.get("boundary_proximity") is not None
                        else None
                    ),
                )
                for index, item in enumerate(metadata["contrasts"])
            )
        return cls(
            design_lock=str(metadata["design_lock"]),
            geometry_digest=metadata["geometry_digest"],
            verdict=str(metadata["verdict"]),
            outcome_lower=metadata["outcome_lower"],
            outcome_upper=metadata["outcome_upper"],
            gamma_grid=np.asarray(metadata["gamma_grid"], dtype=float),
            contrasts=contrasts,
            refused=tuple(metadata["refused"]),
            confidence_level=(
                None if version == 1 else metadata.get("confidence_level")
            ),
            inference_method=(
                "legacy-unavailable" if version == 1 else str(metadata.get("inference_method"))
            ),
            transport_diagnostics=(
                {} if version == 1 else dict(metadata.get("transport_diagnostics", {}))
            ),
            schema_version=version,
        )


def scaled_harmonic_overlap_and_gradient(
    propensity: np.ndarray, active_codes: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth ``omega_S = |S|^2 h_ow,S`` and its simplex gradient."""
    probability = np.asarray(propensity, dtype=float)
    if probability.ndim != 2 or np.any(probability <= 0) or not np.all(np.isfinite(probability)):
        raise ValueError("propensities must be positive and finite")
    first, second = active_codes
    selected = probability[:, [first, second]]
    harmonic = 1 / np.sum(1 / selected, axis=1)
    omega = 4 * harmonic
    gradient = np.zeros_like(probability)
    gradient[:, first] = 4 * np.square(harmonic) / np.square(probability[:, first])
    gradient[:, second] = 4 * np.square(harmonic) / np.square(probability[:, second])
    if np.any(omega < 0) or np.any(omega > 1 + 1e-12):
        raise ValueError("scaled harmonic overlap must lie in [0, 1]")
    return omega, gradient


def bounded_pairwise_anchor(
    *,
    groups: tuple[JsonLabel, JsonLabel],
    group_codes: np.ndarray,
    outcomes: np.ndarray,
    propensity: np.ndarray,
    outcome_predictions: np.ndarray,
    active_codes: tuple[int, int],
    outcome_lower: float,
    outcome_upper: float,
    confidence_level: float,
) -> AnchoredContrastResult:
    """Estimate B1 pairwise endpoints and their corrected influence values."""
    n = len(outcomes)
    first, second = active_codes
    omega, gradient = scaled_harmonic_overlap_and_gradient(propensity, active_codes)
    one_hot = np.eye(propensity.shape[1])[group_codes]
    q = np.einsum("nk,nk->n", gradient, one_hot - propensity)
    delta = outcome_predictions[:, first] - outcome_predictions[:, second]
    residual = (
        (group_codes == first) * (outcomes - outcome_predictions[:, first]) / propensity[:, first]
        - (group_codes == second) * (outcomes - outcome_predictions[:, second])
        / propensity[:, second]
    )
    core_terms = omega * residual + omega * delta + delta * q
    core = float(np.mean(core_terms))
    mass = float(np.mean(1 - omega))
    lower_remainder = outcome_lower - outcome_upper
    upper_remainder = outcome_upper - outcome_lower
    lower = core + lower_remainder * mass
    upper = core + upper_remainder * mass
    core_influence = core_terms - core
    mass_influence = (1 - omega) - mass - q
    lower_influence = core_influence + lower_remainder * mass_influence
    upper_influence = core_influence + upper_remainder * mass_influence
    lower_influence -= lower_influence.mean()
    upper_influence -= upper_influence.mean()
    lower_se = float(np.sqrt(np.sum(np.square(lower_influence)) / (n * (n - 1))))
    upper_se = float(np.sqrt(np.sum(np.square(upper_influence)) / (n * (n - 1))))
    critical = float(norm.ppf(0.5 + confidence_level / 2))
    return AnchoredContrastResult(
        name=f"{groups[0]} - {groups[1]}",
        groups=groups,
        identified_component=core,
        non_support_mass=mass,
        lower_endpoint=float(lower),
        upper_endpoint=float(upper),
        confidence_interval=(
            float(lower - critical * lower_se),
            float(upper + critical * upper_se),
        ),
        lower_influence_values=lower_influence,
        upper_influence_values=upper_influence,
    )


def lipschitz_pairwise_anchor(
    *,
    bounded: AnchoredContrastResult,
    propensity: np.ndarray,
    active_codes: tuple[int, int],
    gamma_grid: np.ndarray,
    smooth_distances: np.ndarray,
    reference_predictions: np.ndarray,
    outcome_lower: float,
    outcome_upper: float,
    transport_residuals: np.ndarray | None = None,
    confidence_level: float | None = None,
    boundary_tolerance: float = 1e-12,
) -> LipschitzContrastResult:
    """Construct clipped experimental B2 endpoints from frozen geometry outputs."""
    omega, _ = scaled_harmonic_overlap_and_gradient(propensity, active_codes)
    if smooth_distances.shape != (len(omega), 2) or reference_predictions.shape != (len(omega), 2):
        raise ValueError("B2 distances and reference predictions must have two aligned columns")
    if transport_residuals is not None and transport_residuals.shape != (len(omega), 2):
        raise ValueError("B2 transport residuals must have two aligned columns")
    lower: list[float] = []
    upper: list[float] = []
    lower_influence: list[np.ndarray] = []
    upper_influence: list[np.ndarray] = []
    boundary: list[bool] = []
    remainder_weight = 1 - omega
    for gamma in gamma_grid:
        raw_lower_a = reference_predictions[:, 0] - gamma * smooth_distances[:, 0]
        raw_upper_a = reference_predictions[:, 0] + gamma * smooth_distances[:, 0]
        raw_lower_b = reference_predictions[:, 1] - gamma * smooth_distances[:, 1]
        raw_upper_b = reference_predictions[:, 1] + gamma * smooth_distances[:, 1]
        lower_a = np.clip(
            raw_lower_a,
            outcome_lower,
            outcome_upper,
        )
        upper_a = np.clip(
            raw_upper_a,
            outcome_lower,
            outcome_upper,
        )
        lower_b = np.clip(
            raw_lower_b,
            outcome_lower,
            outcome_upper,
        )
        upper_b = np.clip(
            raw_upper_b,
            outcome_lower,
            outcome_upper,
        )
        lower_remainder = lower_a - upper_b
        upper_remainder = upper_a - lower_b
        transport = np.zeros(len(omega))
        if transport_residuals is not None:
            transport = remainder_weight * (transport_residuals[:, 0] - transport_residuals[:, 1])
        lower.append(
            float(
                bounded.identified_component
                + np.mean(remainder_weight * lower_remainder)
                + transport.mean()
            )
        )
        upper.append(
            float(
                bounded.identified_component
                + np.mean(remainder_weight * upper_remainder)
                + transport.mean()
            )
        )
        # Start from B1's corrected endpoint EIF, then replace its constant
        # remainder by the frozen smooth-reference remainder.  The augmented
        # residual score is cross-fitted upstream and supplies the regular
        # reference-prediction channel.
        lower_delta = remainder_weight * (lower_remainder - (outcome_lower - outcome_upper))
        upper_delta = remainder_weight * (upper_remainder - (outcome_upper - outcome_lower))
        lower_phi = bounded.lower_influence_values + lower_delta - lower_delta.mean()
        upper_phi = bounded.upper_influence_values + upper_delta - upper_delta.mean()
        if transport_residuals is not None:
            lower_phi = lower_phi + transport - transport.mean()
            upper_phi = upper_phi + transport - transport.mean()
        lower_influence.append(lower_phi)
        upper_influence.append(upper_phi)
        raw = np.column_stack((raw_lower_a, raw_upper_a, raw_lower_b, raw_upper_b))
        boundary.append(bool(
            np.any(np.isclose(raw, outcome_lower, rtol=0, atol=boundary_tolerance))
            or np.any(np.isclose(raw, outcome_upper, rtol=0, atol=boundary_tolerance))
        ))
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)
    lower_influence_array = np.column_stack(lower_influence)
    upper_influence_array = np.column_stack(upper_influence)
    standard_errors, intervals = _imbens_manski_intervals(
        lower_array, upper_array, lower_influence_array, upper_influence_array, confidence_level
    )
    positive = gamma_grid[lower_array > 0]
    return LipschitzContrastResult(
        name=bounded.name,
        groups=bounded.groups,
        identified_component=bounded.identified_component,
        gamma_grid=np.asarray(gamma_grid, dtype=float),
        lower_endpoints=lower_array,
        upper_endpoints=upper_array,
        smooth_distances=smooth_distances,
        reference_predictions=reference_predictions,
        exploratory_positive_gamma=(None if not len(positive) else float(np.max(positive))),
        lower_influence_values=lower_influence_array,
        upper_influence_values=upper_influence_array,
        endpoint_standard_errors=standard_errors,
        confidence_intervals=intervals,
        inference_status=("blocked-boundary" if any(boundary) else "experimental-eif"),
        boundary_proximity=np.asarray(boundary, dtype=bool),
    )


def _imbens_manski_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    lower_influence: np.ndarray,
    upper_influence: np.ndarray,
    confidence_level: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint SEs and Imbens--Manski confidence intervals per Gamma."""
    if confidence_level is None or not 0 < confidence_level < 1:
        raise ValueError("B2 endpoint inference requires a confidence level in (0, 1)")
    n = len(lower_influence)
    if n < 2:
        raise ValueError("B2 endpoint inference requires at least two rows")
    lower_se = np.sqrt(np.sum(np.square(lower_influence), axis=0) / (n * (n - 1)))
    upper_se = np.sqrt(np.sum(np.square(upper_influence), axis=0) / (n * (n - 1)))
    intervals = np.empty((len(lower), 2), dtype=float)
    alpha = 1 - confidence_level
    for index, (left, right, left_se, right_se) in enumerate(
        zip(lower, upper, lower_se, upper_se, strict=True)
    ):
        scale = max(float(left_se), float(right_se))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("B2 endpoint influence values have non-positive standard error")
        standardized_gap = max(float(right - left) / scale, 0.0)
        one_sided = float(norm.ppf(1 - alpha))
        two_sided = float(norm.ppf(1 - alpha / 2))
        def coverage(value: float, gap: float = standardized_gap) -> float:
            return float(norm.cdf(value + gap) - norm.cdf(-value) - confidence_level)

        critical = brentq(coverage, one_sided, two_sided)
        intervals[index] = (left - critical * left_se, right + critical * right_se)
    return np.column_stack((lower_se, upper_se)), intervals
