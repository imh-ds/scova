"""Stage 5A graph-conditional bounded-outcome anchored bounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from ._version import __version__
from .declaration import JsonLabel

ANCHOR_SCHEMA_VERSION = 1


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
                    groups=tuple(item["groups"]),  # type: ignore[arg-type]
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
