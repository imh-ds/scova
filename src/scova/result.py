"""Results, contrasts, and safe versioned persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

SCHEMA_VERSION = 1


class Verdict(str, Enum):
    """Verdicts supported by the fixed-target milestone."""

    CERTIFIED = "certified"
    DESCRIPTIVE_ONLY = "descriptive-only"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    name: str
    weights: np.ndarray
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    z_statistic: float
    p_value: float
    influence_values: np.ndarray


@dataclass(slots=True)
class SCOVAResult:
    """A fitted fixed-target analysis and all data needed for new contrasts."""

    group_labels: tuple[str | int | float | bool, ...]
    covariate_names: tuple[str, ...]
    group_means: np.ndarray
    influence_values: np.ndarray
    covariance: np.ndarray
    fold_assignments: np.ndarray
    propensity_predictions: np.ndarray
    outcome_predictions: np.ndarray
    diagnostics: dict[str, Any]
    declaration_hash: str
    nuisance_metadata: dict[str, Any]
    verdict: Verdict
    package_version: str
    contrasts: dict[str, ContrastEstimate] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def group_standard_errors(self) -> np.ndarray:
        """Pointwise standard errors for the standardized group means."""
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    def group_confidence_intervals(self, confidence_level: float = 0.95) -> np.ndarray:
        """Return pointwise Wald intervals in canonical group order."""
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        critical = float(norm.ppf(0.5 + confidence_level / 2))
        margin = critical * self.group_standard_errors
        return np.column_stack((self.group_means - margin, self.group_means + margin))

    def _weight_array(
        self, weights: Mapping[str | int | float | bool, float] | Sequence[float]
    ) -> np.ndarray:
        if isinstance(weights, Mapping):
            unknown = set(weights).difference(self.group_labels)
            if unknown:
                raise ValueError(f"Contrast contains unknown groups: {sorted(map(str, unknown))}")
            array = np.array([weights.get(label, 0.0) for label in self.group_labels], dtype=float)
        else:
            array = np.asarray(weights, dtype=float)
        if array.shape != (len(self.group_labels),):
            raise ValueError(f"Contrast must have {len(self.group_labels)} weights")
        if not np.all(np.isfinite(array)):
            raise ValueError("Contrast weights must be finite")
        if abs(float(array.sum())) > 1e-10:
            raise ValueError("Contrast weights must sum to zero")
        if np.all(np.abs(array) <= 1e-15):
            raise ValueError("Contrast must have a nonzero weight")
        return array

    def contrast(
        self,
        weights: Mapping[str | int | float | bool, float] | Sequence[float],
        name: str | None = None,
        confidence_level: float = 0.95,
    ) -> ContrastEstimate:
        """Compute a pointwise Wald contrast without refitting nuisances."""
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        array = self._weight_array(weights)
        estimate = float(array @ self.group_means)
        contrast_influence = self.influence_values @ array
        variance = float(array @ self.covariance @ array)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        z_statistic = estimate / standard_error if standard_error > 0 else np.nan
        p_value = float(2 * norm.sf(abs(z_statistic))) if standard_error > 0 else np.nan
        critical = float(norm.ppf(0.5 + confidence_level / 2))
        interval = (estimate - critical * standard_error, estimate + critical * standard_error)
        contrast_name = name or "custom"
        result = ContrastEstimate(
            name=contrast_name,
            weights=array.copy(),
            estimate=estimate,
            standard_error=standard_error,
            confidence_interval=(float(interval[0]), float(interval[1])),
            z_statistic=float(z_statistic),
            p_value=p_value,
            influence_values=contrast_influence,
        )
        if name is not None:
            self.contrasts[name] = result
        return result

    def save(self, path: str | Path) -> None:
        """Save numeric results and JSON metadata without pickle."""
        destination = Path(path)
        contrast_metadata = {
            name: {
                "weights": contrast.weights.tolist(),
                "estimate": contrast.estimate,
                "standard_error": contrast.standard_error,
                "confidence_interval": list(contrast.confidence_interval),
                "z_statistic": contrast.z_statistic,
                "p_value": contrast.p_value,
            }
            for name, contrast in self.contrasts.items()
        }
        metadata = {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "group_labels": list(self.group_labels),
            "covariate_names": list(self.covariate_names),
            "diagnostics": self.diagnostics,
            "declaration_hash": self.declaration_hash,
            "nuisance_metadata": self.nuisance_metadata,
            "verdict": self.verdict.value,
            "contrasts": contrast_metadata,
        }
        arrays: dict[str, np.ndarray] = {
            "metadata": np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
            "group_means": self.group_means,
            "influence_values": self.influence_values,
            "covariance": self.covariance,
            "fold_assignments": self.fold_assignments,
            "propensity_predictions": self.propensity_predictions,
            "outcome_predictions": self.outcome_predictions,
        }
        for name, contrast in self.contrasts.items():
            arrays[f"contrast_influence::{name}"] = contrast.influence_values
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SCOVAResult:
        """Load a result, rejecting unknown future schemas."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["schema_version"] != SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported result schema "
                    f"{metadata['schema_version']}; expected {SCHEMA_VERSION}"
                )
            result = cls(
                group_labels=tuple(metadata["group_labels"]),
                covariate_names=tuple(metadata["covariate_names"]),
                group_means=archive["group_means"].copy(),
                influence_values=archive["influence_values"].copy(),
                covariance=archive["covariance"].copy(),
                fold_assignments=archive["fold_assignments"].copy(),
                propensity_predictions=archive["propensity_predictions"].copy(),
                outcome_predictions=archive["outcome_predictions"].copy(),
                diagnostics=metadata["diagnostics"],
                declaration_hash=metadata["declaration_hash"],
                nuisance_metadata=metadata["nuisance_metadata"],
                verdict=Verdict(metadata["verdict"]),
                package_version=metadata["package_version"],
                schema_version=metadata["schema_version"],
            )
            for name, values in metadata["contrasts"].items():
                result.contrasts[name] = ContrastEstimate(
                    name=name,
                    weights=np.asarray(values["weights"], dtype=float),
                    estimate=float(values["estimate"]),
                    standard_error=float(values["standard_error"]),
                    confidence_interval=tuple(values["confidence_interval"]),
                    z_statistic=float(values["z_statistic"]),
                    p_value=float(values["p_value"]),
                    influence_values=archive[f"contrast_influence::{name}"].copy(),
                )
        return result
