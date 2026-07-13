"""Immutable analysis declarations for the design/analysis contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Any, Literal

JsonLabel = str | int | float | bool
Interpretation = Literal["descriptive", "causal"]


@dataclass(frozen=True, slots=True)
class SupportGeometryDeclaration:
    """Outcome-blind geometry settings for experimental Stage 5B bounds."""

    metric: Literal["shrinkage_mahalanobis"] = "shrinkage_mahalanobis"
    neighbor_count: int = 10
    softmin_temperature: Literal["design_median"] = "design_median"
    gamma_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    reference_partition: Literal["design"] = "design"
    smoothing: Literal["soft_k_nearest"] = "soft_k_nearest"

    def __post_init__(self) -> None:
        grid = tuple(float(value) for value in self.gamma_grid)
        if self.metric != "shrinkage_mahalanobis":
            raise ValueError("Stage 5B requires shrinkage_mahalanobis geometry")
        if self.neighbor_count < 1:
            raise ValueError("support geometry neighbor_count must be positive")
        if self.softmin_temperature != "design_median":
            raise ValueError("Stage 5B requires a design_median soft-min temperature")
        if not grid or grid[0] != 0 or any(not isfinite(value) or value < 0 for value in grid):
            raise ValueError("support geometry gamma_grid must start at zero and be finite")
        if any(right <= left for left, right in zip(grid, grid[1:], strict=False)):
            raise ValueError("support geometry gamma_grid must be strictly increasing")
        if self.reference_partition != "design" or self.smoothing != "soft_k_nearest":
            raise ValueError("Stage 5B requires design references and soft_k_nearest smoothing")
        object.__setattr__(self, "gamma_grid", grid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "neighbor_count": self.neighbor_count,
            "softmin_temperature": self.softmin_temperature,
            "gamma_grid": list(self.gamma_grid),
            "reference_partition": self.reference_partition,
            "smoothing": self.smoothing,
        }


@dataclass(frozen=True, slots=True)
class AnchoredBoundsDeclaration:
    """Outcome-range assumptions locked before Stage 5A outcome analysis."""

    outcome_lower: float
    outcome_upper: float
    assumption: Literal["bounded_outcome"] = "bounded_outcome"
    support_weight: Literal["scaled_harmonic_overlap"] = "scaled_harmonic_overlap"
    support_geometry: SupportGeometryDeclaration | None = None

    def __post_init__(self) -> None:
        lower = float(self.outcome_lower)
        upper = float(self.outcome_upper)
        if not (isfinite(lower) and isfinite(upper) and lower < upper):
            raise ValueError("anchored outcome bounds must be finite with lower < upper")
        if self.assumption != "bounded_outcome":
            raise ValueError("Stage 5A supports only the bounded_outcome assumption")
        if self.support_weight != "scaled_harmonic_overlap":
            raise ValueError("Stage 5A requires scaled_harmonic_overlap support weights")
        object.__setattr__(self, "outcome_lower", lower)
        object.__setattr__(self, "outcome_upper", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_lower": self.outcome_lower,
            "outcome_upper": self.outcome_upper,
            "assumption": self.assumption,
            "support_weight": self.support_weight,
            "support_geometry": (
                None if self.support_geometry is None else self.support_geometry.to_dict()
            ),
        }


def _json_label(value: Any) -> JsonLabel:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    raise TypeError("Group labels in declarations must be JSON scalar values")


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """A named contrast represented as (group label, weight) pairs."""

    name: str
    weights: tuple[tuple[JsonLabel, float], ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Contrast name must not be empty")
        normalized = tuple((_json_label(label), float(weight)) for label, weight in self.weights)
        if len(normalized) < 2:
            raise ValueError("A contrast must contain at least two group weights")
        labels = [label for label, _ in normalized]
        if len(set(labels)) != len(labels):
            raise ValueError("A contrast cannot repeat a group label")
        if abs(sum(weight for _, weight in normalized)) > 1e-10:
            raise ValueError("Contrast weights must sum to zero")
        if all(abs(weight) <= 1e-15 for _, weight in normalized):
            raise ValueError("A contrast must have at least one nonzero weight")
        object.__setattr__(self, "weights", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": [[label, weight] for label, weight in self.weights]}


@dataclass(frozen=True, slots=True)
class SCOVADeclaration:
    """Immutable declaration of columns, interpretation, folds, and contrasts."""

    outcome: str
    group: str
    covariates: tuple[str, ...]
    interpretation: Interpretation = "descriptive"
    n_splits: int = 5
    random_state: int = 0
    contrasts: tuple[ContrastSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariates", tuple(self.covariates))
        object.__setattr__(self, "contrasts", tuple(self.contrasts))
        roles = (self.outcome, self.group, *self.covariates)
        if not self.outcome or not self.group:
            raise ValueError("Outcome and group column names must not be empty")
        if not self.covariates:
            raise ValueError("At least one pre-group covariate is required")
        if len(set(roles)) != len(roles):
            raise ValueError("Outcome, group, and covariate column roles must be distinct")
        if self.interpretation not in ("descriptive", "causal"):
            raise ValueError("Interpretation must be 'descriptive' or 'causal'")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        names = [contrast.name for contrast in self.contrasts]
        if len(set(names)) != len(names):
            raise ValueError("Declared contrast names must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "group": self.group,
            "covariates": list(self.covariates),
            "interpretation": self.interpretation,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
        }

    @property
    def declaration_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


PathTarget = Literal["kway", "pairwise", "subset", "study"]


def _default_path_grid() -> tuple[float, ...]:
    return tuple(index / 20 for index in range(21))


@dataclass(frozen=True, slots=True)
class DesignDeclaration:
    """Immutable, outcome-free declaration for a Stage 4 design decision.

    This intentionally does not have an ``outcome`` field.  It records every
    setting allowed to affect graph selection and is the declaration component
    of a :class:`scova.design.DesignLock`.
    """

    group: str
    covariates: tuple[str, ...]
    interpretation: Interpretation = "descriptive"
    n_splits: int = 5
    random_state: int = 0
    contrasts: tuple[ContrastSpec, ...] = field(default_factory=tuple)
    lambdas: tuple[float, ...] = field(default_factory=_default_path_grid)
    target: PathTarget = "kway"
    active_groups: tuple[JsonLabel, ...] = ()
    candidate_subsets: tuple[tuple[JsonLabel, ...], ...] = field(default_factory=tuple)
    confidence_level: float = 0.95
    design_fraction: float = 0.5
    gate_policy: Mapping[str, Any] = field(default_factory=dict)
    learner_profile: str = "default"
    anchored_bounds: AnchoredBoundsDeclaration | None = None

    def __post_init__(self) -> None:
        covariates = tuple(self.covariates)
        contrasts = tuple(self.contrasts)
        lambdas = tuple(float(value) for value in self.lambdas)
        active_groups = tuple(_json_label(value) for value in self.active_groups)
        subsets = tuple(
            tuple(_json_label(value) for value in subset) for subset in self.candidate_subsets
        )
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "active_groups", active_groups)
        object.__setattr__(self, "candidate_subsets", subsets)
        object.__setattr__(self, "gate_policy", _canonical_mapping(self.gate_policy))
        if not self.group:
            raise ValueError("Group column name must not be empty")
        if not covariates:
            raise ValueError("At least one pre-group covariate is required")
        if self.group in covariates or len(set(covariates)) != len(covariates):
            raise ValueError("Group and covariate column roles must be distinct")
        if self.interpretation not in ("descriptive", "causal"):
            raise ValueError("Interpretation must be 'descriptive' or 'causal'")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.target not in ("kway", "pairwise", "subset", "study"):
            raise ValueError("unknown path target")
        if len(lambdas) < 2 or lambdas[0] != 0.0 or lambdas[-1] != 1.0:
            raise ValueError("lambda grid must contain 0 and 1 as endpoints")
        if any(value < 0 or value > 1 for value in lambdas):
            raise ValueError("lambda grid values must lie in [0, 1]")
        if any(right <= left for left, right in zip(lambdas, lambdas[1:], strict=False)):
            raise ValueError("lambda grid must be strictly increasing without duplicates")
        if self.target == "pairwise" and len(active_groups) != 2:
            raise ValueError("pairwise targets require exactly two active groups")
        if self.target == "subset" and len(active_groups) < 2:
            raise ValueError("subset targets require at least two active groups")
        if self.target in ("kway", "study") and active_groups:
            raise ValueError("kway and study targets cannot declare active groups")
        if len(set(active_groups)) != len(active_groups):
            raise ValueError("active_groups cannot contain duplicates")
        for subset in subsets:
            if len(subset) < 2 or len(set(subset)) != len(subset):
                raise ValueError("candidate subsets must contain at least two distinct groups")
        normalized_subsets = [frozenset(subset) for subset in subsets]
        if len(set(normalized_subsets)) != len(normalized_subsets):
            raise ValueError("candidate subsets cannot repeat the same group set")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if not 0 < self.design_fraction < 1:
            raise ValueError("design_fraction must lie strictly between 0 and 1")
        if not self.learner_profile.strip():
            raise ValueError("learner_profile must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "covariates": list(self.covariates),
            "interpretation": self.interpretation,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
            "lambdas": list(self.lambdas),
            "target": self.target,
            "active_groups": list(self.active_groups),
            "candidate_subsets": [list(subset) for subset in self.candidate_subsets],
            "confidence_level": self.confidence_level,
            "design_fraction": self.design_fraction,
            "gate_policy": self.gate_policy,
            "learner_profile": self.learner_profile,
            "anchored_bounds": (
                None if self.anchored_bounds is None else self.anchored_bounds.to_dict()
            ),
        }

    @property
    def declaration_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DesignDeclaration:
        return cls(
            group=str(values["group"]),
            covariates=tuple(values["covariates"]),
            interpretation=values.get("interpretation", "descriptive"),
            n_splits=int(values.get("n_splits", 5)),
            random_state=int(values.get("random_state", 0)),
            contrasts=tuple(
                ContrastSpec(
                    str(item["name"]),
                    tuple((pair[0], pair[1]) for pair in item["weights"]),
                )
                for item in values.get("contrasts", [])
            ),
            lambdas=tuple(values.get("lambdas", _default_path_grid())),
            target=values.get("target", "kway"),
            active_groups=tuple(values.get("active_groups", [])),
            candidate_subsets=tuple(tuple(item) for item in values.get("candidate_subsets", [])),
            confidence_level=float(values.get("confidence_level", 0.95)),
            design_fraction=float(values.get("design_fraction", 0.5)),
            gate_policy=values.get("gate_policy", {}),
            learner_profile=str(values.get("learner_profile", "default")),
            anchored_bounds=(
                None
                if values.get("anchored_bounds") is None
                else AnchoredBoundsDeclaration(
                    **{
                        **values["anchored_bounds"],
                        "support_geometry": (
                            None
                            if values["anchored_bounds"].get("support_geometry") is None
                            else SupportGeometryDeclaration(
                                **values["anchored_bounds"]["support_geometry"]
                            )
                        ),
                    }
                )
            ),
        )


def _canonical_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-only copy so declaration hashing is deterministic."""
    if not isinstance(values, Mapping):
        raise TypeError("gate_policy must be a mapping")
    try:
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError("gate_policy must be JSON serializable") from error
    if not isinstance(decoded, dict):  # Defensive: JSON object is required for a Mapping input.
        raise TypeError("gate_policy must encode as a JSON object")
    return decoded
