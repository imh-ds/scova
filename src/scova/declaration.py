"""Immutable analysis declarations for the design/analysis contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, Mapping

JsonLabel = str | int | float | bool
Interpretation = Literal["descriptive", "causal"]


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
        }

    @property
    def declaration_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


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
