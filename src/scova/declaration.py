"""Immutable analysis declarations for the design/analysis contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

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
