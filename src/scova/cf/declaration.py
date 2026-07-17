"""Governed declarations for the opt-in SCOVA-CF feature."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Literal, TypeAlias

from ..declaration import ContrastSpec, JsonLabel


class AnalysisMode(str, Enum):
    """Permitted SCOVA-CF analysis modes."""

    RANDOMIZED = "randomized"
    OBSERVATIONAL_CAUSAL = "observational-causal"
    STANDARDIZED_ASSOCIATIONAL = "standardized-associational"


class ClaimClass(str, Enum):
    """Epistemic claim class derived exclusively from the analysis mode."""

    RANDOMIZATION_SUPPORTED = "randomization-supported"
    ASSUMPTION_DEPENDENT_CAUSAL = "assumption-dependent-causal"
    ASSOCIATIONAL = "associational"


def claim_class_for_mode(mode: AnalysisMode) -> ClaimClass:
    if mode is AnalysisMode.RANDOMIZED:
        return ClaimClass.RANDOMIZATION_SUPPORTED
    if mode is AnalysisMode.OBSERVATIONAL_CAUSAL:
        return ClaimClass.ASSUMPTION_DEPENDENT_CAUSAL
    return ClaimClass.ASSOCIATIONAL


def _validate_probability_pairs(
    values: tuple[tuple[JsonLabel, float], ...], *, context: str
) -> None:
    if len(values) < 2:
        raise ValueError(f"{context} must declare at least two groups")
    labels = [label for label, _ in values]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{context} group labels must be unique")
    probabilities = [float(probability) for _, probability in values]
    if any(probability <= 0 or probability > 1 for probability in probabilities):
        raise ValueError(f"{context} probabilities must be positive and at most one")
    if abs(sum(probabilities) - 1.0) > 1e-10:
        raise ValueError(f"{context} probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class KnownAssignment:
    """Known randomization probabilities, optionally varying by design stratum."""

    probabilities: tuple[tuple[JsonLabel, float], ...] = ()
    stratum_column: str | None = None
    stratum_probabilities: tuple[
        tuple[JsonLabel, tuple[tuple[JsonLabel, float], ...]], ...
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        object.__setattr__(self, "stratum_probabilities", tuple(self.stratum_probabilities))
        has_marginal = bool(self.probabilities)
        has_stratified = bool(self.stratum_probabilities)
        if has_marginal == has_stratified:
            raise ValueError(
                "KnownAssignment requires exactly one of probabilities or stratum_probabilities"
            )
        if has_marginal:
            if self.stratum_column is not None:
                raise ValueError("stratum_column requires stratum_probabilities")
            _validate_probability_pairs(self.probabilities, context="Known assignment")
        else:
            if not self.stratum_column:
                raise ValueError("Stratified assignment requires a nonempty stratum_column")
            strata = [stratum for stratum, _ in self.stratum_probabilities]
            if len(set(strata)) != len(strata):
                raise ValueError("Known assignment strata must be unique")
            reference_labels: tuple[JsonLabel, ...] | None = None
            for stratum, probabilities in self.stratum_probabilities:
                normalized = tuple(probabilities)
                _validate_probability_pairs(
                    normalized, context=f"Known assignment stratum {stratum!r}"
                )
                labels = tuple(label for label, _ in normalized)
                if reference_labels is None:
                    reference_labels = labels
                elif set(labels) != set(reference_labels):
                    raise ValueError("Every stratum must declare the same randomized groups")

    @property
    def group_labels(self) -> tuple[JsonLabel, ...]:
        values = self.probabilities or self.stratum_probabilities[0][1]
        return tuple(label for label, _ in values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "known",
            "probabilities": [[label, probability] for label, probability in self.probabilities],
            "stratum_column": self.stratum_column,
            "stratum_probabilities": [
                [
                    stratum,
                    [[label, probability] for label, probability in probabilities],
                ]
                for stratum, probabilities in self.stratum_probabilities
            ],
        }


@dataclass(frozen=True, slots=True)
class EstimatedAssignment:
    """Declared generalized-propensity learner policy for nonrandomized modes."""

    nuisance_strategy: Literal["adaptive", "linear", "custom"] = "adaptive"

    def __post_init__(self) -> None:
        if self.nuisance_strategy not in {"adaptive", "linear", "custom"}:
            raise ValueError("Unsupported assignment nuisance strategy")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "estimated", "nuisance_strategy": self.nuisance_strategy}


AssignmentSpecification: TypeAlias = KnownAssignment | EstimatedAssignment


@dataclass(frozen=True, slots=True)
class SupportPolicy:
    """Versioned support thresholds; defaults are intentionally provisional."""

    min_group_count: int = 20
    min_ess_ratio: float = 0.25
    max_normalized_weight: float = 0.20
    max_top_one_percent_weight_share: float = 0.35
    calibrated: bool = False
    version: str = "cf-provisional-1"

    def __post_init__(self) -> None:
        if self.min_group_count < 2:
            raise ValueError("min_group_count must be at least two")
        if not 0 < self.min_ess_ratio <= 1:
            raise ValueError("min_ess_ratio must lie in (0, 1]")
        if not 0 < self.max_normalized_weight <= 1:
            raise ValueError("max_normalized_weight must lie in (0, 1]")
        if not 0 < self.max_top_one_percent_weight_share <= 1:
            raise ValueError("max_top_one_percent_weight_share must lie in (0, 1]")
        if not self.version:
            raise ValueError("Support policy version must not be empty")
        if self.calibrated:
            raise ValueError(
                "No calibrated SCOVA-CF support profile is shipped in this release"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_group_count": self.min_group_count,
            "min_ess_ratio": self.min_ess_ratio,
            "max_normalized_weight": self.max_normalized_weight,
            "max_top_one_percent_weight_share": self.max_top_one_percent_weight_share,
            "calibrated": self.calibrated,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class DeclarationAmendment:
    """Auditable amendment recorded without mutating a prior declaration."""

    timestamp: str
    reason: str
    changes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "changes", tuple(self.changes))
        if not self.timestamp or not self.reason:
            raise ValueError("Amendments require a timestamp and reason")
        names = [name for name, _ in self.changes]
        if not names or len(set(names)) != len(names):
            raise ValueError("Amendment change names must be nonempty and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "reason": self.reason,
            "changes": {name: value for name, value in self.changes},
        }


@dataclass(frozen=True, slots=True)
class SCOVACFDeclaration:
    """Immutable declaration for a population-counterfactual SCOVA-CF analysis."""

    outcome: str
    group: str
    covariates: tuple[str, ...]
    mode: AnalysisMode
    scientific_question: str
    eligibility: str
    target_population: str
    group_definitions: tuple[tuple[JsonLabel, str], ...]
    outcome_time: str
    outcome_units: str
    covariate_rationales: tuple[tuple[str, str], ...]
    assignment: AssignmentSpecification
    outcome_nuisance_strategy: Literal["adaptive", "linear", "custom"] = "adaptive"
    n_splits: int = 5
    random_state: int = 0
    stability_seeds: tuple[int, ...] = field(default_factory=tuple)
    contrasts: tuple[ContrastSpec, ...] = field(default_factory=tuple)
    post_treatment_covariates: tuple[str, ...] = field(default_factory=tuple)
    outcome_direction: Literal["higher", "lower"] = "higher"
    outcome_type: Literal["continuous"] = "continuous"
    estimand_id: Literal["study-population-standardized-means"] = (
        "study-population-standardized-means"
    )
    estimator: Literal["aipw-unnormalized"] = "aipw-unnormalized"
    interval_method: Literal["wald+multiplier-bootstrap"] = "wald+multiplier-bootstrap"
    missing_outcome_policy: Literal["complete"] = "complete"
    support_policy: SupportPolicy = field(default_factory=SupportPolicy)
    sensitivity_analysis: str | None = None
    amendments: tuple[DeclarationAmendment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "covariates", tuple(self.covariates))
        object.__setattr__(self, "contrasts", tuple(self.contrasts))
        object.__setattr__(self, "stability_seeds", tuple(self.stability_seeds))
        object.__setattr__(self, "group_definitions", tuple(self.group_definitions))
        object.__setattr__(self, "covariate_rationales", tuple(self.covariate_rationales))
        object.__setattr__(self, "post_treatment_covariates", tuple(self.post_treatment_covariates))
        object.__setattr__(self, "amendments", tuple(self.amendments))
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", AnalysisMode(self.mode))
        roles = (self.outcome, self.group, *self.covariates)
        if not self.outcome or not self.group or not self.covariates:
            raise ValueError("Outcome, group, and at least one baseline covariate are required")
        if len(set(roles)) != len(roles):
            raise ValueError("Outcome, group, and covariate roles must be distinct")
        required_text = {
            "scientific_question": self.scientific_question,
            "eligibility": self.eligibility,
            "target_population": self.target_population,
            "outcome_time": self.outcome_time,
            "outcome_units": self.outcome_units,
        }
        empty = [name for name, value in required_text.items() if not value.strip()]
        if empty:
            raise ValueError(f"Required scientific declaration fields are empty: {empty}")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least two")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in self.stability_seeds
        ):
            raise ValueError("stability_seeds must contain nonnegative integers")
        if len(set(self.stability_seeds)) != len(self.stability_seeds):
            raise ValueError("stability_seeds must be unique")
        if self.random_state in self.stability_seeds:
            raise ValueError("stability_seeds must not contain the primary random_state")
        if self.outcome_nuisance_strategy not in {"adaptive", "linear", "custom"}:
            raise ValueError("Unsupported outcome nuisance strategy")
        group_labels = [label for label, definition in self.group_definitions if definition.strip()]
        if (
            len(group_labels) != len(self.group_definitions)
            or len(set(group_labels)) != len(group_labels)
            or len(group_labels) < 2
        ):
            raise ValueError("At least two unique groups require nonempty operational definitions")
        rationales = dict(self.covariate_rationales)
        if len(rationales) != len(self.covariate_rationales) or set(rationales) != set(
            self.covariates
        ) or any(
            not rationale.strip() for rationale in rationales.values()
        ):
            raise ValueError("Every and only declared covariate must have a nonempty rationale")
        if self.mode is AnalysisMode.RANDOMIZED and not isinstance(
            self.assignment, KnownAssignment
        ):
            raise ValueError("Randomized mode requires known assignment probabilities")
        if self.mode is not AnalysisMode.RANDOMIZED and not isinstance(
            self.assignment, EstimatedAssignment
        ):
            raise ValueError("Nonrandomized modes require estimated assignment probabilities")
        names = [contrast.name for contrast in self.contrasts]
        if not names:
            raise ValueError("SCOVA-CF requires at least one prespecified contrast")
        if len(set(names)) != len(names):
            raise ValueError("Declared contrast names must be unique")
        if self.sensitivity_analysis is not None and not self.sensitivity_analysis.strip():
            raise ValueError("A declared sensitivity analysis must not be blank")

    @property
    def claim_class(self) -> ClaimClass:
        return claim_class_for_mode(self.mode)

    @property
    def declared_group_labels(self) -> tuple[JsonLabel, ...]:
        return tuple(label for label, _ in self.group_definitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "outcome": self.outcome,
            "group": self.group,
            "covariates": list(self.covariates),
            "mode": self.mode.value,
            "claim_class": self.claim_class.value,
            "scientific_question": self.scientific_question,
            "eligibility": self.eligibility,
            "target_population": self.target_population,
            "group_definitions": [[label, value] for label, value in self.group_definitions],
            "outcome_time": self.outcome_time,
            "outcome_units": self.outcome_units,
            "outcome_direction": self.outcome_direction,
            "outcome_type": self.outcome_type,
            "covariate_rationales": dict(self.covariate_rationales),
            "assignment": self.assignment.to_dict(),
            "outcome_nuisance_strategy": self.outcome_nuisance_strategy,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "stability_seeds": list(self.stability_seeds),
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
            "post_treatment_covariates": list(self.post_treatment_covariates),
            "estimand_id": self.estimand_id,
            "estimator": self.estimator,
            "interval_method": self.interval_method,
            "missing_outcome_policy": self.missing_outcome_policy,
            "support_policy": self.support_policy.to_dict(),
            "sensitivity_analysis": self.sensitivity_analysis,
            "amendments": [amendment.to_dict() for amendment in self.amendments],
        }

    @property
    def declaration_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()
