"""Experimental finite-grid smooth overlap paths and joint inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._version import __version__
from ..declaration import JsonLabel, SCOVADeclaration
from ..estimator import SCOVA, NuisancePredictions
from ..inference import InferenceStatus
from .gates import (
    DiagnosticThresholds,
    GateDecision,
    GateStatus,
    InferenceRefusedError,
    evaluate_path_gates,
    production_thresholds,
)
from .tilts import geometric_tilt_and_gradient

PathTarget = Literal["kway", "pairwise", "subset", "study"]
PATH_SCHEMA_VERSION = 4


def _default_grid() -> tuple[float, ...]:
    return tuple(index / 20 for index in range(21))


@dataclass(frozen=True, slots=True)
class PathDeclaration:
    """Immutable declaration of a finite, common-target overlap path."""

    base: SCOVADeclaration
    lambdas: tuple[float, ...] = field(default_factory=_default_grid)
    target: PathTarget = "kway"
    active_groups: tuple[JsonLabel, ...] = ()
    contrast_names: tuple[str, ...] = ()
    confidence_level: float = 0.95
    random_state: int | None = None
    thresholds: DiagnosticThresholds = field(default_factory=production_thresholds)

    def __post_init__(self) -> None:
        grid = tuple(float(value) for value in self.lambdas)
        object.__setattr__(self, "lambdas", grid)
        object.__setattr__(self, "active_groups", tuple(self.active_groups))
        object.__setattr__(self, "contrast_names", tuple(self.contrast_names))
        if len(grid) < 2 or grid[0] != 0.0 or grid[-1] != 1.0:
            raise ValueError("lambda grid must contain at least 0 and 1 as endpoints")
        if any(not np.isfinite(value) or value < 0 or value > 1 for value in grid):
            raise ValueError("lambda grid values must be finite and lie in [0, 1]")
        if any(right <= left for left, right in zip(grid, grid[1:], strict=False)):
            raise ValueError("lambda grid must be strictly increasing without duplicates")
        if self.target not in ("kway", "pairwise", "subset", "study"):
            raise ValueError("unknown path target")
        if self.target == "pairwise" and len(self.active_groups) != 2:
            raise ValueError("pairwise paths require exactly two active groups")
        if self.target == "subset" and len(self.active_groups) < 2:
            raise ValueError("subset paths require at least two active groups")
        if self.target in ("kway", "study") and self.active_groups:
            raise ValueError("kway and study paths infer active groups from the data")
        if len(set(self.active_groups)) != len(self.active_groups):
            raise ValueError("active groups cannot contain duplicates")
        if len(set(self.contrast_names)) != len(self.contrast_names):
            raise ValueError("contrast_names cannot contain duplicates")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        production = production_thresholds()
        if production.calibrated:
            self.thresholds.assert_at_least_as_strict_as(production)

    @property
    def seed(self) -> int:
        return self.base.random_state if self.random_state is None else self.random_state

    @property
    def declaration_hash(self) -> str:
        payload = {
            "base_hash": self.base.declaration_hash,
            "lambdas": self.lambdas,
            "target": self.target,
            "active_groups": self.active_groups,
            "contrast_names": self.contrast_names,
            "confidence_level": self.confidence_level,
            "random_state": self.seed,
            "thresholds": self.thresholds.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DriftProfile:
    standardized_mean_shifts: np.ndarray
    target_effective_sample_size: np.ndarray
    group_effective_sample_size: np.ndarray
    top_one_percent_weight_share: np.ndarray
    target_mean_propensity: np.ndarray


@dataclass(frozen=True, slots=True)
class ContrastPathResult:
    name: str
    weights: np.ndarray
    estimates: np.ndarray
    standard_errors: np.ndarray
    influence_values: np.ndarray


@dataclass(frozen=True, slots=True)
class SignCertificate:
    positive_lambdas: tuple[float, ...]
    negative_lambdas: tuple[float, ...]
    positive_suffix_start: float | None
    negative_suffix_start: float | None


@dataclass(frozen=True, slots=True)
class StabilityCertificate:
    estimated_maximum_drift: float
    simultaneous_upper_bound: float
    reference_lambda: float


@dataclass(frozen=True, slots=True)
class PathInferenceResult:
    family: tuple[str, ...]
    lambdas: tuple[float, ...]
    lower_bands: np.ndarray
    upper_bands: np.ndarray
    critical_value: float
    difference_critical_value: float
    sign_certificates: tuple[SignCertificate, ...]
    stability_certificates: tuple[StabilityCertificate, ...]
    confidence_level: float
    n_bootstrap: int
    random_state: int
    status: InferenceStatus
    reasons: tuple[str, ...]


def _suffix_start(mask: np.ndarray, lambdas: np.ndarray) -> float | None:
    for index in range(len(mask)):
        if np.all(mask[index:]):
            return float(lambdas[index])
    return None


@dataclass(slots=True)
class SCOVAPathResult:
    declaration_hash: str
    base_declaration_hash: str
    group_labels: tuple[JsonLabel, ...]
    covariate_names: tuple[str, ...]
    lambdas: np.ndarray
    target: PathTarget
    active_groups: tuple[JsonLabel, ...]
    group_means: np.ndarray
    naive_group_means: np.ndarray
    influence_values: np.ndarray
    naive_influence_values: np.ndarray
    standard_errors: np.ndarray
    drift: DriftProfile
    contrasts: dict[str, ContrastPathResult]
    diagnostics: dict[str, Any]
    thresholds: DiagnosticThresholds
    gate_decision: GateDecision
    random_state: int
    package_version: str
    inferences: list[PathInferenceResult] = field(default_factory=list)
    schema_version: int = PATH_SCHEMA_VERSION

    def infer(
        self,
        family: Sequence[str] | None = None,
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1999,
        random_state: int | None = None,
        batch_size: int = 256,
    ) -> PathInferenceResult:
        if self.gate_decision.status is GateStatus.REFUSE:
            raise InferenceRefusedError(
                "confirmatory path inference refused: " + "; ".join(self.gate_decision.reasons)
            )
        names = tuple(self.contrasts) if family is None else tuple(family)
        if not names:
            raise ValueError("path inference requires at least one contrast")
        if len(set(names)) != len(names):
            raise ValueError("path inference family cannot contain duplicate names")
        unknown = [name for name in names if name not in self.contrasts]
        if unknown:
            raise ValueError(f"unknown path contrast names: {unknown}")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if n_bootstrap < 1 or batch_size < 1:
            raise ValueError("n_bootstrap and batch_size must be positive")
        selected = [self.contrasts[name] for name in names]
        estimates = np.stack([item.estimates for item in selected])
        errors = np.stack([item.standard_errors for item in selected])
        if np.any(errors <= 0) or not np.all(np.isfinite(errors)):
            raise ValueError("every path point must have a finite positive standard error")
        influence = np.concatenate([item.influence_values for item in selected], axis=1)
        flat_errors = errors.reshape(-1)
        n = influence.shape[0]
        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        maxima = np.empty(n_bootstrap)
        difference_maxima = np.empty(n_bootstrap)
        difference_influence = np.concatenate(
            [item.influence_values[:, :-1] - item.influence_values[:, [-1]] for item in selected],
            axis=1,
        )
        difference_estimates = np.stack(
            [item.estimates[:-1] - item.estimates[-1] for item in selected]
        )
        difference_errors = np.sqrt(
            np.sum(np.square(difference_influence), axis=0) / (n * (n - 1))
        )
        if np.any(difference_errors <= 0) or not np.all(np.isfinite(difference_errors)):
            raise ValueError("path differences must have finite positive standard errors")
        offset = 0
        while offset < n_bootstrap:
            current = min(batch_size, n_bootstrap - offset)
            multipliers = rng.normal(size=(current, n))
            multipliers -= multipliers.mean(axis=1, keepdims=True)
            statistics = (multipliers @ influence) / n / flat_errors
            difference_statistics = (
                (multipliers @ difference_influence) / n / difference_errors
            )
            maxima[offset : offset + current] = np.max(np.abs(statistics), axis=1)
            difference_maxima[offset : offset + current] = np.max(
                np.abs(difference_statistics), axis=1
            )
            offset += current
        critical = float(np.quantile(maxima, confidence_level, method="higher"))
        difference_critical = float(
            np.quantile(difference_maxima, confidence_level, method="higher")
        )
        lower = estimates - critical * errors
        upper = estimates + critical * errors
        lambdas = np.asarray(self.lambdas)
        signs: list[SignCertificate] = []
        stability: list[StabilityCertificate] = []
        for row in range(len(names)):
            positive = lower[row] > 0
            negative = upper[row] < 0
            signs.append(
                SignCertificate(
                    positive_lambdas=tuple(float(value) for value in lambdas[positive]),
                    negative_lambdas=tuple(float(value) for value in lambdas[negative]),
                    positive_suffix_start=_suffix_start(positive, lambdas),
                    negative_suffix_start=_suffix_start(negative, lambdas),
                )
            )
            row_errors = difference_errors[
                row * (len(lambdas) - 1) : (row + 1) * (len(lambdas) - 1)
            ]
            row_difference = difference_estimates[row]
            stability.append(
                StabilityCertificate(
                    estimated_maximum_drift=float(np.max(np.abs(row_difference))),
                    simultaneous_upper_bound=float(
                        np.max(np.abs(row_difference) + difference_critical * row_errors)
                    ),
                    reference_lambda=float(lambdas[-1]),
                )
            )
        warnings = tuple(self.gate_decision.reasons)
        result = PathInferenceResult(
            family=names,
            lambdas=tuple(float(value) for value in lambdas),
            lower_bands=lower,
            upper_bands=upper,
            critical_value=critical,
            difference_critical_value=difference_critical,
            sign_certificates=tuple(signs),
            stability_certificates=tuple(stability),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            random_state=seed,
            status=InferenceStatus.WARNING if warnings else InferenceStatus.COMPLETE,
            reasons=warnings,
        )
        self.inferences.append(result)
        return result

    def save(self, path: str | Path) -> None:
        metadata = {
            "schema_version": self.schema_version,
            "declaration_hash": self.declaration_hash,
            "base_declaration_hash": self.base_declaration_hash,
            "group_labels": self.group_labels,
            "covariate_names": self.covariate_names,
            "target": self.target,
            "active_groups": self.active_groups,
            "contrast_names": list(self.contrasts),
            "contrast_weights": {
                name: value.weights.tolist() for name, value in self.contrasts.items()
            },
            "diagnostics": self.diagnostics,
            "random_state": self.random_state,
            "package_version": self.package_version,
            "thresholds": self.thresholds.to_dict(),
            "gate_decision": self.gate_decision.to_dict(),
        }
        arrays: dict[str, np.ndarray] = {
            "metadata": np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
            "lambdas": self.lambdas,
            "group_means": self.group_means,
            "naive_group_means": self.naive_group_means,
            "influence_values": self.influence_values,
            "naive_influence_values": self.naive_influence_values,
            "standard_errors": self.standard_errors,
            "drift_standardized_mean_shifts": self.drift.standardized_mean_shifts,
            "drift_target_ess": self.drift.target_effective_sample_size,
            "drift_group_ess": self.drift.group_effective_sample_size,
            "drift_concentration": self.drift.top_one_percent_weight_share,
            "drift_mean_propensity": self.drift.target_mean_propensity,
        }
        for name, contrast in self.contrasts.items():
            arrays[f"contrast_estimates::{name}"] = contrast.estimates
            arrays[f"contrast_standard_errors::{name}"] = contrast.standard_errors
            arrays[f"contrast_influence::{name}"] = contrast.influence_values
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SCOVAPathResult:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            stored_schema = int(metadata["schema_version"])
            if stored_schema not in (3, PATH_SCHEMA_VERSION):
                raise ValueError("unsupported experimental path schema")
            contrasts = {
                name: ContrastPathResult(
                    name=name,
                    weights=np.asarray(metadata["contrast_weights"][name]),
                    estimates=archive[f"contrast_estimates::{name}"].copy(),
                    standard_errors=archive[f"contrast_standard_errors::{name}"].copy(),
                    influence_values=archive[f"contrast_influence::{name}"].copy(),
                )
                for name in metadata["contrast_names"]
            }
            drift = DriftProfile(
                standardized_mean_shifts=archive[
                    "drift_standardized_mean_shifts"
                ].copy(),
                target_effective_sample_size=archive["drift_target_ess"].copy(),
                group_effective_sample_size=archive["drift_group_ess"].copy(),
                top_one_percent_weight_share=archive["drift_concentration"].copy(),
                target_mean_propensity=archive["drift_mean_propensity"].copy(),
            )
            thresholds = DiagnosticThresholds.from_dict(
                metadata.get("thresholds", DiagnosticThresholds().to_dict())
            )
            gate_decision = (
                GateDecision.from_dict(metadata["gate_decision"])
                if "gate_decision" in metadata
                else GateDecision(
                    status=GateStatus.WARNING,
                    metrics=(),
                    reasons=("migrated schema-3 result requires gate reevaluation",),
                    threshold_version=thresholds.version,
                    calibrated=False,
                )
            )
            return cls(
                declaration_hash=metadata["declaration_hash"],
                base_declaration_hash=metadata["base_declaration_hash"],
                group_labels=tuple(metadata["group_labels"]),
                covariate_names=tuple(metadata["covariate_names"]),
                lambdas=archive["lambdas"].copy(),
                target=metadata["target"],
                active_groups=tuple(metadata["active_groups"]),
                group_means=archive["group_means"].copy(),
                naive_group_means=archive["naive_group_means"].copy(),
                influence_values=archive["influence_values"].copy(),
                naive_influence_values=archive["naive_influence_values"].copy(),
                standard_errors=archive["standard_errors"].copy(),
                drift=drift,
                contrasts=contrasts,
                diagnostics=metadata["diagnostics"],
                thresholds=thresholds,
                gate_decision=gate_decision,
                random_state=int(metadata["random_state"]),
                package_version=metadata["package_version"],
            )


def _drift_profile(
    x: np.ndarray,
    group_codes: np.ndarray,
    propensity: np.ndarray,
    tilt: np.ndarray,
) -> DriftProfile:
    normalized = tilt / tilt.sum(axis=0, keepdims=True)
    study_mean = x.mean(axis=0)
    study_sd = np.where(x.std(axis=0, ddof=1) > 0, x.std(axis=0, ddof=1), 1.0)
    target_means = normalized.T @ x
    shifts = (target_means - study_mean) / study_sd
    target_ess = 1 / np.square(normalized).sum(axis=0)
    n_groups = propensity.shape[1]
    group_ess = np.empty((tilt.shape[1], n_groups))
    for code in range(n_groups):
        weights = (group_codes == code)[:, None] * tilt / propensity[:, code, None]
        group_ess[:, code] = np.square(weights.sum(axis=0)) / np.square(weights).sum(axis=0)
    top_count = max(1, int(np.ceil(0.01 * len(x))))
    concentration = np.sort(normalized, axis=0)[-top_count:].sum(axis=0)
    mean_propensity = normalized.T @ propensity
    return DriftProfile(shifts, target_ess, group_ess, concentration, mean_propensity)


def fit_path(
    data: Any,
    declaration: PathDeclaration,
    *,
    estimator: SCOVA | None = None,
    nuisance_predictions: NuisancePredictions | None = None,
    crossfit_instability: float = 0.0,
) -> SCOVAPathResult:
    """Fit one experimental finite-grid target path without refitting nuisances."""
    engine = estimator or SCOVA()
    base_result = engine.fit(
        data, declaration.base, nuisance_predictions=nuisance_predictions
    )
    x, outcome, group_codes, labels = engine._validate_data(data, declaration.base)
    propensity = base_result.propensity_predictions
    outcome_regression = base_result.outcome_predictions
    label_to_code = {label: code for code, label in enumerate(labels)}
    if declaration.target in ("kway", "study"):
        active_codes = tuple(range(len(labels)))
        active_groups = labels
    else:
        unknown = set(declaration.active_groups).difference(labels)
        if unknown:
            raise ValueError(f"unknown active groups: {sorted(map(str, unknown))}")
        active_codes = tuple(label_to_code[label] for label in declaration.active_groups)
        active_groups = declaration.active_groups
    lambdas = np.asarray(declaration.lambdas)
    tilt, gradient = geometric_tilt_and_gradient(
        propensity, lambdas, active_codes, target=declaration.target
    )
    one_hot = np.eye(len(labels))[group_codes]
    q = np.einsum("nlk,nk->nl", gradient, one_hot - propensity)
    residual = one_hot / propensity * (outcome[:, None] - outcome_regression)
    denominator = tilt.sum(axis=0)
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise ValueError("path normalization is non-positive or non-finite")
    plug_in = np.einsum("nl,nk->lk", tilt, outcome_regression) / denominator[:, None]
    weighted_residual = tilt[:, :, None] * residual[:, None, :]
    correction = (outcome_regression[:, None, :] - plug_in[None, :, :]) * q[:, :, None]
    group_means = plug_in + (weighted_residual + correction).sum(axis=0) / denominator[:, None]
    naive_group_means = plug_in + weighted_residual.sum(axis=0) / denominator[:, None]
    eta = denominator / len(data)
    influence = (
        weighted_residual
        + (outcome_regression[:, None, :] - group_means[None, :, :])
        * (tilt + q)[:, :, None]
    ) / eta[None, :, None]
    influence -= influence.mean(axis=0, keepdims=True)
    naive_influence = (
        weighted_residual
        + (outcome_regression[:, None, :] - naive_group_means[None, :, :])
        * tilt[:, :, None]
    ) / eta[None, :, None]
    naive_influence -= naive_influence.mean(axis=0, keepdims=True)
    standard_errors = np.sqrt(
        np.sum(np.square(influence), axis=0) / (len(data) * (len(data) - 1))
    )
    available = declaration.contrast_names or tuple(base_result.contrasts)
    unknown_contrasts = [name for name in available if name not in base_result.contrasts]
    if unknown_contrasts:
        raise ValueError(f"unknown contrast names: {unknown_contrasts}")
    contrasts: dict[str, ContrastPathResult] = {}
    inactive = set(range(len(labels))).difference(active_codes)
    for name in available:
        base_contrast = base_result.contrasts[name]
        if declaration.target in ("pairwise", "subset") and any(
            abs(base_contrast.weights[code]) > 1e-15 for code in inactive
        ):
            if declaration.contrast_names:
                raise ValueError(f"contrast {name!r} uses a group outside the target subset")
            continue
        estimates = group_means @ base_contrast.weights
        contrast_influence = np.einsum("nlk,k->nl", influence, base_contrast.weights)
        errors = np.sqrt(
            np.sum(np.square(contrast_influence), axis=0)
            / (len(data) * (len(data) - 1))
        )
        contrasts[name] = ContrastPathResult(
            name, base_contrast.weights.copy(), estimates, errors, contrast_influence
        )
    if not contrasts:
        raise ValueError("no fitted contrasts are compatible with the declared target")
    diagnostics = dict(base_result.diagnostics)
    diagnostics["experimental"] = True
    diagnostics["inference_scope"] = "declared-finite-grid"
    squared_influence = np.square(influence)
    top_count = max(1, int(np.ceil(0.01 * len(data))))
    influence_denominator = squared_influence.sum(axis=0)
    influence_top = np.sort(squared_influence, axis=0)[-top_count:].sum(axis=0)
    influence_share = np.divide(
        influence_top,
        influence_denominator,
        out=np.zeros_like(influence_top),
        where=influence_denominator > 0,
    )
    drift = _drift_profile(x, group_codes, propensity, tilt)
    calibration = diagnostics["propensity_calibration"]
    target_weights = tilt / denominator[None, :]
    target_covariate_means = target_weights.T @ x
    covariate_scale = np.where(x.std(axis=0, ddof=1) > 0, x.std(axis=0, ddof=1), 1.0)
    path_balance = np.empty((len(lambdas), len(labels)))
    for code in range(len(labels)):
        raw_group_weights = (
            (group_codes == code)[:, None] * tilt / propensity[:, code, None]
        )
        normalized_group_weights = raw_group_weights / raw_group_weights.sum(
            axis=0, keepdims=True
        )
        group_covariate_means = normalized_group_weights.T @ x
        path_balance[:, code] = np.max(
            np.abs((group_covariate_means - target_covariate_means) / covariate_scale),
            axis=1,
        )
    contrast_influence_share: dict[str, np.ndarray] = {}
    for name, contrast in contrasts.items():
        squared = np.square(contrast.influence_values)
        total = squared.sum(axis=0)
        top = np.sort(squared, axis=0)[-top_count:].sum(axis=0)
        contrast_influence_share[name] = np.divide(
            top,
            total,
            out=np.zeros_like(top),
            where=total > 0,
        )
    maximum_contrast_influence_share = max(
        float(np.max(values)) for values in contrast_influence_share.values()
    )
    diagnostics["path_gate_grid"] = {
        "schema_version": 1,
        "lambdas": lambdas.tolist(),
        "target_ess_ratio": (drift.target_effective_sample_size / len(data)).tolist(),
        "group_effective_sample_size": drift.group_effective_sample_size.tolist(),
        "group_influence_variance_share": influence_share.tolist(),
        "contrast_influence_variance_share": {
            name: values.tolist() for name, values in contrast_influence_share.items()
        },
        "target_weight_concentration": drift.top_one_percent_weight_share.tolist(),
        "maximum_weighted_covariate_imbalance": path_balance.tolist(),
        "normalization_finite": np.isfinite(denominator).tolist(),
        "group_variance_finite": np.isfinite(standard_errors).tolist(),
        "contrast_variance_finite": {
            name: np.isfinite(contrast.standard_errors).tolist()
            for name, contrast in contrasts.items()
        },
    }
    if not np.isfinite(crossfit_instability) or crossfit_instability < 0:
        raise ValueError("crossfit_instability must be finite and nonnegative")
    diagnostics["crossfit_instability"] = float(crossfit_instability)
    gate_decision = evaluate_path_gates(
        min_group_ess=float(np.min(drift.group_effective_sample_size)),
        target_ess_ratio=float(np.min(drift.target_effective_sample_size) / len(data)),
        max_influence_share=max(
            float(np.max(influence_share)), maximum_contrast_influence_share
        ),
        max_weight_concentration=float(np.max(drift.top_one_percent_weight_share)),
        min_propensity_q01=float(
            min(item["q01"] for item in diagnostics["propensity_quantiles"].values())
        ),
        max_calibration_error=float(calibration["worst_class_expected_calibration_error"]),
        max_balance=float(np.max(path_balance)),
        crossfit_instability=float(crossfit_instability),
        numerical_valid=bool(
            np.all(np.isfinite(group_means))
            and np.all(np.isfinite(influence))
            and np.all(standard_errors > 0)
        ),
        thresholds=declaration.thresholds,
    )
    return SCOVAPathResult(
        declaration_hash=declaration.declaration_hash,
        base_declaration_hash=declaration.base.declaration_hash,
        group_labels=labels,
        covariate_names=declaration.base.covariates,
        lambdas=lambdas,
        target=declaration.target,
        active_groups=active_groups,
        group_means=group_means,
        naive_group_means=naive_group_means,
        influence_values=influence,
        naive_influence_values=naive_influence,
        standard_errors=standard_errors,
        drift=drift,
        contrasts=contrasts,
        diagnostics=diagnostics,
        thresholds=declaration.thresholds,
        gate_decision=gate_decision,
        random_state=declaration.seed,
        package_version=__version__,
    )
