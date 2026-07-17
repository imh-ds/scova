"""Label-preserving results and separate persistence for SCOVA-CF."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from .._version import __version__
from ..declaration import JsonLabel
from ..inference import SimultaneousInferenceResult, run_simultaneous_inference
from .declaration import AnalysisMode, ClaimClass, claim_class_for_mode
from .status import SCOVACFStatus, SupportStatus

CF_ARTIFACT_TYPE = "scova-cf-result"
CF_SCHEMA_VERSION = 1


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _loaded_float(value: Any) -> float:
    return np.nan if value is None else float(value)


@dataclass(frozen=True, slots=True)
class CFDesignLock:
    declaration_hash: str
    design_hash: str
    fold_hash: str
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "declaration_hash": self.declaration_hash,
            "design_hash": self.design_hash,
            "fold_hash": self.fold_hash,
            "row_count": self.row_count,
        }


@dataclass(frozen=True, slots=True)
class SCOVACFContrastEstimate:
    name: str
    weights: np.ndarray
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    z_statistic: float
    p_value: float
    influence_values: np.ndarray
    mode: AnalysisMode
    claim_class: ClaimClass
    estimand_id: str
    target_population: str
    contrast_scale: str
    support_status: SupportStatus
    interval_type: str = "pointwise-wald"
    confirmatory: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": self.weights.tolist(),
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "confidence_interval": list(self.confidence_interval),
            "z_statistic": _json_float(self.z_statistic),
            "p_value": _json_float(self.p_value),
            "mode": self.mode.value,
            "claim_class": self.claim_class.value,
            "estimand_id": self.estimand_id,
            "target_population": self.target_population,
            "contrast_scale": self.contrast_scale,
            "support_status": self.support_status.value,
            "interval_type": self.interval_type,
            "confirmatory": self.confirmatory,
        }


@dataclass(frozen=True, slots=True)
class SCOVACFOmnibusResult:
    status_code: str
    reason: str
    statistic: float | None
    degrees_of_freedom: int | None
    p_value: float | None
    condition_number: float | None
    mode: AnalysisMode
    claim_class: ClaimClass
    support_status: SupportStatus
    confirmatory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "reason": self.reason,
            "statistic": self.statistic,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": self.p_value,
            "condition_number": self.condition_number,
            "mode": self.mode.value,
            "claim_class": self.claim_class.value,
            "support_status": self.support_status.value,
            "confirmatory": self.confirmatory,
        }


@dataclass(frozen=True, slots=True)
class SCOVACFInferenceResult:
    core: SimultaneousInferenceResult
    mode: AnalysisMode
    claim_class: ClaimClass
    estimand_id: str
    target_population: str
    contrast_scale: str
    support_status: SupportStatus
    confirmatory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference": self.core.to_dict(),
            "mode": self.mode.value,
            "claim_class": self.claim_class.value,
            "estimand_id": self.estimand_id,
            "target_population": self.target_population,
            "contrast_scale": self.contrast_scale,
            "support_status": self.support_status.value,
            "interval_type": "simultaneous-max-t",
            "confirmatory": self.confirmatory,
        }


def guarded_omnibus(
    *,
    means: np.ndarray,
    covariance: np.ndarray,
    mode: AnalysisMode,
    claim_class: ClaimClass,
    status: SCOVACFStatus,
    condition_limit: float = 1e12,
) -> SCOVACFOmnibusResult:
    """Calculate the canonical K-group omnibus without generalized inversion."""
    k = len(means)
    contrast = np.column_stack((np.eye(k - 1), -np.ones(k - 1)))
    contrast_covariance = contrast @ covariance @ contrast.T
    rank = int(np.linalg.matrix_rank(contrast_covariance))
    condition = float(np.linalg.cond(contrast_covariance))
    if rank != k - 1 or not np.isfinite(condition) or condition > condition_limit:
        return SCOVACFOmnibusResult(
            status_code="refused/singular-omnibus",
            reason="The omnibus contrast covariance lacks the required rank or conditioning",
            statistic=None,
            degrees_of_freedom=None,
            p_value=None,
            condition_number=condition if np.isfinite(condition) else None,
            mode=mode,
            claim_class=claim_class,
            support_status=status.support,
            confirmatory=False,
        )
    effect = contrast @ means
    statistic = float(effect @ np.linalg.solve(contrast_covariance, effect))
    return SCOVACFOmnibusResult(
        status_code="complete" if status.confirmatory else "limited/nonconfirmatory",
        reason=(
            "Calibrated support checks passed"
            if status.confirmatory
            else "Computed for navigation only; support status is not confirmatory"
        ),
        statistic=statistic,
        degrees_of_freedom=k - 1,
        p_value=float(chi2.sf(statistic, k - 1)),
        condition_number=condition,
        mode=mode,
        claim_class=claim_class,
        support_status=status.support,
        confirmatory=status.confirmatory,
    )


@dataclass(slots=True)
class SCOVACFResult:
    """Governed population-counterfactual result from the SCOVA-CF feature."""

    group_labels: tuple[JsonLabel, ...]
    covariate_names: tuple[str, ...]
    group_means: np.ndarray
    influence_values: np.ndarray
    covariance: np.ndarray
    fold_assignments: np.ndarray
    propensity_predictions: np.ndarray
    outcome_predictions: np.ndarray
    diagnostics: dict[str, Any]
    declaration: dict[str, Any]
    declaration_hash: str
    design_lock: CFDesignLock
    nuisance_metadata: dict[str, Any]
    mode: AnalysisMode
    claim_class: ClaimClass
    status: SCOVACFStatus
    estimand_id: str
    target_population: str
    outcome_units: str
    benchmarks: dict[str, Any]
    evidence_card: dict[str, Any]
    omnibus: SCOVACFOmnibusResult
    random_state: int
    package_version: str = __version__
    contrasts: dict[str, SCOVACFContrastEstimate] = field(default_factory=dict)
    inferences: dict[str, SCOVACFInferenceResult] = field(default_factory=dict)
    schema_version: int = CF_SCHEMA_VERSION

    @property
    def group_standard_errors(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    def group_confidence_intervals(self, confidence_level: float = 0.95) -> np.ndarray:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        critical = float(norm.ppf(0.5 + confidence_level / 2))
        margin = critical * self.group_standard_errors
        return np.column_stack((self.group_means - margin, self.group_means + margin))

    def _weight_array(
        self, weights: Mapping[JsonLabel, float] | Sequence[float]
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
        weights: Mapping[JsonLabel, float] | Sequence[float],
        *,
        name: str,
        confidence_level: float = 0.95,
    ) -> SCOVACFContrastEstimate:
        if not name:
            raise ValueError("A declared contrast name is required")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        array = self._weight_array(weights)
        estimate = float(array @ self.group_means)
        contrast_influence = self.influence_values @ array
        variance = float(array @ self.covariance @ array)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        z_statistic = estimate / standard_error if standard_error > 0 else np.nan
        p_value = float(2 * norm.sf(abs(z_statistic))) if standard_error > 0 else np.nan
        critical = float(norm.ppf(0.5 + confidence_level / 2))
        interval = (estimate - critical * standard_error, estimate + critical * standard_error)
        result = SCOVACFContrastEstimate(
            name=name,
            weights=array.copy(),
            estimate=estimate,
            standard_error=standard_error,
            confidence_interval=(float(interval[0]), float(interval[1])),
            z_statistic=float(z_statistic),
            p_value=p_value,
            influence_values=contrast_influence,
            mode=self.mode,
            claim_class=self.claim_class,
            estimand_id=self.estimand_id,
            target_population=self.target_population,
            contrast_scale=f"mean-difference ({self.outcome_units})",
            support_status=self.status.support,
            confirmatory=self.status.confirmatory,
        )
        self.contrasts[name] = result
        return result

    def infer(
        self,
        family: Sequence[str] | None = None,
        *,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1999,
        random_state: int | None = None,
        batch_size: int = 256,
    ) -> SCOVACFInferenceResult:
        names = tuple(self.contrasts) if family is None else tuple(family)
        if not names or len(set(names)) != len(names):
            raise ValueError("Inference requires a nonempty family of unique declared contrasts")
        unknown = [name for name in names if name not in self.contrasts]
        if unknown:
            raise ValueError(f"Unknown contrast names: {unknown}")
        selected = tuple(self.contrasts[name] for name in names)
        estimates = np.array([contrast.estimate for contrast in selected])
        standard_errors = np.array([contrast.standard_error for contrast in selected])
        influence = np.column_stack([contrast.influence_values for contrast in selected])
        weights = np.vstack([contrast.weights for contrast in selected])
        warning_reasons = () if self.status.confirmatory else (self.status.reason,)
        core = run_simultaneous_inference(
            family=names,
            estimates=estimates,
            standard_errors=standard_errors,
            influence_values=influence,
            weights=weights,
            group_covariance=self.covariance,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            random_state=self.random_state if random_state is None else random_state,
            batch_size=batch_size,
            warning_reasons=warning_reasons,
        )
        result = SCOVACFInferenceResult(
            core=core,
            mode=self.mode,
            claim_class=self.claim_class,
            estimand_id=self.estimand_id,
            target_population=self.target_population,
            contrast_scale=f"mean-difference ({self.outcome_units})",
            support_status=self.status.support,
            confirmatory=self.status.confirmatory,
        )
        self.inferences[core.configuration_key] = result
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": CF_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "declaration_hash": self.declaration_hash,
            "design_lock": self.design_lock.to_dict(),
            "mode": self.mode.value,
            "claim_class": self.claim_class.value,
            "status": self.status.to_dict(),
            "estimand_id": self.estimand_id,
            "target_population": self.target_population,
            "outcome_units": self.outcome_units,
            "group_labels": list(self.group_labels),
            "group_means": self.group_means.tolist(),
            "group_standard_errors": self.group_standard_errors.tolist(),
            "group_interval_type": "pointwise-wald",
            "group_confirmatory": self.status.confirmatory,
            "contrasts": {name: value.to_dict() for name, value in self.contrasts.items()},
            "omnibus": self.omnibus.to_dict(),
            "diagnostics": self.diagnostics,
            "benchmarks": self.benchmarks,
            "evidence_card": self.evidence_card,
            "nuisance_metadata": self.nuisance_metadata,
            "inferences": {name: value.to_dict() for name, value in self.inferences.items()},
        }

    def save(self, path: str | Path) -> None:
        """Persist a separate, non-pickle SCOVA-CF artifact."""
        metadata = self.to_dict()
        metadata["declaration"] = self.declaration
        metadata["covariate_names"] = list(self.covariate_names)
        metadata["random_state"] = self.random_state
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
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SCOVACFResult:
        """Load only a SCOVA-CF artifact, rejecting base-SCOVA artifacts."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("artifact_type") != CF_ARTIFACT_TYPE:
                raise ValueError("Artifact is not a SCOVA-CF result")
            if int(metadata["schema_version"]) != CF_SCHEMA_VERSION:
                raise ValueError("Unsupported SCOVA-CF artifact schema")
            mode = AnalysisMode(metadata["mode"])
            claim_class = ClaimClass(metadata["claim_class"])
            if claim_class is not claim_class_for_mode(mode):
                raise ValueError("SCOVA-CF artifact claim class does not match its analysis mode")
            status_values = metadata["status"]
            status = SCOVACFStatus(
                support=SupportStatus(status_values["support"]),
                code=str(status_values["code"]),
                reason=str(status_values["reason"]),
                confirmatory=bool(status_values["confirmatory"]),
            )
            lock_values = metadata["design_lock"]
            lock = CFDesignLock(
                declaration_hash=str(lock_values["declaration_hash"]),
                design_hash=str(lock_values["design_hash"]),
                fold_hash=str(lock_values["fold_hash"]),
                row_count=int(lock_values["row_count"]),
            )
            if lock.declaration_hash != metadata["declaration_hash"]:
                raise ValueError("SCOVA-CF design lock does not match the declaration")
            omnibus_values = metadata["omnibus"]
            omnibus = SCOVACFOmnibusResult(
                status_code=str(omnibus_values["status_code"]),
                reason=str(omnibus_values["reason"]),
                statistic=omnibus_values["statistic"],
                degrees_of_freedom=omnibus_values["degrees_of_freedom"],
                p_value=omnibus_values["p_value"],
                condition_number=omnibus_values["condition_number"],
                mode=AnalysisMode(omnibus_values["mode"]),
                claim_class=ClaimClass(omnibus_values["claim_class"]),
                support_status=SupportStatus(omnibus_values["support_status"]),
                confirmatory=bool(omnibus_values["confirmatory"]),
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
                declaration=metadata["declaration"],
                declaration_hash=str(metadata["declaration_hash"]),
                design_lock=lock,
                nuisance_metadata=metadata["nuisance_metadata"],
                mode=mode,
                claim_class=claim_class,
                status=status,
                estimand_id=str(metadata["estimand_id"]),
                target_population=str(metadata["target_population"]),
                outcome_units=str(metadata["outcome_units"]),
                benchmarks=metadata["benchmarks"],
                evidence_card=metadata["evidence_card"],
                omnibus=omnibus,
                random_state=int(metadata["random_state"]),
                package_version=str(metadata["package_version"]),
            )
            for name, values in metadata["contrasts"].items():
                result.contrasts[name] = SCOVACFContrastEstimate(
                    name=name,
                    weights=np.asarray(values["weights"], dtype=float),
                    estimate=float(values["estimate"]),
                    standard_error=float(values["standard_error"]),
                    confidence_interval=tuple(values["confidence_interval"]),
                    z_statistic=_loaded_float(values["z_statistic"]),
                    p_value=_loaded_float(values["p_value"]),
                    influence_values=archive[f"contrast_influence::{name}"].copy(),
                    mode=AnalysisMode(values["mode"]),
                    claim_class=ClaimClass(values["claim_class"]),
                    estimand_id=str(values["estimand_id"]),
                    target_population=str(values["target_population"]),
                    contrast_scale=str(values["contrast_scale"]),
                    support_status=SupportStatus(values["support_status"]),
                    interval_type=str(values["interval_type"]),
                    confirmatory=bool(values["confirmatory"]),
                )
            for key, values in metadata.get("inferences", {}).items():
                core = SimultaneousInferenceResult.from_dict(values["inference"])
                if key != core.configuration_key:
                    raise ValueError("Persisted CF inference key does not match its configuration")
                result.inferences[key] = SCOVACFInferenceResult(
                    core=core,
                    mode=AnalysisMode(values["mode"]),
                    claim_class=ClaimClass(values["claim_class"]),
                    estimand_id=str(values["estimand_id"]),
                    target_population=str(values["target_population"]),
                    contrast_scale=str(values["contrast_scale"]),
                    support_status=SupportStatus(values["support_status"]),
                    confirmatory=bool(values["confirmatory"]),
                )
        return result
