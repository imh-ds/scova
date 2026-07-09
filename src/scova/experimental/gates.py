"""Typed reliability gates for experimental path inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


class GateStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class DiagnosticThresholds:
    """Versioned monotone warning/refusal thresholds."""

    version: str = "stage3-candidate-v1"
    calibrated: bool = False
    artifact_sha256: str | None = None
    min_group_ess_warning: float = 50.0
    min_group_ess_refuse: float = 10.0
    min_target_ess_ratio_warning: float = 0.25
    min_target_ess_ratio_refuse: float = 0.10
    max_influence_share_warning: float = 0.25
    max_influence_share_refuse: float = 0.75
    max_weight_concentration_warning: float = 0.10
    max_weight_concentration_refuse: float = 0.20
    min_propensity_q01_warning: float = 0.01
    min_propensity_q01_refuse: float = 0.001
    max_calibration_error_warning: float = 0.10
    max_calibration_error_refuse: float = 0.20
    max_balance_warning: float = 0.20
    max_balance_refuse: float = 0.50
    max_crossfit_instability_warning: float = 0.25
    max_crossfit_instability_refuse: float = 0.50

    def __post_init__(self) -> None:
        if self.calibrated and not self.artifact_sha256:
            raise ValueError("calibrated thresholds require a verified artifact hash")
        pairs = (
            (self.min_group_ess_refuse, self.min_group_ess_warning, "group ESS"),
            (
                self.min_target_ess_ratio_refuse,
                self.min_target_ess_ratio_warning,
                "target ESS ratio",
            ),
        )
        for refusal, warning, name in pairs:
            if refusal < 0 or warning < refusal:
                raise ValueError(f"invalid monotone minimum thresholds for {name}")
        maximum_pairs = (
            (
                self.max_influence_share_warning,
                self.max_influence_share_refuse,
                "influence share",
            ),
            (
                self.max_weight_concentration_warning,
                self.max_weight_concentration_refuse,
                "weight concentration",
            ),
            (
                self.max_calibration_error_warning,
                self.max_calibration_error_refuse,
                "calibration error",
            ),
            (self.max_balance_warning, self.max_balance_refuse, "balance"),
            (
                self.max_crossfit_instability_warning,
                self.max_crossfit_instability_refuse,
                "cross-fit instability",
            ),
        )
        for warning, refusal, name in maximum_pairs:
            if warning < 0 or refusal < warning:
                raise ValueError(f"invalid monotone maximum thresholds for {name}")
        if not (0 < self.min_propensity_q01_refuse <= self.min_propensity_q01_warning <= 1):
            raise ValueError("invalid propensity quantile thresholds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_at_least_as_strict_as(self, baseline: DiagnosticThresholds) -> None:
        minimum_fields = (
            "min_group_ess_warning",
            "min_group_ess_refuse",
            "min_target_ess_ratio_warning",
            "min_target_ess_ratio_refuse",
            "min_propensity_q01_warning",
            "min_propensity_q01_refuse",
        )
        maximum_fields = (
            "max_influence_share_warning",
            "max_influence_share_refuse",
            "max_weight_concentration_warning",
            "max_weight_concentration_refuse",
            "max_calibration_error_warning",
            "max_calibration_error_refuse",
            "max_balance_warning",
            "max_balance_refuse",
            "max_crossfit_instability_warning",
            "max_crossfit_instability_refuse",
        )
        weaker = [name for name in minimum_fields if getattr(self, name) < getattr(baseline, name)]
        weaker.extend(
            name for name in maximum_fields if getattr(self, name) > getattr(baseline, name)
        )
        if weaker:
            raise ValueError(
                "user thresholds cannot weaken production defaults: " + ", ".join(weaker)
            )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DiagnosticThresholds:
        return cls(**values)

    @classmethod
    def from_calibration_artifact(cls, values: dict[str, Any]) -> DiagnosticThresholds:
        if not values.get("calibrated") or "sha256" not in values:
            raise ValueError("threshold artifact is not calibrated and checksummed")
        unsigned = dict(values)
        claimed_hash = str(unsigned.pop("sha256"))
        actual_hash = sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if claimed_hash != actual_hash:
            raise ValueError("threshold artifact checksum is invalid")
        passing = values["pass_profile"]
        floor = values["warning_floor_profile"]
        return cls(
            version=values["version"],
            calibrated=True,
            artifact_sha256=claimed_hash,
            min_group_ess_warning=passing["min_group_ess"],
            min_group_ess_refuse=floor["min_group_ess"],
            min_target_ess_ratio_warning=passing["min_target_ess_ratio"],
            min_target_ess_ratio_refuse=floor["min_target_ess_ratio"],
            max_influence_share_warning=passing["max_influence_share"],
            max_influence_share_refuse=floor["max_influence_share"],
            max_weight_concentration_warning=passing["max_weight_concentration"],
            max_weight_concentration_refuse=floor["max_weight_concentration"],
            min_propensity_q01_warning=passing["min_propensity_q01"],
            min_propensity_q01_refuse=floor["min_propensity_q01"],
            max_calibration_error_warning=passing["max_calibration_error"],
            max_calibration_error_refuse=floor["max_calibration_error"],
            max_balance_warning=passing["max_balance"],
            max_balance_refuse=floor["max_balance"],
            max_crossfit_instability_warning=passing["max_crossfit_instability"],
            max_crossfit_instability_refuse=floor["max_crossfit_instability"],
        )


def production_thresholds() -> DiagnosticThresholds:
    """Load the packaged lockfile, falling back to non-certifying provisional gates."""
    artifact = Path(__file__).with_name("data") / "stage3_thresholds.json"
    if not artifact.exists():
        return DiagnosticThresholds()
    values = json.loads(artifact.read_text(encoding="utf-8"))
    return DiagnosticThresholds.from_calibration_artifact(values)


@dataclass(frozen=True, slots=True)
class GateMetric:
    name: str
    value: float
    warning_threshold: float
    refusal_threshold: float
    direction: str
    status: GateStatus


@dataclass(frozen=True, slots=True)
class GateDecision:
    status: GateStatus
    metrics: tuple[GateMetric, ...]
    reasons: tuple[str, ...]
    threshold_version: str
    calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "warning_threshold": metric.warning_threshold,
                    "refusal_threshold": metric.refusal_threshold,
                    "direction": metric.direction,
                    "status": metric.status.value,
                }
                for metric in self.metrics
            ],
            "reasons": list(self.reasons),
            "threshold_version": self.threshold_version,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GateDecision:
        return cls(
            status=GateStatus(values["status"]),
            metrics=tuple(
                GateMetric(
                    name=item["name"],
                    value=float(item["value"]),
                    warning_threshold=float(item["warning_threshold"]),
                    refusal_threshold=float(item["refusal_threshold"]),
                    direction=item["direction"],
                    status=GateStatus(item["status"]),
                )
                for item in values["metrics"]
            ),
            reasons=tuple(values["reasons"]),
            threshold_version=values["threshold_version"],
            calibrated=bool(values["calibrated"]),
        )


class InferenceRefusedError(RuntimeError):
    """Raised when confirmatory inference is prohibited by a hard gate."""


def _minimum_metric(name: str, value: float, warning: float, refusal: float) -> GateMetric:
    if not np.isfinite(value) or value < refusal:
        status = GateStatus.REFUSE
    elif value < warning:
        status = GateStatus.WARNING
    else:
        status = GateStatus.PASS
    return GateMetric(name, value, warning, refusal, "minimum", status)


def _maximum_metric(name: str, value: float, warning: float, refusal: float) -> GateMetric:
    if not np.isfinite(value) or value > refusal:
        status = GateStatus.REFUSE
    elif value > warning:
        status = GateStatus.WARNING
    else:
        status = GateStatus.PASS
    return GateMetric(name, value, warning, refusal, "maximum", status)


def evaluate_path_gates(
    *,
    min_group_ess: float,
    target_ess_ratio: float,
    max_influence_share: float,
    max_weight_concentration: float,
    min_propensity_q01: float,
    max_calibration_error: float,
    max_balance: float,
    crossfit_instability: float,
    numerical_valid: bool,
    thresholds: DiagnosticThresholds,
) -> GateDecision:
    metrics = (
        _minimum_metric(
            "min_group_ess",
            min_group_ess,
            thresholds.min_group_ess_warning,
            thresholds.min_group_ess_refuse,
        ),
        _minimum_metric(
            "target_ess_ratio",
            target_ess_ratio,
            thresholds.min_target_ess_ratio_warning,
            thresholds.min_target_ess_ratio_refuse,
        ),
        _maximum_metric(
            "max_influence_share",
            max_influence_share,
            thresholds.max_influence_share_warning,
            thresholds.max_influence_share_refuse,
        ),
        _maximum_metric(
            "max_weight_concentration",
            max_weight_concentration,
            thresholds.max_weight_concentration_warning,
            thresholds.max_weight_concentration_refuse,
        ),
        _minimum_metric(
            "min_propensity_q01",
            min_propensity_q01,
            thresholds.min_propensity_q01_warning,
            thresholds.min_propensity_q01_refuse,
        ),
        _maximum_metric(
            "max_calibration_error",
            max_calibration_error,
            thresholds.max_calibration_error_warning,
            thresholds.max_calibration_error_refuse,
        ),
        _maximum_metric(
            "max_balance",
            max_balance,
            thresholds.max_balance_warning,
            thresholds.max_balance_refuse,
        ),
        _maximum_metric(
            "crossfit_instability",
            crossfit_instability,
            thresholds.max_crossfit_instability_warning,
            thresholds.max_crossfit_instability_refuse,
        ),
    )
    if not numerical_valid or any(metric.status is GateStatus.REFUSE for metric in metrics):
        status = GateStatus.REFUSE
    elif any(metric.status is GateStatus.WARNING for metric in metrics):
        status = GateStatus.WARNING
    elif thresholds.calibrated:
        status = GateStatus.PASS
    else:
        status = GateStatus.WARNING
    reasons = [
        f"{metric.name}={metric.value:.6g} ({metric.status.value})"
        for metric in metrics
        if metric.status is not GateStatus.PASS
    ]
    if not numerical_valid:
        reasons.insert(0, "numerical validity check failed")
    if not thresholds.calibrated:
        reasons.append("thresholds are provisional and cannot certify inference")
    return GateDecision(
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
        threshold_version=thresholds.version,
        calibrated=thresholds.calibrated,
    )


def evaluate_design_gates(
    *,
    min_group_ess: float,
    target_ess_ratio: float,
    max_weight_concentration: float,
    min_propensity_q01: float,
    max_calibration_error: float,
    max_balance: float,
    crossfit_instability: float,
    numerical_valid: bool,
    thresholds: DiagnosticThresholds,
) -> GateDecision:
    """Evaluate only (X, A)-measurable Stage 3 diagnostics.

    Influence concentration is intentionally absent: it depends on outcomes
    and therefore cannot be used while the Stage 4 design firewall is locked.
    """
    metrics = (
        _minimum_metric(
            "min_group_ess",
            min_group_ess,
            thresholds.min_group_ess_warning,
            thresholds.min_group_ess_refuse,
        ),
        _minimum_metric(
            "target_ess_ratio",
            target_ess_ratio,
            thresholds.min_target_ess_ratio_warning,
            thresholds.min_target_ess_ratio_refuse,
        ),
        _maximum_metric(
            "max_weight_concentration",
            max_weight_concentration,
            thresholds.max_weight_concentration_warning,
            thresholds.max_weight_concentration_refuse,
        ),
        _minimum_metric(
            "min_propensity_q01",
            min_propensity_q01,
            thresholds.min_propensity_q01_warning,
            thresholds.min_propensity_q01_refuse,
        ),
        _maximum_metric(
            "max_calibration_error",
            max_calibration_error,
            thresholds.max_calibration_error_warning,
            thresholds.max_calibration_error_refuse,
        ),
        _maximum_metric(
            "max_balance",
            max_balance,
            thresholds.max_balance_warning,
            thresholds.max_balance_refuse,
        ),
        _maximum_metric(
            "crossfit_instability",
            crossfit_instability,
            thresholds.max_crossfit_instability_warning,
            thresholds.max_crossfit_instability_refuse,
        ),
    )
    if not numerical_valid or any(metric.status is GateStatus.REFUSE for metric in metrics):
        status = GateStatus.REFUSE
    elif any(metric.status is GateStatus.WARNING for metric in metrics):
        status = GateStatus.WARNING
    elif thresholds.calibrated:
        status = GateStatus.PASS
    else:
        status = GateStatus.WARNING
    reasons = [
        f"{metric.name}={metric.value:.6g} ({metric.status.value})"
        for metric in metrics
        if metric.status is not GateStatus.PASS
    ]
    if not numerical_valid:
        reasons.insert(0, "numerical validity check failed")
    if not thresholds.calibrated:
        reasons.append("thresholds are provisional and cannot certify inference")
    return GateDecision(
        status=status,
        metrics=metrics,
        reasons=tuple(reasons),
        threshold_version=thresholds.version,
        calibrated=thresholds.calibrated,
    )
