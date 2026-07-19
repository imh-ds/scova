"""Auditable validation contracts for SCOVA-CF support-profile promotion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal


def canonical_checksum(values: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 identity of a JSON-compatible mapping."""
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SeedPartition:
    """Contiguous, non-overlapping simulation seed namespace."""

    start: int
    count: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.count < 1:
            raise ValueError("Seed partitions require a nonnegative start and positive count")

    @property
    def stop(self) -> int:
        return self.start + self.count

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "count": self.count}


@dataclass(frozen=True, slots=True)
class CFValidationProtocol:
    """Frozen reference-campaign specification with disjoint evidence lanes."""

    protocol_id: str
    reference_profile: Mapping[str, Any]
    factors: Mapping[str, tuple[Any, ...]]
    retained_cells: tuple[Mapping[str, Any], ...]
    pilot: SeedPartition
    calibration: SeedPartition
    validation: SeedPartition
    learners: tuple[str, ...]
    metrics: Mapping[str, float]
    software: Mapping[str, str]
    dataset_checksums: Mapping[str, str] | None = None
    dependency_lock_checksum: str | None = None
    design_selection: Mapping[str, Any] | None = None
    plasmode_cells: tuple[Mapping[str, Any], ...] = ()
    inference_cells: tuple[Mapping[str, Any], ...] = ()
    external_cells: tuple[Mapping[str, Any], ...] = ()
    external: SeedPartition | None = None
    inference: SeedPartition | None = None
    calibration_fit_fraction: float = 0.60
    threshold_quantiles: Mapping[str, tuple[float, ...]] | None = None
    calibration_screening: Mapping[str, float] | None = None
    calibration_candidate_retention_fraction: float = 1.0
    calibration_source: Mapping[str, str] | None = None
    frozen: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must not be empty")
        if not self.factors or any(not values for values in self.factors.values()):
            raise ValueError("Every campaign factor must contain at least one level")
        if not self.retained_cells:
            raise ValueError("The protocol must freeze at least one retained design cell")
        if self.schema_version >= 2:
            if not self.frozen:
                raise ValueError("Version-2 validation protocols must be frozen")
            if len(self.retained_cells) != 48:
                raise ValueError("The version-2 protocol requires 48 simulation cells")
            if len(self.plasmode_cells) != 12:
                raise ValueError("The version-2 protocol requires 12 plasmode cells")
            if len(self.inference_cells) != 6:
                raise ValueError("The version-2 protocol requires six inference cells")
            if len(self.external_cells) != 8:
                raise ValueError("The version-2 protocol requires eight external cells")
            if self.external is None or self.inference is None:
                raise ValueError("Version-2 protocols require external and inference seeds")
            if self.external.count != 50:
                raise ValueError("Each external comparison cell requires 50 replications")
            if self.inference.count != 2000:
                raise ValueError("Each focused inference cell requires 2,000 replications")
            if set(self.dataset_checksums or {}) != {"diabetes", "breast-cancer"}:
                raise ValueError("Version-2 protocols require both plasmode source checksums")
            if not self.dependency_lock_checksum:
                raise ValueError("Version-2 protocols require a dependency-lock checksum")
            if not self.design_selection:
                raise ValueError("Version-2 protocols require pairwise-design provenance")
        if not 0 < self.calibration_fit_fraction < 1:
            raise ValueError("calibration_fit_fraction must lie in (0, 1)")
        if not 0 < self.calibration_candidate_retention_fraction <= 1:
            raise ValueError("calibration_candidate_retention_fraction must lie in (0, 1]")
        for cell in self.retained_cells:
            if set(cell) != set(self.factors):
                raise ValueError("Every retained cell must specify every campaign factor")
            invalid = {
                name: value
                for name, value in cell.items()
                if value not in self.factors[name]
            }
            if invalid:
                raise ValueError(f"Retained cell contains undeclared factor levels: {invalid}")
        partitions = tuple(
            partition
            for partition in (
                self.pilot,
                self.calibration,
                self.validation,
                self.external,
                self.inference,
            )
            if partition is not None
        )
        maximum_cells = max(
            len(self.retained_cells) + len(self.plasmode_cells),
            len(self.external_cells),
            len(self.inference_cells),
        )
        intervals = sorted(
            (part.start, part.start + part.count * maximum_cells)
            for part in partitions
        )
        if any(
            left[1] > right[0]
            for left, right in pairwise(intervals)
        ):
            raise ValueError("Pilot, calibration, and validation seed partitions must be disjoint")
        if self.calibration.count < 1000:
            raise ValueError("The frozen calibration lane requires at least 1,000 replications")
        if self.validation.count < 2000:
            raise ValueError("The held-out validation lane requires at least 2,000 replications")
        required_metrics = {
            "confidence_level",
            "type_i_error",
            "monte_carlo_standard_error_multiplier",
            "maximum_standardized_bias",
            "minimum_se_ratio",
            "maximum_se_ratio",
            "strong_support_minimum_expected_arm_count",
        }
        missing = required_metrics.difference(self.metrics)
        if missing:
            raise ValueError(f"Validation protocol is missing metrics: {sorted(missing)}")
        if self.calibration_screening is not None:
            missing = required_metrics.difference(self.calibration_screening)
            if missing:
                raise ValueError(
                    "Calibration screening is missing metrics: "
                    f"{sorted(missing)}"
                )
        if self.calibration_source is not None:
            required_source = {
                "protocol_id",
                "protocol_checksum",
                "evidence_checksum",
                "git_commit",
            }
            missing = required_source.difference(self.calibration_source)
            if missing:
                raise ValueError(
                    "Calibration source is missing fields: " f"{sorted(missing)}"
                )
            if any(
                not self.calibration_source[name]
                for name in ("protocol_id", "protocol_checksum", "evidence_checksum", "git_commit")
            ):
                raise ValueError("Calibration source values must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frozen": self.frozen,
            "protocol_id": self.protocol_id,
            "reference_profile": dict(self.reference_profile),
            "factors": {name: list(values) for name, values in self.factors.items()},
            "retained_cells": [dict(cell) for cell in self.retained_cells],
            "plasmode_cells": [dict(cell) for cell in self.plasmode_cells],
            "inference_cells": [dict(cell) for cell in self.inference_cells],
            "external_cells": [dict(cell) for cell in self.external_cells],
            "seed_partitions": {
                "pilot": self.pilot.to_dict(),
                "calibration": self.calibration.to_dict(),
                "validation": self.validation.to_dict(),
                **({} if self.external is None else {"external": self.external.to_dict()}),
                **({} if self.inference is None else {"inference": self.inference.to_dict()}),
            },
            "calibration_fit_fraction": self.calibration_fit_fraction,
            **(
                {}
                if self.calibration_candidate_retention_fraction == 1.0
                else {
                    "calibration_candidate_retention_fraction": (
                        self.calibration_candidate_retention_fraction
                    )
                }
            ),
            **(
                {}
                if self.threshold_quantiles is None
                else {
                    "threshold_quantiles": {
                        name: list(levels)
                        for name, levels in self.threshold_quantiles.items()
                    }
                }
            ),
            **(
                {}
                if self.calibration_screening is None
                else {"calibration_screening": dict(self.calibration_screening)}
            ),
            **(
                {}
                if self.calibration_source is None
                else {"calibration_source": dict(self.calibration_source)}
            ),
            "learners": list(self.learners),
            "metrics": dict(self.metrics),
            "software": dict(self.software),
            "dataset_checksums": dict(self.dataset_checksums or {}),
            "dependency_lock_checksum": self.dependency_lock_checksum,
            "design_selection": dict(self.design_selection or {}),
        }

    @property
    def checksum(self) -> str:
        return canonical_checksum(self.to_dict())

    @property
    def calibration_gate_metrics(self) -> Mapping[str, float]:
        """Return v4 screening metrics or the legacy common metrics."""
        return self.metrics if self.calibration_screening is None else self.calibration_screening

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CFValidationProtocol:
        partitions = values["seed_partitions"]
        return cls(
            schema_version=int(values.get("schema_version", 1)),
            frozen=bool(values.get("frozen", False)),
            protocol_id=str(values["protocol_id"]),
            reference_profile=dict(values["reference_profile"]),
            factors={
                str(name): tuple(levels) for name, levels in values["factors"].items()
            },
            retained_cells=tuple(dict(cell) for cell in values["retained_cells"]),
            plasmode_cells=tuple(dict(cell) for cell in values.get("plasmode_cells", ())),
            inference_cells=tuple(dict(cell) for cell in values.get("inference_cells", ())),
            external_cells=tuple(dict(cell) for cell in values.get("external_cells", ())),
            pilot=SeedPartition(**partitions["pilot"]),
            calibration=SeedPartition(**partitions["calibration"]),
            validation=SeedPartition(**partitions["validation"]),
            external=(
                None
                if "external" not in partitions
                else SeedPartition(**partitions["external"])
            ),
            inference=(
                None
                if "inference" not in partitions
                else SeedPartition(**partitions["inference"])
            ),
            calibration_fit_fraction=float(values.get("calibration_fit_fraction", 0.60)),
            calibration_candidate_retention_fraction=float(
                values.get("calibration_candidate_retention_fraction", 1.0)
            ),
            threshold_quantiles=(
                None
                if "threshold_quantiles" not in values
                else {
                    str(name): tuple(float(level) for level in levels)
                    for name, levels in values["threshold_quantiles"].items()
                }
            ),
            calibration_screening=(
                None
                if values.get("calibration_screening") is None
                else {
                    str(name): float(value)
                    for name, value in values["calibration_screening"].items()
                }
            ),
            calibration_source=(
                None
                if values.get("calibration_source") is None
                else {
                    str(name): str(value)
                    for name, value in values["calibration_source"].items()
                }
            ),
            learners=tuple(str(value) for value in values["learners"]),
            metrics={str(name): float(value) for name, value in values["metrics"].items()},
            software={str(name): str(value) for name, value in values["software"].items()},
            dataset_checksums={
                str(name): str(value)
                for name, value in values.get("dataset_checksums", {}).items()
            },
            dependency_lock_checksum=(
                None
                if values.get("dependency_lock_checksum") is None
                else str(values["dependency_lock_checksum"])
            ),
            design_selection=dict(values.get("design_selection", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> CFValidationProtocol:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class CFSupportProfile:
    """Immutable candidate or promoted support profile with evidence identities."""

    profile_id: str
    protocol_checksum: str
    calibration_evidence_checksum: str
    validation_evidence_checksum: str | None
    thresholds: Mapping[str, float]
    compatibility: Mapping[str, Any] | None = None
    state: Literal["candidate", "promoted"] = "candidate"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.profile_id or not self.protocol_checksum:
            raise ValueError("Support profiles require identifiers and a protocol checksum")
        if not self.calibration_evidence_checksum:
            raise ValueError("Support profiles require calibration evidence")
        if self.state == "promoted" and not self.validation_evidence_checksum:
            raise ValueError("A promoted support profile requires held-out validation evidence")
        if self.state == "promoted" and not self.compatibility:
            raise ValueError("A promoted support profile requires an explicit compatibility lock")
        if not self.thresholds or any(
            not isinstance(value, (int, float)) for value in self.thresholds.values()
        ):
            raise ValueError("Support-profile thresholds must be nonempty numeric values")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "protocol_checksum": self.protocol_checksum,
            "calibration_evidence_checksum": self.calibration_evidence_checksum,
            "validation_evidence_checksum": self.validation_evidence_checksum,
            "thresholds": dict(self.thresholds),
            "compatibility": dict(self.compatibility or {}),
            "state": self.state,
        }

    @property
    def checksum(self) -> str:
        return canonical_checksum(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "profile_checksum": self.checksum}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CFSupportProfile:
        profile = cls(
            schema_version=int(values.get("schema_version", 1)),
            profile_id=str(values["profile_id"]),
            protocol_checksum=str(values["protocol_checksum"]),
            calibration_evidence_checksum=str(values["calibration_evidence_checksum"]),
            validation_evidence_checksum=(
                None
                if values.get("validation_evidence_checksum") is None
                else str(values["validation_evidence_checksum"])
            ),
            thresholds={
                str(name): float(value) for name, value in values["thresholds"].items()
            },
            compatibility=(
                None
                if not values.get("compatibility")
                else dict(values["compatibility"])
            ),
            state=str(values["state"]),  # type: ignore[arg-type]
        )
        if values.get("profile_checksum") != profile.checksum:
            raise ValueError("Support-profile checksum does not match its payload")
        return profile
