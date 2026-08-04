"""Versioned observational applicability matrices for SCOVA-CF."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from typing import Any, Mapping


class ApplicabilityClassification(str, Enum):
    IN_SCOPE = "in-scope"
    KNOWN_LIMITATION = "known-limitation"
    UNTESTED = "untested"


@dataclass(frozen=True, slots=True)
class ApplicabilityMatrix:
    matrix_id: str
    state: str
    mode: str
    source_protocol: str
    assignment: str
    nuisance_strategy: str
    group_counts: tuple[int, ...]
    maximum_covariate_count: int
    classifications: Mapping[ApplicabilityClassification, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True, slots=True)
class ApplicabilityAssessment:
    matrix_id: str
    classification: ApplicabilityClassification
    reason: str
    runtime_checkable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "classification": self.classification.value,
            "reason": self.reason,
            "runtime_checkable": self.runtime_checkable,
        }


def _matrix_from_dict(values: Mapping[str, Any]) -> ApplicabilityMatrix:
    required = {
        "matrix_id",
        "state",
        "mode",
        "source_protocol",
        "candidate_qualification_envelope",
        "classifications",
    }
    if set(values) != required:
        raise ValueError("Applicability matrix has unexpected or missing fields")
    envelope = values["candidate_qualification_envelope"]
    classes = values["classifications"]
    if set(classes) != {item.value for item in ApplicabilityClassification}:
        raise ValueError("Applicability matrix must declare every classification")
    if values["mode"] != "observational-causal" or values["state"] != "provisional":
        raise ValueError("Only a provisional observational applicability matrix is supported")
    if envelope.get("assignment") != "estimated" or envelope.get("nuisance_strategy") != "adaptive":
        raise ValueError("Observational applicability envelope requires estimated adaptive nuisances")
    group_counts = tuple(int(value) for value in envelope.get("n_groups", ()))
    maximum_covariate_count = int(envelope.get("maximum_covariate_count", 0))
    if group_counts != (2, 3) or maximum_covariate_count != 5:
        raise ValueError("Provisional observational envelope must cover only two/three groups and five covariates")
    normalized: dict[ApplicabilityClassification, tuple[Mapping[str, Any], ...]] = {}
    entry_ids: set[str] = set()
    for classification in ApplicabilityClassification:
        entries = classes[classification.value]
        if not isinstance(entries, list):
            raise ValueError("Applicability classification entries must be lists")
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("entry_id") or not entry.get("statement"):
                raise ValueError("Applicability entries require an id and statement")
            if entry["entry_id"] in entry_ids or not isinstance(entry.get("evidence"), dict):
                raise ValueError("Applicability entries require unique ids and evidence")
            entry_ids.add(str(entry["entry_id"]))
        normalized[classification] = tuple(dict(entry) for entry in entries)
    if normalized[ApplicabilityClassification.IN_SCOPE]:
        raise ValueError("A provisional observational matrix cannot claim an in-scope entry")
    return ApplicabilityMatrix(
        matrix_id=str(values["matrix_id"]),
        state=str(values["state"]),
        mode=str(values["mode"]),
        source_protocol=str(values["source_protocol"]),
        assignment=str(envelope["assignment"]),
        nuisance_strategy=str(envelope["nuisance_strategy"]),
        group_counts=group_counts,
        maximum_covariate_count=maximum_covariate_count,
        classifications=normalized,
    )


def observational_applicability_matrix() -> ApplicabilityMatrix:
    resource = files("scova.cf").joinpath("data").joinpath("applicability_matrices.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "matrices"} or payload["schema_version"] != 1:
        raise ValueError("Unsupported applicability-matrix schema")
    matrices = [_matrix_from_dict(value) for value in payload["matrices"]]
    if len(matrices) != 1 or len({matrix.matrix_id for matrix in matrices}) != len(matrices):
        raise ValueError("Applicability matrices must contain one uniquely identified matrix")
    return matrices[0]


def assess_observational_applicability(
    *, n_groups: int, n_covariates: int, nuisance_strategy: str
) -> ApplicabilityAssessment:
    """Describe the provisional matrix boundary without qualifying an analysis."""
    matrix = observational_applicability_matrix()
    if nuisance_strategy != matrix.nuisance_strategy:
        return ApplicabilityAssessment(
            matrix.matrix_id,
            ApplicabilityClassification.KNOWN_LIMITATION,
            "Non-adaptive observational nuisance strategies are contract-ineligible",
            True,
        )
    if n_covariates > matrix.maximum_covariate_count:
        return ApplicabilityAssessment(
            matrix.matrix_id,
            ApplicabilityClassification.UNTESTED,
            "Outside provisional observational predictor scope; more than five covariates are untested",
            True,
        )
    if n_groups not in matrix.group_counts:
        return ApplicabilityAssessment(
            matrix.matrix_id,
            ApplicabilityClassification.UNTESTED,
            "Outside the two- and three-group candidate qualification envelope",
            True,
        )
    return ApplicabilityAssessment(
        matrix.matrix_id,
        ApplicabilityClassification.UNTESTED,
        "Inside the provisional candidate envelope, but no observational profile is promoted",
        True,
    )
