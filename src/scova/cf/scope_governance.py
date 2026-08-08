"""Fail-closed governance for prospective SCOVA-CF scope changes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

from .validation import CFValidationProtocol, canonical_checksum


DECISION_REGISTRY_SCHEMA_VERSION = 1
DECISION_MANIFEST_SCHEMA_VERSION = 1
_PATHS = {"improve-method", "exclude-with-observable-rule", "retain-limitation"}
_STATUSES = {"open", "resolved"}

QUALIFICATION_REQUIRED_DECISION_IDS = (
    "v11-linear-smooth-nonlinear-failure",
    "v11-linear-threshold-failure",
    "v11-adaptive-cell-2-limitation",
    "v11-density-boundary-defect",
    "predictor-count-above-five-gap",
    "v11-limited-small-sample-status",
    "unpinned-local-ablation-discrepancy",
)


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    decision_id: str
    status: str
    path: str
    checksum: str
    payload: Mapping[str, Any]


def _record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in record.items() if name != "record_checksum"}


def _validate_approval(value: Any, role: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} approval must be a record")
    required = {"name", "approved_at", "review_reference"}
    if set(value) != required or not all(isinstance(value[name], str) and value[name] for name in required):
        raise ValueError(f"{role} approval requires name, approved_at, and review_reference")


def _validate_exclusion(rule: Any) -> None:
    if not isinstance(rule, Mapping):
        raise ValueError("An exclusion decision requires an exclusion rule")
    required = {"description", "runtime_checkable", "pre_outcome", "independent_of_simulation_outcome"}
    if set(rule) != required or not all(isinstance(rule[name], bool) for name in required - {"description"}):
        raise ValueError("Exclusion rule has an invalid schema")
    if not isinstance(rule["description"], str) or not rule["description"]:
        raise ValueError("Exclusion rule needs a description")
    if not all(rule[name] for name in ("runtime_checkable", "pre_outcome", "independent_of_simulation_outcome")):
        raise ValueError("Scope exclusions must be runtime-checkable, pre-outcome, and outcome-independent")
    forbidden = ("outcome surface", "confounding truth", "observed bias", "simulation outcome")
    if any(term in rule["description"].lower() for term in forbidden):
        raise ValueError("Scope exclusions may not encode unobservable DGP truth or observed performance")


def validate_record(record: Mapping[str, Any]) -> ScopeDecision:
    required = {
        "decision_id", "status", "issue_type", "affected_protocols", "affected_artifacts",
        "evidence_ids", "competing_explanations", "uncertainty", "path", "rationale",
        "prior_evidence_consequences", "change_consequences", "exclusion_rule", "approvals",
        "record_checksum",
    }
    if set(record) != required:
        raise ValueError("Scope decision has unexpected or missing fields")
    if not isinstance(record["decision_id"], str) or not record["decision_id"]:
        raise ValueError("Scope decision requires a stable id")
    if record["status"] not in _STATUSES or record["path"] not in _PATHS:
        raise ValueError("Scope decision has an invalid status or path")
    for name in ("affected_protocols", "affected_artifacts", "evidence_ids", "competing_explanations"):
        if not isinstance(record[name], list) or not all(isinstance(item, str) and item for item in record[name]):
            raise ValueError(f"Scope decision {name} must be a nonempty string list")
    for name in ("issue_type", "uncertainty", "rationale", "prior_evidence_consequences"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError(f"Scope decision {name} must be a nonempty string")
    changes = record["change_consequences"]
    if set(changes) != {"new_protocol_freeze", "new_contract_version", "new_matrix_version"} or not all(
        isinstance(value, bool) for value in changes.values()
    ):
        raise ValueError("Scope decision must declare freeze consequences")
    if record["path"] == "exclude-with-observable-rule":
        _validate_exclusion(record["exclusion_rule"])
    elif record["exclusion_rule"] is not None:
        raise ValueError("Only an exclusion decision may define an exclusion rule")
    approvals = record["approvals"]
    if set(approvals) != {"owner", "independent_reviewer"}:
        raise ValueError("Scope decision approvals must name owner and independent reviewer")
    if record["status"] == "resolved":
        # Scope decisions govern development work and can be resolved by the
        # named owner. Independent review is a later promotion safeguard.
        _validate_approval(approvals["owner"], "owner")
        reviewer = approvals["independent_reviewer"]
        if reviewer is not None:
            _validate_approval(reviewer, "independent reviewer")
            if approvals["owner"]["name"] == reviewer["name"]:
                raise ValueError("Owner and independent reviewer must be different people")
    elif approvals["owner"] is not None or approvals["independent_reviewer"] is not None:
        raise ValueError("Open scope decisions cannot carry partial approvals")
    payload = _record_payload(record)
    checksum = canonical_checksum(payload)
    if record["record_checksum"] != checksum:
        raise ValueError("Scope decision checksum does not match its payload")
    return ScopeDecision(record["decision_id"], record["status"], record["path"], checksum, payload)


def validate_registry(values: Mapping[str, Any]) -> dict[str, ScopeDecision]:
    if set(values) != {"schema_version", "records"} or values["schema_version"] != DECISION_REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported scope-decision registry")
    if not isinstance(values["records"], list):
        raise ValueError("Scope-decision registry records must be a list")
    decisions = [validate_record(record) for record in values["records"]]
    ids = [decision.decision_id for decision in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("Scope-decision ids must be unique")
    return {decision.decision_id: decision for decision in decisions}


def scope_decision_registry() -> dict[str, ScopeDecision]:
    resource = files("scova.cf").joinpath("data/scope_decisions.json")
    return validate_registry(json.loads(resource.read_text(encoding="utf-8")))


def applicability_matrix_checksum() -> str:
    resource = files("scova.cf").joinpath("data/applicability_matrices.json")
    return canonical_checksum(json.loads(resource.read_text(encoding="utf-8")))


def build_qualification_manifest(
    protocol: CFValidationProtocol, *, contract_version: str, matrix_id: str,
    required_decision_ids: tuple[str, ...], registry: Mapping[str, ScopeDecision] | None = None,
) -> dict[str, Any]:
    registry = scope_decision_registry() if registry is None else dict(registry)
    missing = set(required_decision_ids).difference(registry)
    if missing:
        raise ValueError(f"Qualification manifest references missing decisions: {sorted(missing)}")
    payload = {
        "schema_version": DECISION_MANIFEST_SCHEMA_VERSION,
        "program_type": "qualification",
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "design_checksum": str((protocol.design_selection or {}).get("design_checksum", "")),
        "contract_version": contract_version,
        "matrix_id": matrix_id,
        "matrix_checksum": applicability_matrix_checksum(),
        "dependency_lock_checksum": protocol.dependency_lock_checksum,
        "required_decisions": [
            {"decision_id": decision_id, "record_checksum": registry[decision_id].checksum}
            for decision_id in required_decision_ids
        ],
    }
    return {**payload, "manifest_checksum": canonical_checksum(payload)}


def validate_manifest(
    manifest: Mapping[str, Any], protocol: CFValidationProtocol, *, contract_version: str,
    matrix_id: str, registry: Mapping[str, ScopeDecision] | None = None,
) -> None:
    required = {
        "schema_version", "program_type", "protocol_id", "protocol_checksum", "design_checksum",
        "contract_version", "matrix_id", "matrix_checksum", "dependency_lock_checksum",
        "required_decisions", "manifest_checksum",
    }
    if set(manifest) != required or manifest["schema_version"] != DECISION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported qualification decision manifest")
    payload = {name: value for name, value in manifest.items() if name != "manifest_checksum"}
    if manifest["manifest_checksum"] != canonical_checksum(payload):
        raise ValueError("Qualification decision-manifest checksum does not match its payload")
    expected = {
        "program_type": "qualification", "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "design_checksum": str((protocol.design_selection or {}).get("design_checksum", "")),
        "contract_version": contract_version, "matrix_id": matrix_id,
        "matrix_checksum": applicability_matrix_checksum(),
        "dependency_lock_checksum": protocol.dependency_lock_checksum,
    }
    for name, value in expected.items():
        if manifest[name] != value:
            raise ValueError(f"Qualification decision manifest {name} does not match the frozen context")
    registry = scope_decision_registry() if registry is None else dict(registry)
    decisions = manifest["required_decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Qualification decision manifest requires at least one decision")
    ids = [entry.get("decision_id") for entry in decisions if isinstance(entry, Mapping)]
    if len(ids) != len(decisions) or len(ids) != len(set(ids)):
        raise ValueError("Qualification decision manifest has invalid decision ids")
    for entry in decisions:
        decision = registry.get(entry["decision_id"])
        if decision is None or entry.get("record_checksum") != decision.checksum:
            raise ValueError("Qualification decision manifest is not bound to its decision record")
        if decision.status != "resolved":
            raise ValueError(f"Qualification dispatch blocked by unresolved decision {decision.decision_id}")
        evidence = decision.payload["evidence_ids"]
        if all(item.startswith("methods:") for item in evidence):
            raise ValueError("Methods-study evidence cannot alone resolve a qualification decision")
