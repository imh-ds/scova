from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.cf_qualification_program import qualification_protocol
from benchmarks.cf_reference_campaign import simulate_reference_cell
from scova.cf import (
    QUALIFICATION_REQUIRED_DECISION_IDS,
    CFValidationProtocol,
    applicability_matrix_checksum,
    build_qualification_manifest,
    canonical_checksum,
    scope_decision_registry,
    validate_manifest,
    validate_record,
    validate_registry,
)
from scripts.render_cf_scope_decisions import render


def _record(*, path: str = "retain-limitation", status: str = "resolved") -> dict[str, object]:
    record: dict[str, object] = {
        "decision_id": "test-decision",
        "status": status,
        "issue_type": "test",
        "affected_protocols": ["test-protocol"],
        "affected_artifacts": ["test-artifact"],
        "evidence_ids": ["campaign:123"],
        "competing_explanations": ["one", "two"],
        "uncertainty": "Uncertainty is recorded.",
        "path": path,
        "rationale": "Prospective rationale.",
        "prior_evidence_consequences": "No historical relabeling.",
        "change_consequences": {
            "new_protocol_freeze": True,
            "new_contract_version": False,
            "new_matrix_version": False,
        },
        "exclusion_rule": None,
        "approvals": {
            "owner": {
                "name": "Owner",
                "approved_at": "2026-08-03T00:00:00Z",
                "review_reference": "review:1",
            },
            "independent_reviewer": {
                "name": "Reviewer",
                "approved_at": "2026-08-03T00:01:00Z",
                "review_reference": "review:2",
            },
        },
    }
    if path == "exclude-with-observable-rule":
        record["exclusion_rule"] = {
            "description": "Declared covariate count exceeds five.",
            "runtime_checkable": True,
            "pre_outcome": True,
            "independent_of_simulation_outcome": True,
        }
    payload = dict(record)
    record["record_checksum"] = canonical_checksum(payload)
    return record


def test_scope_record_requires_checksum_and_owner_approval() -> None:
    record = _record()
    assert validate_record(record).status == "resolved"
    tampered = deepcopy(record)
    tampered["rationale"] = "Changed after review."
    with pytest.raises(ValueError, match="checksum"):
        validate_record(tampered)
    same_reviewer = deepcopy(record)
    same_reviewer["approvals"]["independent_reviewer"]["name"] = "Owner"  # type: ignore[index]
    same_reviewer["record_checksum"] = canonical_checksum(
        {key: value for key, value in same_reviewer.items() if key != "record_checksum"}
    )
    with pytest.raises(ValueError, match="different people"):
        validate_record(same_reviewer)
    owner_only = deepcopy(record)
    owner_only["approvals"]["independent_reviewer"] = None  # type: ignore[index]
    owner_only["record_checksum"] = canonical_checksum(
        {key: value for key, value in owner_only.items() if key != "record_checksum"}
    )
    assert validate_record(owner_only).status == "resolved"
    incomplete = deepcopy(owner_only)
    incomplete["approvals"]["owner"] = None  # type: ignore[index]
    incomplete["record_checksum"] = canonical_checksum(
        {key: value for key, value in incomplete.items() if key != "record_checksum"}
    )
    with pytest.raises(ValueError, match="approval"):
        validate_record(incomplete)


def test_scope_exclusions_must_be_observable_before_outcomes() -> None:
    record = _record(path="exclude-with-observable-rule")
    assert validate_record(record).path == "exclude-with-observable-rule"
    invalid = deepcopy(record)
    invalid["exclusion_rule"]["description"] = "Observed bias exceeds the gate."  # type: ignore[index]
    invalid["record_checksum"] = canonical_checksum(
        {key: value for key, value in invalid.items() if key != "record_checksum"}
    )
    with pytest.raises(ValueError, match="unobservable"):
        validate_record(invalid)


def test_registry_has_unique_known_blockers_and_renders_for_review() -> None:
    registry = scope_decision_registry()
    assert set(QUALIFICATION_REQUIRED_DECISION_IDS).issubset(registry)
    assert "v11-downstream-evidence-prerequisites" in registry
    raw = {
        "schema_version": 1,
        "records": [
            dict(item.payload, record_checksum=item.checksum) for item in registry.values()
        ],
    }
    assert "v11-limited-small-sample-status" in render(raw)
    duplicate = deepcopy(raw)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_registry(duplicate)


def test_current_qualification_manifest_is_context_bound_after_owner_resolutions() -> None:
    protocol = qualification_protocol()
    manifest = build_qualification_manifest(
        protocol,
        contract_version="1.0.0",
        matrix_id="cf-observational-provisional-v1",
        required_decision_ids=QUALIFICATION_REQUIRED_DECISION_IDS,
    )
    assert manifest["matrix_checksum"] == applicability_matrix_checksum()
    assert {entry["decision_id"] for entry in manifest["required_decisions"]} == set(
        QUALIFICATION_REQUIRED_DECISION_IDS
    )
    validate_manifest(
        manifest,
        protocol,
        contract_version="1.0.0",
        matrix_id="cf-observational-provisional-v1",
    )
    changed = dict(manifest, contract_version="2.0.0")
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest(
            changed,
            protocol,
            contract_version="1.0.0",
            matrix_id="cf-observational-provisional-v1",
        )


def test_methods_evidence_cannot_alone_resolve_a_qualification_manifest() -> None:
    record = _record()
    record["evidence_ids"] = ["methods:factorial-v1"]
    record["record_checksum"] = canonical_checksum(
        {key: value for key, value in record.items() if key != "record_checksum"}
    )
    validated = validate_record(record)
    registry = {validated.decision_id: validated}
    protocol = qualification_protocol()
    manifest = build_qualification_manifest(
        protocol,
        contract_version="1.0.0",
        matrix_id="cf-observational-provisional-v1",
        required_decision_ids=(validated.decision_id,),
        registry=registry,
    )
    with pytest.raises(ValueError, match="Methods-study evidence"):
        validate_manifest(
            manifest,
            protocol,
            contract_version="1.0.0",
            matrix_id="cf-observational-provisional-v1",
            registry=registry,
        )


def test_v11_limited_status_is_reproducible_rare_arm_feasibility_not_an_execution_error() -> None:
    protocol = CFValidationProtocol.load("benchmarks/specs/cf_reference_v11.json")
    cell = protocol.retained_cells[25]
    assert cell["n_groups"] == 3 and cell["allocation"] == "rare"
    for repetition in (534, 817, 865, 911, 1443, 1715, 1972, 1984):
        seed = protocol.calibration.start + 25 * protocol.calibration.count + repetition
        counts = simulate_reference_cell(cell, seed=seed).data["group"].value_counts()
        assert int(counts.min()) < 3
