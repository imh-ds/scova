"""Evaluate a frozen SCOVA-CF profile once on untouched held-out evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from calibrate_cf_support import (
    _cell_gate,
    _passes,
    _profile_eligible,
    _structural,
    _unstable_enrichment,
    _verify_evidence,
    read_json,
)

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum


def _candidate_matches_protocol(
    protocol: CFValidationProtocol, candidate: CFSupportProfile
) -> bool:
    """Accept either a native candidate or the exact candidate frozen by the protocol."""
    if candidate.protocol_checksum == protocol.checksum:
        return True
    source = protocol.candidate_source
    return bool(
        source
        and candidate.protocol_checksum == source.get("protocol_checksum")
        and candidate.checksum == source.get("profile_checksum")
    )


def _external_matches_protocol(
    protocol: CFValidationProtocol, evidence: dict[str, Any]
) -> bool:
    """Accept current-protocol evidence or the exact frozen external source."""
    if evidence.get("protocol_checksum") == protocol.checksum:
        return True
    source = protocol.external_source
    return bool(
        source
        and evidence.get("protocol_checksum") == source.get("protocol_checksum")
        and evidence.get("evidence_checksum") == source.get("evidence_checksum")
        and evidence.get("git_commit") == source.get("git_commit")
    )


def _inference_matches_protocol(
    protocol: CFValidationProtocol, evidence: dict[str, Any]
) -> bool:
    if evidence.get("protocol_checksum") == protocol.checksum:
        return True
    source = protocol.inference_source
    return bool(
        source
        and evidence.get("protocol_checksum") == source.get("protocol_checksum")
        and evidence.get("evidence_checksum") == source.get("evidence_checksum")
        and evidence.get("git_commit") == source.get("git_commit")
    )


def validate(
    protocol: CFValidationProtocol,
    campaign: dict[str, Any],
    candidate: CFSupportProfile,
    *,
    inference_evidence: dict[str, Any] | None = None,
    external_evidence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], CFSupportProfile | None]:
    _verify_evidence(campaign)
    if campaign["lane"] != "validation" or not campaign["complete_frozen_lane"]:
        raise ValueError("Only the complete frozen held-out lane can promote a profile")
    if campaign["protocol_checksum"] != protocol.checksum:
        raise ValueError("Validation evidence uses a different protocol")
    if candidate.state != "candidate" or not _candidate_matches_protocol(protocol, candidate):
        raise ValueError("Candidate profile is not bound to this protocol")
    if campaign.get("candidate_profile_checksum") != candidate.checksum:
        raise ValueError("Held-out evidence was not generated under this candidate profile")
    thresholds = dict(candidate.thresholds)
    records = campaign["records"]
    execution_failure_count = sum(
        record.get("status_code") == "execution-error" for record in records
    )
    usable = [record for record in records if not record["refused"]]
    audits: list[dict[str, Any]] = []
    all_passed = execution_failure_count == 0
    for cell_index in sorted({int(record["cell_index"]) for record in records}):
        all_cell = [record for record in records if record["cell_index"] == cell_index]
        cell = all_cell[0]["cell"]
        kind = all_cell[0]["cell_kind"]
        if _structural(cell):
            passed = all(record["refused"] for record in all_cell)
            audit = {
                "passed": passed,
                "structural_refusal_rate": float(np.mean([r["refused"] for r in all_cell])),
            }
        elif not _profile_eligible(protocol, cell, kind):
            passed = True
            audit = {
                "passed": True,
                "reason": "outside-promotable-profile-scope",
                "supported_replications": 0,
            }
        else:
            supported = [
                record
                for record in usable
                if record["cell_index"] == cell_index and _passes(record, thresholds)
            ]
            passed, audit = _cell_gate(supported, protocol.metrics)
            if not supported:
                passed = True
                audit = {"passed": True, "reason": "unstable-cell-no-supported-results"}
        all_passed &= passed
        audits.append({"cell_index": cell_index, "cell_kind": kind, "cell": cell, **audit})

    strong_cells = [
        audit
        for audit in audits
        if _profile_eligible(protocol, audit["cell"], audit["cell_kind"])
        and not _structural(audit["cell"])
    ]
    useful_cells = [
        audit
        for audit in strong_cells
        if int(audit.get("supported_replications", 0))
        / protocol.validation.count
        >= float(protocol.metrics["minimum_strong_replication_pass_fraction"])
    ]
    usefulness_passed = bool(
        strong_cells
        and len(useful_cells) / len(strong_cells)
        >= float(protocol.metrics["minimum_strong_cell_pass_fraction"])
    )
    all_passed &= usefulness_passed
    enrichment = _unstable_enrichment(usable, thresholds, protocol.metrics)
    enrichment_passed = bool(enrichment["passed"])
    all_passed &= enrichment_passed
    inference_passed = bool(
        inference_evidence
        and inference_evidence.get("all_inference_gates_passed", False)
        and _inference_matches_protocol(protocol, inference_evidence)
    )
    external_passed = bool(
        external_evidence
        and external_evidence.get("all_numerical_agreement_gates_passed", False)
        and _external_matches_protocol(protocol, external_evidence)
    )
    all_passed &= inference_passed and external_passed
    result: dict[str, Any] = {
        "artifact_type": "scova-cf-support-validation",
        "schema_version": 2,
        "protocol_checksum": protocol.checksum,
        "campaign_evidence_checksum": campaign["evidence_checksum"],
        "candidate_profile_checksum": candidate.checksum,
        "inference_evidence_checksum": (
            None if inference_evidence is None else inference_evidence.get("evidence_checksum")
        ),
        "external_evidence_checksum": (
            None if external_evidence is None else external_evidence.get("evidence_checksum")
        ),
        "all_validation_gates_passed": all_passed,
        "execution_failure_count": execution_failure_count,
        "usefulness": {
            "passed": usefulness_passed,
            "strong_cell_count": len(strong_cells),
            "useful_strong_cell_count": len(useful_cells),
        },
        "unstable_enrichment": enrichment,
        "inference_gate_passed": inference_passed,
        "external_gate_passed": external_passed,
        "audit": audits,
    }
    result["evidence_checksum"] = canonical_checksum(result)
    promoted = None
    if all_passed:
        promoted = CFSupportProfile(
            profile_id=protocol.protocol_id + "-promoted",
            protocol_checksum=protocol.checksum,
            calibration_evidence_checksum=candidate.calibration_evidence_checksum,
            validation_evidence_checksum=result["evidence_checksum"],
            thresholds=candidate.thresholds,
            compatibility=candidate.compatibility,
            state="promoted",
        )
    return result, promoted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--inference-evidence", type=Path, required=True)
    parser.add_argument("--external-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    args = parser.parse_args()
    result, profile = validate(
        CFValidationProtocol.load(args.spec),
        read_json(args.validation_evidence),
        CFSupportProfile.from_dict(read_json(args.candidate_profile)),
        inference_evidence=read_json(args.inference_evidence),
        external_evidence=read_json(args.external_evidence),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    if profile is None:
        raise SystemExit("Held-out validation failed; no promoted profile was written")
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
