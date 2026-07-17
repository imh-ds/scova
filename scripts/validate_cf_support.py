"""Evaluate a frozen candidate profile on the untouched held-out lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from calibrate_cf_support import _cell_gate, _passes, _verify_evidence

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum


def validate(
    protocol: CFValidationProtocol,
    campaign: dict[str, Any],
    candidate: CFSupportProfile,
) -> tuple[dict[str, Any], CFSupportProfile | None]:
    _verify_evidence(campaign)
    if campaign["lane"] != "validation" or not campaign["complete_frozen_lane"]:
        raise ValueError("Only the complete frozen held-out lane can promote a profile")
    if campaign["protocol_checksum"] != protocol.checksum:
        raise ValueError("Validation evidence uses a different protocol")
    if candidate.state != "candidate" or candidate.protocol_checksum != protocol.checksum:
        raise ValueError("Candidate profile is not bound to this protocol")
    thresholds = dict(candidate.thresholds)
    audits: list[dict[str, Any]] = []
    all_passed = True
    nonrefused = [record for record in campaign["records"] if not record["refused"]]
    for summary in campaign["summaries"]:
        cell_index = summary["cell_index"]
        cell = summary["cell"]
        records = [
            record
            for record in nonrefused
            if record["cell_index"] == cell_index and _passes(record, thresholds)
        ]
        if cell["support"] == "structural-failure":
            passed = summary["refusal_rate"] == 1.0
            audit = {"passed": passed, "structural_refusal_rate": summary["refusal_rate"]}
        else:
            passed, audit = _cell_gate(records, dict(protocol.metrics))
        all_passed &= passed
        audits.append({"cell_index": cell_index, "cell": cell, **audit})

    supported = [record for record in nonrefused if _passes(record, thresholds)]
    unstable = [record for record in nonrefused if not _passes(record, thresholds)]

    def bad_rate(records: list[dict[str, Any]]) -> float:
        if not records:
            return 0.0
        return float(
            np.mean(
                [
                    (not record["covered"])
                    or abs(record["estimate"] - record["truth"])
                    > 2 * record["standard_error"]
                    for record in records
                ]
            )
        )

    supported_bad = bad_rate(supported)
    unstable_bad = bad_rate(unstable)
    enrichment_ratio = (
        np.inf if supported_bad == 0 and unstable_bad > 0 else unstable_bad / supported_bad
        if supported_bad > 0
        else 0.0
    )
    enrichment_passed = bool(
        unstable
        and unstable_bad >= supported_bad + 0.10
        and enrichment_ratio >= 2.0
    )
    all_passed &= enrichment_passed
    result: dict[str, Any] = {
        "artifact_type": "scova-cf-support-validation",
        "schema_version": 1,
        "protocol_checksum": protocol.checksum,
        "campaign_evidence_checksum": campaign["evidence_checksum"],
        "candidate_profile_checksum": candidate.checksum,
        "all_validation_gates_passed": all_passed,
        "audit": audits,
        "unstable_enrichment": {
            "passed": enrichment_passed,
            "supported_count": len(supported),
            "unstable_count": len(unstable),
            "supported_bad_rate": supported_bad,
            "unstable_bad_rate": unstable_bad,
            "risk_ratio": None if not np.isfinite(enrichment_ratio) else enrichment_ratio,
        },
    }
    result["evidence_checksum"] = canonical_checksum(result)
    promoted = None
    if all_passed:
        promoted = CFSupportProfile(
            profile_id=candidate.profile_id.removesuffix("-candidate") + "-promoted",
            protocol_checksum=candidate.protocol_checksum,
            calibration_evidence_checksum=candidate.calibration_evidence_checksum,
            validation_evidence_checksum=result["evidence_checksum"],
            thresholds=candidate.thresholds,
            state="promoted",
        )
    return result, promoted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    args = parser.parse_args()
    result, profile = validate(
        CFValidationProtocol.load(args.spec),
        json.loads(args.validation_evidence.read_text(encoding="utf-8")),
        CFSupportProfile.from_dict(
            json.loads(args.candidate_profile.read_text(encoding="utf-8"))
        ),
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
