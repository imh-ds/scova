"""Fail-closed sequencing checks for frozen SCOVA-CF campaign tiers."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum

Stage = Literal["external", "inference", "validation"]


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def _current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _valid_checksum(values: dict[str, Any], field: str) -> bool:
    supplied = values.get(field)
    return supplied == canonical_checksum({k: v for k, v in values.items() if k != field})


def prerequisite_reasons(
    stage: Stage,
    protocol: CFValidationProtocol,
    *,
    calibration_campaign: dict[str, Any],
    calibration_audit: dict[str, Any],
    candidate: dict[str, Any] | None,
    expected_commit: str,
    external: dict[str, Any] | None = None,
    inference: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    profile: CFSupportProfile | None = None
    if candidate is None:
        reasons.append(
            "candidate profile is missing because calibration did not promote a support policy"
        )
    else:
        try:
            profile = CFSupportProfile.from_dict(candidate)
        except (KeyError, TypeError, ValueError) as error:
            reasons.append(f"invalid candidate profile: {error}")
        else:
            if profile.state != "candidate" or profile.protocol_checksum != protocol.checksum:
                reasons.append("candidate is not frozen for this protocol")
    if calibration_campaign.get("protocol_checksum") != protocol.checksum:
        reasons.append("calibration campaign protocol mismatch")
    if calibration_campaign.get("git_commit") != expected_commit:
        reasons.append("calibration campaign commit mismatch")
    if not _valid_checksum(calibration_campaign, "evidence_checksum"):
        reasons.append("calibration campaign checksum mismatch")
    if not calibration_audit.get("all_calibration_gates_passed", False):
        reasons.append("calibration gates did not pass")
    if calibration_audit.get("calibration_evidence_checksum") != calibration_campaign.get(
        "evidence_checksum"
    ):
        reasons.append("calibration audit is not bound to its campaign")
    if (
        profile is not None
        and profile.calibration_evidence_checksum != calibration_campaign.get("evidence_checksum")
    ):
        reasons.append("candidate is not bound to the calibration campaign")

    if stage in {"inference", "validation"}:
        if external is None:
            reasons.append("external-agreement evidence is required")
        else:
            if not _valid_checksum(external, "evidence_checksum"):
                reasons.append("external evidence checksum mismatch")
            if external.get("protocol_checksum") != protocol.checksum:
                reasons.append("external evidence protocol mismatch")
            if external.get("git_commit") != expected_commit:
                reasons.append("external evidence commit mismatch")
            if not external.get("all_numerical_agreement_gates_passed", False):
                reasons.append("external numerical agreement did not pass")
    if stage == "validation":
        if inference is None:
            reasons.append("simultaneous-inference evidence is required")
        else:
            if not _valid_checksum(inference, "evidence_checksum"):
                reasons.append("inference evidence checksum mismatch")
            if inference.get("protocol_checksum") != protocol.checksum:
                reasons.append("inference evidence protocol mismatch")
            if inference.get("git_commit") != expected_commit:
                reasons.append("inference evidence commit mismatch")
            if not inference.get("all_inference_gates_passed", False):
                reasons.append("simultaneous-inference gates did not pass")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("external", "inference", "validation"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--calibration-campaign", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--external-evidence", type=Path)
    parser.add_argument("--inference-evidence", type=Path)
    args = parser.parse_args()
    reasons = prerequisite_reasons(
        args.stage,
        CFValidationProtocol.load(args.spec),
        calibration_campaign=_read(args.calibration_campaign),
        calibration_audit=_read(args.calibration_audit),
        candidate=(
            _read(args.candidate_profile) if args.candidate_profile.is_file() else None
        ),
        expected_commit=_current_commit(),
        external=None if args.external_evidence is None else _read(args.external_evidence),
        inference=None if args.inference_evidence is None else _read(args.inference_evidence),
    )
    if reasons:
        raise SystemExit("Campaign prerequisites failed:\n- " + "\n- ".join(reasons))


if __name__ == "__main__":
    main()
