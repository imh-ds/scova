"""Fail closed until every frozen SCOVA-CF v2 promotion artifact agrees."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum


def _read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def _checksum_valid(values: dict[str, Any], field: str) -> bool:
    supplied = values.get(field)
    payload = {name: value for name, value in values.items() if name != field}
    return supplied == canonical_checksum(payload)


def blocking_reasons(
    root: Path,
    spec: Path,
    *,
    packaged_manifest: Path | None = None,
) -> list[str]:
    protocol = CFValidationProtocol.load(spec)
    paths = {
        "calibration campaign": root / "cf-reference-calibration-campaign.json.gz",
        "calibration": root / "cf-reference-calibration.json",
        "validation campaign": root / "cf-reference-validation-campaign.json.gz",
        "validation": root / "cf-reference-validation.json",
        "inference": root / "cf-reference-inference.json",
        "external": root / "cf-reference-external-agreement.json",
        "profile": root / "cf-reference-support-profile.json",
    }
    report = root / "cf-reference-validation-report.md"
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return [f"missing evidence artifacts: {missing}"]
    if not report.exists():
        return ["missing validation report"]
    values = {name: _read(path) for name, path in paths.items()}
    reasons: list[str] = []
    checksummed = (
        "calibration campaign",
        "validation campaign",
        "validation",
        "inference",
        "external",
    )
    for name in checksummed:
        if not _checksum_valid(values[name], "evidence_checksum"):
            reasons.append(f"{name} evidence checksum mismatch")
        if values[name].get("protocol_checksum") != protocol.checksum:
            reasons.append(f"{name} protocol checksum mismatch")
    if not _checksum_valid(values["calibration"], "calibration_artifact_checksum"):
        reasons.append("calibration artifact checksum mismatch")
    calibration_campaign = values["calibration campaign"]
    validation_campaign = values["validation campaign"]
    calibration = values["calibration"]
    validation = values["validation"]
    inference = values["inference"]
    external = values["external"]
    profile = CFSupportProfile.from_dict(values["profile"])
    campaign_commits = {
        calibration_campaign.get("git_commit"),
        validation_campaign.get("git_commit"),
        inference.get("git_commit"),
        external.get("git_commit"),
    }
    if None in campaign_commits or len(campaign_commits) != 1:
        reasons.append("campaign evidence does not share one frozen commit")
    if profile.state != "promoted":
        reasons.append("support profile is not promoted")
    if profile.protocol_checksum != protocol.checksum:
        reasons.append("support profile protocol checksum mismatch")
    if calibration.get("calibration_evidence_checksum") != calibration_campaign.get(
        "evidence_checksum"
    ):
        reasons.append("calibration is not bound to its campaign")
    if not calibration.get("all_calibration_gates_passed", False):
        reasons.append("calibration gates did not all pass")
    if validation.get("campaign_evidence_checksum") != validation_campaign.get(
        "evidence_checksum"
    ):
        reasons.append("validation is not bound to its held-out campaign")
    if validation.get("inference_evidence_checksum") != inference.get("evidence_checksum"):
        reasons.append("validation is not bound to simultaneous-inference evidence")
    if validation.get("external_evidence_checksum") != external.get("evidence_checksum"):
        reasons.append("validation is not bound to external-agreement evidence")
    if not validation.get("all_validation_gates_passed", False):
        reasons.append("held-out validation gates did not all pass")
    if not inference.get("all_inference_gates_passed", False):
        reasons.append("simultaneous-inference gates did not all pass")
    implementations = external.get("end_to_end", {}).get("implementations", [])
    complete = {
        item.get("implementation")
        for item in implementations
        if item.get("status") == "complete"
    }
    if not {"DoubleMLAPOS", "EconML.DRLearner"}.issubset(complete):
        reasons.append("two independent external comparisons have not completed")
    if not external.get("shared_score", {}).get("passed", False):
        reasons.append("shared-score external comparison did not pass")
    if not external.get("all_numerical_agreement_gates_passed", False):
        reasons.append("external numerical-agreement tolerances did not all pass")
    if profile.calibration_evidence_checksum != calibration_campaign.get("evidence_checksum"):
        reasons.append("profile is not bound to calibration campaign evidence")
    if profile.validation_evidence_checksum != validation.get("evidence_checksum"):
        reasons.append("profile is not bound to held-out validation evidence")
    report_text = report.read_text(encoding="utf-8")
    required_report_identities = (
        protocol.checksum,
        calibration_campaign.get("evidence_checksum"),
        calibration.get("calibration_artifact_checksum"),
        validation_campaign.get("evidence_checksum"),
        validation.get("evidence_checksum"),
        inference.get("evidence_checksum"),
        external.get("evidence_checksum"),
        profile.checksum,
    )
    if any(str(identity) not in report_text for identity in required_report_identities):
        reasons.append("validation report does not carry every evidence identity")
    if packaged_manifest is not None:
        if not packaged_manifest.exists():
            reasons.append("packaged support-profile manifest is missing")
        else:
            manifest = _read(packaged_manifest)
            packaged = [
                item
                for item in manifest.get("profiles", [])
                if item.get("profile_id") == profile.profile_id
            ]
            if len(packaged) != 1 or packaged[0] != profile.to_dict():
                reasons.append("promoted profile is not identically packaged")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--packaged-manifest", type=Path)
    args = parser.parse_args()
    reasons = blocking_reasons(
        args.evidence_root,
        args.spec,
        packaged_manifest=args.packaged_manifest,
    )
    if reasons:
        raise SystemExit("SCOVA-CF reference release blocked:\n- " + "\n- ".join(reasons))


if __name__ == "__main__":
    main()
