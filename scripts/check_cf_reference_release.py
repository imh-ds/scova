"""Fail closed until all SCOVA-CF randomized-reference promotion evidence exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum


def blocking_reasons(root: Path, spec: Path) -> list[str]:
    protocol = CFValidationProtocol.load(spec)
    paths = {
        "calibration": root / "cf-reference-calibration.json",
        "validation": root / "cf-reference-validation.json",
        "external": root / "cf-reference-external-agreement.json",
        "profile": root / "cf-reference-support-profile.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return [f"missing evidence artifacts: {missing}"]
    reasons: list[str] = []
    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    external = json.loads(paths["external"].read_text(encoding="utf-8"))
    profile_values = json.loads(paths["profile"].read_text(encoding="utf-8"))
    profile = CFSupportProfile.from_dict(profile_values)
    if profile.state != "promoted":
        reasons.append("support profile is not promoted")
    if profile.protocol_checksum != protocol.checksum:
        reasons.append("support profile protocol checksum mismatch")
    if not calibration.get("all_calibration_gates_passed", False):
        reasons.append("calibration gates did not all pass")
    if not validation.get("all_validation_gates_passed", False):
        reasons.append("held-out validation gates did not all pass")
    if profile.validation_evidence_checksum != validation.get("evidence_checksum"):
        reasons.append("profile is not bound to the held-out validation evidence")
    implementations = external.get("implementations", [])
    complete = {
        value.get("implementation")
        for value in implementations
        if value.get("status") == "complete"
    }
    if not {"DoubleMLAPOS", "EconML.DRLearner"}.issubset(complete):
        reasons.append("two independent external comparisons have not completed")
    if not external.get("all_numerical_agreement_gates_passed", False):
        reasons.append("external numerical-agreement tolerances did not all pass")
    supplied = external.get("evidence_checksum")
    external_payload = {k: v for k, v in external.items() if k != "evidence_checksum"}
    if supplied != canonical_checksum(external_payload):
        reasons.append("external agreement evidence checksum mismatch")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    reasons = blocking_reasons(args.evidence_root, args.spec)
    if reasons:
        raise SystemExit("SCOVA-CF reference release blocked:\n- " + "\n- ".join(reasons))


if __name__ == "__main__":
    main()
