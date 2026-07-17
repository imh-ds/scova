"""Render the checksum-bound SCOVA-CF validation or blocking report."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from scova.cf import CFValidationProtocol


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def render(root: Path, protocol: CFValidationProtocol) -> str:
    names = {
        "Calibration campaign": "cf-reference-calibration-campaign.json.gz",
        "Calibration audit": "cf-reference-calibration.json",
        "Held-out campaign": "cf-reference-validation-campaign.json.gz",
        "Held-out audit": "cf-reference-validation.json",
        "Simultaneous inference": "cf-reference-inference.json",
        "External agreement": "cf-reference-external-agreement.json",
        "Promoted profile": "cf-reference-support-profile.json",
    }
    evidence = {label: _read(root / filename) for label, filename in names.items()}
    validation = evidence["Held-out audit"] or {}
    promoted = bool(
        validation.get("all_validation_gates_passed", False)
        and evidence["Promoted profile"]
    )
    lines = [
        "# SCOVA-CF randomized reference validation report",
        "",
        f"- Protocol: `{protocol.protocol_id}`",
        f"- Protocol checksum: `{protocol.checksum}`",
        f"- Decision: `{'promoted' if promoted else 'blocked'}`",
        "",
        "## Evidence identities",
        "",
    ]
    for label, values in evidence.items():
        if values is None:
            lines.append(f"- {label}: missing")
            continue
        checksum = (
            values.get("evidence_checksum")
            or values.get("calibration_artifact_checksum")
            or values.get("profile_checksum")
        )
        lines.append(f"- {label}: `{checksum}`")
    lines.extend(["", "## Gate summary", ""])
    for label, field in (
        ("Calibration", "all_calibration_gates_passed"),
        ("Held-out validation", "all_validation_gates_passed"),
        ("Simultaneous inference", "all_inference_gates_passed"),
        ("External numerical agreement", "all_numerical_agreement_gates_passed"),
    ):
        source = {
            "Calibration": evidence["Calibration audit"],
            "Held-out validation": evidence["Held-out audit"],
            "Simultaneous inference": evidence["Simultaneous inference"],
            "External numerical agreement": evidence["External agreement"],
        }[label]
        status = "missing" if source is None else "pass" if source.get(field) else "fail"
        lines.append(f"- {label}: **{status}**")
    lines.extend(
        [
            "",
            "The profile is limited to randomized independent-unit continuous-outcome analyses ",
            "with known assignment probabilities and unnormalized cross-fitted AIPW. No result ",
            "in this report identifies respondent-specific missing outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(args.evidence_root, CFValidationProtocol.load(args.spec)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
