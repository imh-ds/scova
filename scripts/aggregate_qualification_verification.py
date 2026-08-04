"""Fail-closed promotion-prerequisite aggregation for qualification lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scova.cf import canonical_checksum


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(*, external: dict[str, Any] | None, inference: dict[str, Any] | None,
              validation: dict[str, Any] | None, boundary: dict[str, Any] | None) -> dict[str, Any]:
    artifacts = {"external": external, "inference": inference, "validation": validation}
    reasons: list[str] = []
    identity: dict[str, Any] | None = None
    for lane, artifact in artifacts.items():
        if artifact is None:
            reasons.append(f"missing {lane} evidence")
            continue
        checksum_field = "evidence_checksum"
        if artifact.get(checksum_field) != canonical_checksum({k: v for k, v in artifact.items() if k != checksum_field}):
            reasons.append(f"{lane} checksum mismatch")
        if artifact.get("program_type") != "qualification" or artifact.get("verification_lane") != lane:
            reasons.append(f"{lane} artifact role mismatch")
        if not artifact.get("informative", False):
            reasons.append(f"{lane} evidence is incomplete or uninformative")
        gate = "all_numerical_agreement_gates_passed" if lane == "external" else (
            "all_inference_gates_passed" if lane == "inference" else "all_validation_gates_passed"
        )
        if not artifact.get(gate, False):
            reasons.append(f"{lane} gate did not pass")
        fields = {name: artifact.get(name) for name in ("protocol_checksum", "design_checksum", "dependency_lock_checksum", "decision_manifest_checksum")}
        if identity is None:
            identity = fields
        elif fields != identity:
            reasons.append(f"{lane} provenance does not match the other promotion evidence")
    if boundary is not None and boundary.get("promotion_required"):
        reasons.append("boundary diagnostic must not become a promotion prerequisite")
    result = {
        "artifact_type": "scova-cf-qualification-verification-aggregate", "schema_version": 1,
        "program_type": "qualification", "verification_lane": "aggregate",
        "promotion_decision": "eligible-for-human-promotion-review" if not reasons else "blocked",
        "reasons": reasons, "boundary_status": None if boundary is None else boundary.get("status"),
        **({} if identity is None else identity),
    }
    return {**result, "artifact_checksum": canonical_checksum(result)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        external=None if args.external is None else _read(args.external),
        inference=None if args.inference is None else _read(args.inference),
        validation=None if args.validation is None else _read(args.validation),
        boundary=None if args.boundary is None else _read(args.boundary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result["reasons"]:
        raise SystemExit("Qualification verification aggregation blocked:\n- " + "\n- ".join(result["reasons"]))


if __name__ == "__main__":
    main()
