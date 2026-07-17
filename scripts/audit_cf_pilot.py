"""Audit the full frozen pilot and its runtime margin without promoting it."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

from scova.cf import CFValidationProtocol, canonical_checksum


def _read_evidence(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def audit_pilot(
    evidence: dict[str, Any],
    metadata_paths: list[Path],
    protocol: CFValidationProtocol,
    *,
    job_limit_minutes: float = 360.0,
    required_margin: float = 0.25,
) -> dict[str, Any]:
    """Return a checksum-bound pilot verdict using worst-shard throughput."""
    reasons: list[str] = []
    if evidence.get("protocol_checksum") != protocol.checksum:
        reasons.append("pilot protocol checksum mismatch")
    if evidence.get("lane") != "pilot" or not evidence.get("complete_frozen_lane"):
        reasons.append("pilot is not the complete frozen lane")
    if evidence.get("cell_count") != 60 or evidence.get("replications_per_cell") != 20:
        reasons.append("pilot must contain all 60 cells and 20 replications per cell")
    if evidence.get("shard_count") != 16 or len(metadata_paths) != 16:
        reasons.append("pilot must contain exactly 16 shards")
    if evidence.get("execution_error_count") != 0:
        reasons.append("pilot contains execution errors")

    elapsed: list[float] = []
    records: list[int] = []
    indices: list[int] = []
    for path in metadata_paths:
        values = json.loads(path.read_text(encoding="utf-8"))
        supplied = values.pop("metadata_checksum", None)
        if supplied != canonical_checksum(values):
            reasons.append(f"pilot shard metadata checksum mismatch: {path.name}")
            continue
        if not values.get("complete_frozen_lane_configuration"):
            reasons.append(f"pilot shard is reduced: {path.name}")
        if values.get("protocol_checksum") != protocol.checksum:
            reasons.append(f"pilot shard protocol mismatch: {path.name}")
        indices.append(int(values["shard_index"]))
        elapsed.append(float(values["elapsed_seconds"]))
        records.append(int(values["record_count"]))
    if sorted(indices) != list(range(16)):
        reasons.append("pilot shard indices are incomplete or duplicated")

    calibration_records_per_shard = math.ceil(60 * protocol.calibration.count / 128)
    projected_seconds = max(
        (
            duration * calibration_records_per_shard / count
            for duration, count in zip(elapsed, records, strict=True)
            if count > 0
        ),
        default=float("inf"),
    )
    allowed_seconds = job_limit_minutes * 60 * (1 - required_margin)
    if not math.isfinite(projected_seconds) or projected_seconds > allowed_seconds:
        reasons.append("projected calibration shard runtime lacks the required 25% margin")
    result = {
        "artifact_type": "scova-cf-full-pilot-audit",
        "schema_version": 1,
        "protocol_checksum": protocol.checksum,
        "pilot_evidence_checksum": evidence.get("evidence_checksum"),
        "job_limit_minutes": job_limit_minutes,
        "required_margin": required_margin,
        "maximum_pilot_shard_seconds": max(elapsed, default=None),
        "projected_maximum_calibration_shard_minutes": (
            None if not math.isfinite(projected_seconds) else projected_seconds / 60
        ),
        "passed": not reasons,
        "blocking_reasons": reasons,
        "promotion_eligible": False,
    }
    result["evidence_checksum"] = canonical_checksum(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--pilot-evidence", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-limit-minutes", type=float, default=360.0)
    args = parser.parse_args()
    result = audit_pilot(
        _read_evidence(args.pilot_evidence),
        args.metadata,
        CFValidationProtocol.load(args.spec),
        job_limit_minutes=args.job_limit_minutes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["passed"]:
        raise SystemExit("Full pilot failed:\n- " + "\n- ".join(result["blocking_reasons"]))


if __name__ == "__main__":
    main()
