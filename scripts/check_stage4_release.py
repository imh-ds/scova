"""Reject Stage 4 promotion unless complete frozen evidence is internally consistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scova.experimental.gates import DiagnosticThresholds


def blocking_reasons(root: Path, spec_path: Path | None = None) -> list[str]:
    spec_path = spec_path or root / "benchmarks/specs/stage4_graph_release.json"
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    evidence_path = root / "release/artifacts/stage4-evidence.json"
    threshold_path = root / "release/artifacts/stage3-directional-thresholds.json"
    reasons: list[str] = []
    if not threshold_path.exists():
        return ["locked Stage 3 threshold artifact is missing"]
    try:
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        thresholds = DiagnosticThresholds.from_calibration_artifact(threshold)
        if not thresholds.calibrated:
            reasons.append("Stage 3 thresholds are not calibrated")
    except (OSError, ValueError, TypeError, KeyError) as error:
        return [f"Stage 3 threshold artifact is invalid: {error}"]
    if not evidence_path.exists():
        return reasons + ["Stage 4 evidence artifact is missing"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if evidence.get("protocol") != spec.get("protocol"):
        reasons.append("evidence protocol does not match frozen spec")
    if evidence.get("threshold_artifact_sha256") != thresholds.artifact_sha256:
        reasons.append("evidence threshold digest does not match Stage 3 artifact")
    if evidence.get("status") != "pass":
        reasons.append("evidence status is not pass")
    required = spec["directional_pass_criteria"]
    missing = [name for name in required if evidence.get("criteria", {}).get(name) is not True]
    if missing:
        reasons.append("failed criteria: " + ", ".join(missing))
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("benchmarks/specs/stage4_graph_release.json"),
    )
    args = parser.parse_args()
    reasons = blocking_reasons(args.root, args.spec)
    if reasons:
        raise SystemExit("Stage 4 promotion blocked: " + "; ".join(reasons))
    print("Stage 4 promotion evidence passes the frozen directional gate")


if __name__ == "__main__":
    main()
