"""Assemble checksummed Stage 4 directional evidence from verified summaries."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def _digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--robustness", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    threshold = json.loads(args.thresholds.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness.read_text(encoding="utf-8"))
    if validation["protocol"] != robustness["protocol"]:
        raise SystemExit("Stage 4 summaries use different protocols")
    criteria = {
        **validation["criteria"],
        **{f"robustness:{key}": value for key, value in robustness["criteria"].items()},
    }
    metrics = {
        **validation.get("metrics", {}),
        **{f"robustness:{key}": value for key, value in robustness.get("metrics", {}).items()},
    }
    payload = {
        "schema_version": 2,
        "protocol": validation["protocol"],
        "status": "pass" if validation["status"] == robustness["status"] == "pass" else "fail",
        "threshold_artifact_sha256": threshold.get("sha256"),
        "criteria": criteria,
        "metrics": metrics,
        "validation_summary_sha256": validation["summary_sha256"],
        "robustness_summary_sha256": robustness["summary_sha256"],
    }
    payload["sha256"] = _digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
