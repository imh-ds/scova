"""Artifact-gated Stage 4 campaign shard runner.

This runner intentionally refuses to manufacture validation evidence without
the locked Stage 3 threshold artifact.  Workers write independent JSON shards;
aggregation and promotion are separate operations.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from scova.experimental.gates import DiagnosticThresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True, choices=("pull_request", "calibration", "directional_validation", "directional_robustness", "publication_release"))
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must lie in [0, shard-count)")
    artifact = json.loads(args.thresholds.read_text(encoding="utf-8"))
    thresholds = DiagnosticThresholds.from_calibration_artifact(artifact)
    if not thresholds.calibrated:
        raise SystemExit("Stage 4 campaign requires calibrated Stage 3 thresholds")
    specification = Path(__file__).with_name("specs") / "stage4_graph_release.json"
    spec = json.loads(specification.read_text(encoding="utf-8"))
    tier = spec["tiers"][args.tier]
    payload = {
        "protocol": spec["protocol"],
        "tier": args.tier,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "threshold_version": thresholds.version,
        "threshold_artifact_sha256": thresholds.artifact_sha256,
        "requested_cells": tier["cells"],
        "requested_repetitions": tier["repetitions"],
        "requested_bootstrap": tier["bootstrap"],
        "status": "scheduled",
    }
    payload["sha256"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
