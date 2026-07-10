"""Verify complete, artifact-backed Stage 4 calibration/preflight evidence."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stage4_campaign import frozen_cells, load_specification, work_items  # noqa: E402


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def verify(
    paths: list[Path], specification: dict[str, Any], *, shard_count: int = 64
) -> dict[str, int]:
    tier = "calibration"
    cells = frozen_cells(tier, specification)
    repetitions = int(specification["tiers"][tier]["repetitions"])
    expected = set(work_items(cells, repetitions))
    observed: set[tuple[int, int]] = set()
    fingerprints: set[tuple[object, ...]] = set()
    rare_refusals = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("sha256", None)
        if claimed != _digest(payload):
            raise ValueError(f"invalid calibration shard checksum: {path}")
        fingerprint = (
            payload.get("schema_version"),
            payload.get("protocol"),
            payload.get("specification_sha256"),
            payload.get("catalog_sha256"),
            payload.get("threshold_artifact_sha256"),
            payload.get("shard_count"),
        )
        fingerprints.add(fingerprint)
        if payload.get("tier") != tier or payload.get("shard_count") != shard_count:
            raise ValueError("calibration artifact has an unexpected tier or shard count")
        if not payload.get("threshold_artifact_sha256"):
            raise ValueError("calibration artifact lacks a Stage 3 threshold digest")
        for record in payload["records"]:
            key = (int(record["cell_index"]), int(record["repetition"]))
            if key not in expected or key in observed:
                raise ValueError("calibration artifact has an unexpected or duplicate work item")
            if record.get("status") != "completed":
                raise ValueError(f"calibration work item {key} did not complete")
            if cells[key[0]].scenario == "rare_group":
                if record.get("expected_refusal") != "insufficient_per_split_group_count":
                    raise ValueError(
                        f"calibration rare-group work item {key} did not refuse safely"
                    )
                rare_refusals += 1
            observed.add(key)
    if len(fingerprints) != 1:
        raise ValueError("calibration artifacts disagree on frozen provenance")
    if observed != expected:
        raise ValueError(
            f"incomplete calibration evidence: expected {len(expected)}, found {len(observed)}"
        )
    return {"records": len(observed), "cells": len(cells), "rare_refusals": rare_refusals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("benchmarks/specs/stage4_graph_release.json"),
    )
    args = parser.parse_args()
    result = verify(args.inputs, load_specification(args.spec))
    print(
        "Verified {records} Stage 4 calibration records across {cells} cells "
        "with {rare_refusals} expected rare-group refusals".format(**result)
    )


if __name__ == "__main__":
    main()
