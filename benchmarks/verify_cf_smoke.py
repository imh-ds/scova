"""Verify a reduced four-shard campaign smoke run without treating it as evidence."""

from __future__ import annotations

import argparse
import gzip
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from scova.cf import CFValidationProtocol, canonical_checksum


def verify_smoke(paths: list[Path], protocol: CFValidationProtocol) -> dict[str, Any]:
    if len(paths) != 4:
        raise ValueError("The campaign smoke check requires exactly four shards")
    records: list[dict[str, Any]] = []
    indices = []
    for path in paths:
        digest = sha256(path.read_bytes()).hexdigest()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError("Smoke shard checksum mismatch")
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        supplied = metadata.pop("metadata_checksum")
        if supplied != canonical_checksum(metadata):
            raise ValueError("Smoke shard metadata checksum mismatch")
        if metadata["protocol_checksum"] != protocol.checksum:
            raise ValueError("Smoke shard uses a different protocol")
        if metadata["complete_frozen_lane_configuration"]:
            raise ValueError("Smoke verification cannot be mistaken for frozen evidence")
        indices.append(int(metadata["shard_index"]))
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    if sorted(indices) != [0, 1, 2, 3]:
        raise ValueError("Smoke shards are missing or duplicated")
    keys = {(int(record["cell_index"]), int(record["repetition"])) for record in records}
    if keys != {(cell, 0) for cell in range(4)}:
        raise ValueError("Smoke records do not cover the frozen four-cell fixture")
    result = {
        "artifact_type": "scova-cf-campaign-smoke",
        "schema_version": 2,
        "protocol_checksum": protocol.checksum,
        "record_count": len(records),
        "verified": True,
        "promotion_eligible": False,
    }
    result["evidence_checksum"] = canonical_checksum(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_smoke(args.inputs, CFValidationProtocol.load(args.spec))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
