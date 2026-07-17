"""Verify and aggregate SCOVA-CF campaign shards without relaxing provenance."""

from __future__ import annotations

import argparse
import gzip
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cf_reference_campaign import (
    plasmode_source_checksum,
    write_deterministic_gzip,
)
from scova.cf import CFValidationProtocol, canonical_checksum


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing shard metadata: {metadata_path}")
    values = json.loads(metadata_path.read_text(encoding="utf-8"))
    supplied = values.pop("metadata_checksum")
    if supplied != canonical_checksum(values):
        raise ValueError(f"Shard metadata checksum mismatch: {path}")
    values["metadata_checksum"] = supplied
    return values


def _read_records(path: Path) -> list[dict[str, Any]]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != sha256(
        path.read_bytes()
    ).hexdigest():
        raise ValueError(f"Shard checksum mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cell_index in sorted({int(record["cell_index"]) for record in records}):
        cell_records = [record for record in records if record["cell_index"] == cell_index]
        usable = [record for record in cell_records if not record["refused"]]
        contrasts = [item for record in usable for item in record["contrasts"]]
        errors = np.array(
            [item["estimate"] - item["truth"] for item in contrasts], dtype=float
        )
        standard_errors = np.array(
            [item["standard_error"] for item in contrasts], dtype=float
        )
        empirical_sd = float(errors.std(ddof=1)) if len(errors) > 1 else None
        nulls = [item for item in contrasts if item["null"]]
        result.append(
            {
                "cell_index": cell_index,
                "cell_kind": cell_records[0]["cell_kind"],
                "cell": cell_records[0]["cell"],
                "replications": len(cell_records),
                "refusal_rate": 1 - len(usable) / len(cell_records),
                "coverage": (
                    None
                    if not contrasts
                    else float(np.mean([item["covered"] for item in contrasts]))
                ),
                "type_i_error": (
                    None if not nulls else float(np.mean([item["rejected"] for item in nulls]))
                ),
                "bias": None if not len(errors) else float(errors.mean()),
                "empirical_standard_deviation": empirical_sd,
                "mean_standard_error": (
                    None if not len(standard_errors) else float(standard_errors.mean())
                ),
                "standard_error_ratio": (
                    None
                    if empirical_sd in (None, 0)
                    else float(standard_errors.mean() / empirical_sd)
                ),
            }
        )
    return result


def aggregate_shards(
    paths: list[Path], *, protocol: CFValidationProtocol, lane: str
) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one shard is required")
    metadata = [_read_metadata(path) for path in paths]
    for path, values in zip(paths, metadata, strict=True):
        if values["records_sha256"] != sha256(path.read_bytes()).hexdigest():
            raise ValueError("Campaign shard metadata does not bind its record archive")
    shard_counts = {int(value["shard_count"]) for value in metadata}
    if len(shard_counts) != 1:
        raise ValueError("Campaign shards disagree on shard_count")
    shard_count = shard_counts.pop()
    if lane in {"calibration", "validation"} and shard_count != 128:
        raise ValueError("Frozen calibration and validation evidence requires 128 shards")
    indices = [int(value["shard_index"]) for value in metadata]
    if sorted(indices) != list(range(shard_count)):
        raise ValueError("Campaign shard indices are incomplete or duplicated")
    commits = {value["git_commit"] for value in metadata}
    environments = {json.dumps(value["environment"], sort_keys=True) for value in metadata}
    if len(commits) != 1 or len(environments) != 1 or "unavailable" in commits:
        raise ValueError("Campaign shards mix commits or environments")
    candidate_checksums = {value.get("candidate_profile_checksum") for value in metadata}
    nonnull_candidates = {value for value in candidate_checksums if value is not None}
    invalid_candidate_lock = (
        len(candidate_checksums) != 1
        or (lane == "validation" and None in candidate_checksums)
        or (lane != "validation" and candidate_checksums != {None})
    )
    if invalid_candidate_lock:
        raise ValueError("Campaign shards have a missing or mixed candidate-profile lock")
    expected_sources = {
        name: plasmode_source_checksum(name) for name in ("diabetes", "breast-cancer")
    }
    if expected_sources != dict(protocol.dataset_checksums or {}):
        raise ValueError("Installed plasmode datasets do not match the frozen protocol")
    for value in metadata:
        if value["protocol_checksum"] != protocol.checksum or value["lane"] != lane:
            raise ValueError("Campaign shard uses the wrong protocol or lane")
        if value["plasmode_source_checksums"] != expected_sources:
            raise ValueError("Campaign shard plasmode source checksum mismatch")
        if value.get("dependency_lock_checksum") != protocol.dependency_lock_checksum:
            raise ValueError("Campaign shard dependency-lock checksum mismatch")
        if not value["complete_frozen_lane_configuration"]:
            raise ValueError("Smoke or reduced shards cannot form release evidence")
        for package, expected_version in protocol.software.items():
            if (
                package in value["environment"]
                and value["environment"][package] != expected_version
            ):
                raise ValueError(f"Campaign shard has wrong {package} version")
    partition = getattr(protocol, lane)
    cells = [*protocol.retained_cells, *protocol.plasmode_cells]
    records = []
    for path, values in zip(paths, metadata, strict=True):
        shard_records = _read_records(path)
        for record in shard_records:
            global_index = (
                int(record["cell_index"]) * partition.count + int(record["repetition"])
            )
            if global_index % shard_count != int(values["shard_index"]):
                raise ValueError("Campaign record is assigned to the wrong shard")
        records.extend(shard_records)
    expected = len(cells) * partition.count
    if len(records) != expected:
        raise ValueError(f"Campaign has {len(records)} records; expected {expected}")
    keys: set[tuple[int, int]] = set()
    for record in records:
        cell_index = int(record["cell_index"])
        repetition = int(record["repetition"])
        key = (cell_index, repetition)
        if key in keys:
            raise ValueError(f"Duplicate campaign record: {key}")
        keys.add(key)
        if record["cell"] != dict(cells[cell_index]):
            raise ValueError("Campaign record does not match the frozen cell")
        expected_seed = partition.start + cell_index * partition.count + repetition
        if int(record["seed"]) != expected_seed:
            raise ValueError("Campaign record has an invalid deterministic seed")
    records.sort(key=lambda value: (value["cell_index"], value["repetition"]))
    evidence = {
        "artifact_type": "scova-cf-reference-campaign",
        "schema_version": 2,
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "lane": lane,
        "complete_frozen_lane": True,
        "git_commit": commits.pop(),
        "environment": json.loads(environments.pop()),
        "plasmode_source_checksums": expected_sources,
        "candidate_profile_checksum": (
            None if not nonnull_candidates else nonnull_candidates.pop()
        ),
        "replications_per_cell": partition.count,
        "cell_count": len(cells),
        "shard_count": shard_count,
        "summaries": _summaries(records),
        "records": records,
        "promotion_decision": "blocked/pending-profile-validation",
    }
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lane", choices=("pilot", "calibration", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = aggregate_shards(
        args.inputs, protocol=CFValidationProtocol.load(args.spec), lane=args.lane
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_gzip(
        args.output,
        json.dumps(evidence, sort_keys=True, allow_nan=False),
    )
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        sha256(args.output.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
