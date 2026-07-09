"""Verify completeness, checksums, and provenance of Stage-3 campaign shards."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def _sidecar_hash(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing checksum sidecar for {path}")
    claimed = sidecar.read_text(encoding="ascii").strip()
    actual = sha256(path.read_bytes()).hexdigest()
    if claimed != actual:
        raise ValueError(f"checksum mismatch for {path}")
    return actual


def verify_shards(
    paths: list[Path],
    *,
    specification_path: Path,
    tier: str,
    seed_set: str,
    threshold_path: Path | None,
) -> dict:
    specification_bytes = specification_path.read_bytes()
    specification = json.loads(specification_bytes)
    if not specification.get("frozen"):
        raise ValueError("campaign specification is not frozen")
    tier_spec = specification["tiers"][tier]
    payloads = []
    shard_records = []
    for path in paths:
        checksum = _sidecar_hash(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        shard_records.append(
            {
                "path": str(path),
                "sha256": checksum,
                "shard_index": payload["shard_index"],
                "records": len(payload["records"]),
            }
        )
    if not payloads:
        raise ValueError("at least one campaign shard is required")
    shard_counts = {payload["shard_count"] for payload in payloads}
    if len(shard_counts) != 1:
        raise ValueError("campaign shards disagree on shard_count")
    shard_count = shard_counts.pop()
    indices = [payload["shard_index"] for payload in payloads]
    if sorted(indices) != list(range(shard_count)):
        raise ValueError("campaign shard indices are incomplete or duplicated")
    specification_hash = sha256(specification_bytes).hexdigest()
    expected_namespace = specification[f"{seed_set}_seed_namespace"]
    commits = {payload["git_commit"] for payload in payloads}
    if len(commits) != 1 or "unavailable" in commits:
        raise ValueError("campaign shards do not identify one available git commit")
    manifests = {json.dumps(payload["cell_manifest"], sort_keys=True) for payload in payloads}
    if len(manifests) != 1:
        raise ValueError("campaign shards do not share one cell manifest")
    cell_manifest = json.loads(manifests.pop())
    if len(cell_manifest) != tier_spec["cells"]:
        raise ValueError("campaign cell manifest has the wrong size")
    source_indices = [item["source_cell_index"] for item in cell_manifest]
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("campaign cell manifest contains duplicate source cells")
    if "cell_indices" in tier_spec and source_indices != tier_spec["cell_indices"]:
        raise ValueError("campaign cell manifest does not match the frozen explicit indices")
    expected_threshold_hash = None
    if threshold_path is not None:
        threshold_values = json.loads(threshold_path.read_text(encoding="utf-8"))
        expected_threshold_hash = threshold_values.get("sha256")
        if not expected_threshold_hash:
            raise ValueError("locked threshold artifact has no checksum")
    all_records = []
    for payload in payloads:
        if payload["tier"] != tier or payload["seed_set"] != seed_set:
            raise ValueError("campaign shard has the wrong tier or seed set")
        if payload["seed_namespace"] != expected_namespace:
            raise ValueError("campaign shard has the wrong seed namespace")
        if payload["specification_sha256"] != specification_hash:
            raise ValueError("campaign shard does not match the frozen specification")
        if payload["validation_level"] != specification["validation_level"]:
            raise ValueError("campaign shard has the wrong validation level")
        if payload["repetitions"] != tier_spec["repetitions"]:
            raise ValueError("campaign shard has the wrong repetition count")
        if payload["bootstrap"] != tier_spec["bootstrap"]:
            raise ValueError("campaign shard has the wrong bootstrap count")
        if payload.get("threshold_artifact_sha256") != expected_threshold_hash:
            raise ValueError("campaign shard has the wrong threshold artifact")
        all_records.extend(payload["records"])
    expected_records = tier_spec["cells"] * tier_spec["repetitions"]
    if len(all_records) != expected_records:
        raise ValueError(f"campaign has {len(all_records)} records; expected {expected_records}")
    observed_keys = set()
    manifest_by_source = {item["source_cell_index"]: item["spec"] for item in cell_manifest}
    for record in all_records:
        key = (record["source_cell_index"], record["repetition"])
        if key in observed_keys:
            raise ValueError(f"duplicate campaign record: {key}")
        observed_keys.add(key)
        if record["spec"] != manifest_by_source[record["source_cell_index"]]:
            raise ValueError("campaign record does not match its declared source cell")
        expected_seed = expected_namespace + record["cell_id"] * 100_000 + record["repetition"]
        if record["seed"] != expected_seed:
            raise ValueError("campaign record has an invalid deterministic seed")
    result = {
        "schema_version": 1,
        "protocol": specification["protocol"],
        "tier": tier,
        "seed_set": seed_set,
        "git_commit": commits.pop(),
        "specification_sha256": specification_hash,
        "threshold_artifact_sha256": expected_threshold_hash,
        "shard_count": shard_count,
        "cell_count": len(cell_manifest),
        "record_count": len(all_records),
        "shards": sorted(shard_records, key=lambda item: item["shard_index"]),
    }
    encoded = json.dumps(result, sort_keys=True).encode()
    result["sha256"] = sha256(encoded).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/specs/stage3_release.json"))
    parser.add_argument("--tier", required=True)
    parser.add_argument(
        "--seed-set",
        choices=("calibration", "validation", "publication", "pilot"),
        required=True,
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_shards(
        args.inputs,
        specification_path=args.spec,
        tier=args.tier,
        seed_set=args.seed_set,
        threshold_path=args.thresholds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
