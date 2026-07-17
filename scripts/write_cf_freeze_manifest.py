"""Record the immutable commit and checksums for the SCOVA-CF v3 campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from benchmarks.cf_reference_campaign import dependency_lock_checksum
from scova.cf import CFValidationProtocol, canonical_checksum


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def freeze_manifest(
    protocol: CFValidationProtocol, *, required_tag: str | None = None
) -> dict[str, object]:
    commit = _git("rev-parse", "HEAD")
    tags = tuple(value for value in _git("tag", "--points-at", "HEAD").splitlines() if value)
    if required_tag is not None and required_tag not in tags:
        raise ValueError(f"HEAD is not tagged {required_tag!r}")
    if dependency_lock_checksum() != protocol.dependency_lock_checksum:
        raise ValueError("Dependency lock does not match the frozen protocol")
    values: dict[str, object] = {
        "artifact_type": "scova-cf-reference-v3-freeze",
        "schema_version": 1,
        "git_commit": commit,
        "required_tag": required_tag,
        "tags_at_commit": list(tags),
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "dependency_lock_checksum": protocol.dependency_lock_checksum,
        "dataset_checksums": dict(protocol.dataset_checksums or {}),
    }
    values["manifest_checksum"] = canonical_checksum(values)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-tag")
    args = parser.parse_args()
    values = freeze_manifest(
        CFValidationProtocol.load(args.spec), required_tag=args.require_tag
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
