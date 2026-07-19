"""Focused simultaneous-inference campaign for the SCOVA-CF reference profile."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cf_reference_campaign import (
    dependency_lock_checksum,
    fit_campaign_record,
    simulate_reference_cell,
    write_deterministic_gzip,
)
from scova.cf import CFValidationProtocol, canonical_checksum

N_BOOTSTRAP = 999

_CF_NUMERICAL_PATHS = (
    "src/scova/_aipw.py",
    "src/scova/cf",
    "benchmarks/cf_external_agreement.py",
    "benchmarks/cf_external_validation.py",
    "benchmarks/cf_reference_campaign.py",
)


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _cf_numerical_fingerprint(commit: str) -> str:
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", *_CF_NUMERICAL_PATHS],
        text=True,
    ).splitlines()
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(subprocess.check_output(["git", "show", f"{commit}:{path}"]))
        digest.update(b"\0")
    return digest.hexdigest()


def run_shard(
    protocol: CFValidationProtocol,
    *,
    output: Path,
    shard_index: int,
    shard_count: int,
    replications: int | None = None,
    resume: bool = False,
) -> None:
    assert protocol.inference is not None
    count = protocol.inference.count if replications is None else replications
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial.ndjson")
    completed: set[tuple[int, int]] = set()
    if resume and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            focused_index = int(record["focused_cell_index"])
            repetition = int(record["repetition"])
            global_index = focused_index * count + repetition
            expected_seed = (
                protocol.inference.start
                + focused_index * protocol.inference.count
                + repetition
            )
            if (
                not 0 <= focused_index < len(protocol.inference_cells)
                or not 0 <= repetition < count
                or global_index % shard_count != shard_index
                or int(record["seed"]) != expected_seed
            ):
                raise ValueError("Inference checkpoint does not belong to this shard")
            completed.add((focused_index, repetition))
    elif partial.exists():
        partial.unlink()
    with partial.open("a", encoding="utf-8") as stream:
        written = 0
        for focused_index, reference in enumerate(protocol.inference_cells):
            cell_index = int(reference["simulation_cell_index"])
            cell = protocol.retained_cells[cell_index]
            for repetition in range(count):
                global_index = focused_index * count + repetition
                if (
                    global_index % shard_count != shard_index
                    or (focused_index, repetition) in completed
                ):
                    continue
                seed = (
                    protocol.inference.start
                    + focused_index * protocol.inference.count
                    + repetition
                )
                try:
                    generated = simulate_reference_cell(cell, seed=seed)
                    fitted = fit_campaign_record(
                        generated,
                        cell,
                        include_stability=False,
                        simultaneous_bootstrap=N_BOOTSTRAP,
                        seed=seed,
                    )
                except Exception as error:  # retain failures as auditable records
                    fitted = {
                        "refused": True,
                        "status_code": "execution-error",
                        "execution_error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                record = {
                    "focused_cell_index": focused_index,
                    "simulation_cell_index": cell_index,
                    "cell": dict(cell),
                    "repetition": repetition,
                    "seed": seed,
                    **fitted,
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                written += 1
                if written % 25 == 0:
                    stream.flush()
    text = partial.read_text(encoding="utf-8")
    write_deterministic_gzip(output, text)
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    metadata = {
        "artifact_type": "scova-cf-inference-shard",
        "schema_version": 2,
        "protocol_checksum": protocol.checksum,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "replications_per_cell": count,
        "bootstrap_replications": N_BOOTSTRAP,
        "dependency_lock_checksum": dependency_lock_checksum(),
        "git_commit": _commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: _version(name) for name in ("scova", "numpy", "scipy", "scikit-learn")},
        },
        "records_sha256": digest,
        "record_count": len(text.splitlines()),
    }
    metadata["metadata_checksum"] = canonical_checksum(metadata)
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def aggregate(
    paths: list[Path], *, protocol: CFValidationProtocol
) -> dict[str, Any]:
    assert protocol.inference is not None
    records = []
    metadata = []
    for path in paths:
        digest = sha256(path.read_bytes()).hexdigest()
        if path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip() != digest:
            raise ValueError("Inference shard checksum mismatch")
        values = json.loads(
            path.with_suffix(path.suffix + ".metadata.json").read_text(encoding="utf-8")
        )
        supplied = values.pop("metadata_checksum")
        if supplied != canonical_checksum(values) or values["records_sha256"] != digest:
            raise ValueError("Inference shard metadata mismatch")
        if values.get("dependency_lock_checksum") != protocol.dependency_lock_checksum:
            raise ValueError("Inference shard dependency-lock checksum mismatch")
        if values.get("protocol_checksum") != protocol.checksum:
            raise ValueError("Inference shard protocol checksum mismatch")
        metadata.append(values)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            shard_records = [json.loads(line) for line in stream if line.strip()]
        if values["replications_per_cell"] != protocol.inference.count:
            raise ValueError("Reduced inference shards cannot form release evidence")
        for record in shard_records:
            global_index = (
                int(record["focused_cell_index"]) * protocol.inference.count
                + int(record["repetition"])
            )
            if global_index % int(values["shard_count"]) != int(values["shard_index"]):
                raise ValueError("Inference record is assigned to the wrong shard")
        records.extend(shard_records)
    shard_counts = {value["shard_count"] for value in metadata}
    indices = sorted(value["shard_index"] for value in metadata)
    if len(shard_counts) != 1 or indices != list(range(shard_counts.pop())):
        raise ValueError("Inference shards are incomplete or inconsistent")
    if len(metadata) != 64:
        raise ValueError("Frozen simultaneous-inference evidence requires 64 shards")
    commits = {value["git_commit"] for value in metadata}
    environments = {json.dumps(value["environment"], sort_keys=True) for value in metadata}
    if len(commits) != 1 or len(environments) != 1:
        raise ValueError("Inference shards mix commits or environments")
    source_commits = sorted(commits)
    execution_commit = _commit()
    if execution_commit == "unavailable":
        raise ValueError("Inference aggregation requires a Git commit")
    source_fingerprints = {_cf_numerical_fingerprint(commit) for commit in source_commits}
    if source_fingerprints != {_cf_numerical_fingerprint(execution_commit)}:
        raise ValueError("Inference shards use a different SCOVA-CF numerical implementation")
    environment = json.loads(next(iter(environments)))
    for package, expected_version in protocol.software.items():
        if package in environment and environment[package] != expected_version:
            raise ValueError(f"Inference shard has wrong {package} version")
    expected = len(protocol.inference_cells) * protocol.inference.count
    if len(records) != expected:
        raise ValueError(f"Inference campaign has {len(records)} records; expected {expected}")
    keys = set()
    audits = []
    multiplier = float(protocol.metrics["monte_carlo_standard_error_multiplier"])
    for record in records:
        key = (record["focused_cell_index"], record["repetition"])
        if key in keys:
            raise ValueError("Duplicate inference campaign record")
        keys.add(key)
        expected_seed = (
            protocol.inference.start
            + record["focused_cell_index"] * protocol.inference.count
            + record["repetition"]
        )
        if record["seed"] != expected_seed:
            raise ValueError("Inference record has an invalid seed")
    all_passed = True
    for focused_index in range(len(protocol.inference_cells)):
        cell_records = [r for r in records if r["focused_cell_index"] == focused_index]
        if any(r["refused"] for r in cell_records):
            audit = {"passed": False, "reason": "unexpected-refusal"}
        else:
            fwer = float(np.mean([r["simultaneous"]["any_null_rejected"] for r in cell_records]))
            coverage = float(np.mean([r["simultaneous"]["covered_family"] for r in cell_records]))
            mcse = np.sqrt(0.05 * 0.95 / len(cell_records))
            effect = str(cell_records[0]["cell"]["effect"])
            omnibus_size = (
                None
                if effect != "null"
                else float(
                    np.mean(
                        [
                            record["omnibus"]["p_value"] < 0.05
                            for record in cell_records
                        ]
                    )
                )
            )
            omnibus_passed = bool(
                omnibus_size is None
                or abs(omnibus_size - 0.05) <= multiplier * mcse
            )
            passed = bool(
                abs(fwer - 0.05) <= multiplier * mcse
                and coverage >= 0.95 - multiplier * mcse
                and omnibus_passed
            )
            audit = {
                "passed": passed,
                "familywise_error": fwer,
                "family_coverage": coverage,
                "omnibus_size": omnibus_size,
            }
        all_passed &= bool(audit["passed"])
        audits.append({"focused_cell_index": focused_index, **audit})
    evidence = {
        "artifact_type": "scova-cf-inference-validation",
        "schema_version": 2,
        "protocol_checksum": protocol.checksum,
        "git_commit": execution_commit,
        "source_git_commits": source_commits,
        "numerical_implementation_checksum": source_fingerprints.pop(),
        "environment": json.loads(environments.pop()),
        "bootstrap_replications": N_BOOTSTRAP,
        "all_inference_gates_passed": all_passed,
        "audit": audits,
    }
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--aggregate", type=Path, nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = CFValidationProtocol.load(args.spec)
    if args.aggregate:
        evidence = aggregate(args.aggregate, protocol=protocol)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    else:
        if args.shard_index is None or args.shard_count is None:
            parser.error("shard index and count are required")
        run_shard(
            protocol,
            output=args.output,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            replications=args.replications,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
