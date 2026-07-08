"""Fail unless every artifact-backed Stage-3 promotion gate is satisfied."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

try:
    from scripts.check_critical_coverage import coverage_failures
except ModuleNotFoundError:  # direct execution places scripts/ on sys.path
    from check_critical_coverage import coverage_failures


def _read_json(path: Path, reasons: list[str], role: str) -> dict | None:
    if not path.exists():
        reasons.append(f"missing {role} artifact: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        reasons.append(f"invalid {role} artifact: {error}")
        return None


def _verify_embedded_hash(
    values: dict,
    field: str,
    *,
    indent: int | None = None,
    allow_nan: bool = True,
) -> bool:
    unsigned = dict(values)
    claimed = unsigned.pop(field, None)
    encoded = json.dumps(
        unsigned,
        indent=indent,
        sort_keys=True,
        allow_nan=allow_nan,
    ).encode()
    return claimed == sha256(encoded).hexdigest()


def blocking_reasons(manifest: dict, root: Path = Path(".")) -> list[str]:
    reasons: list[str] = []
    if manifest.get("validation_level") != "directional":
        reasons.append("promotion manifest is not a directional validation manifest")
    for key, value in manifest.get("theory", {}).items():
        if not value:
            reasons.append(f"theory.{key} is false")

    protocol = manifest["protocol"]
    specification_path = root / protocol["specification"]
    candidate_path = root / protocol["threshold_candidates"]
    specification = _read_json(specification_path, reasons, "protocol specification")
    candidates = _read_json(candidate_path, reasons, "threshold candidates")
    if specification is not None and (
        not specification.get("frozen")
        or specification.get("validation_level") != "directional"
    ):
        reasons.append("protocol specification is not frozen directional validation")
    if candidates is not None and not candidates.get("frozen"):
        reasons.append("threshold candidate specification is not frozen")

    paths = {name: root / path for name, path in manifest["artifacts"].items()}
    artifacts = {
        name: _read_json(path, reasons, name) for name, path in paths.items()
    }
    thresholds = artifacts["thresholds"]
    packaged_thresholds = artifacts["packaged_thresholds"]
    threshold_hash = None
    if thresholds is not None:
        threshold_hash = thresholds.get("sha256")
        if not _verify_embedded_hash(thresholds, "sha256"):
            reasons.append("threshold artifact checksum is invalid")
        if not thresholds.get("calibrated") or thresholds.get("validation_level") != "directional":
            reasons.append("threshold artifact is not directional and calibrated")
    if thresholds is not None and packaged_thresholds != thresholds:
        reasons.append("packaged production thresholds do not match the locked artifact")

    specification_hash = (
        sha256(specification_path.read_bytes()).hexdigest() if specification_path.exists() else None
    )
    shard_commits = set()
    for role, expected_tier in (
        ("calibration_shards", "calibration"),
        ("validation_shards", "directional_validation"),
        ("robustness_shards", "directional_robustness"),
    ):
        shard_manifest = artifacts[role]
        if shard_manifest is None:
            continue
        if not _verify_embedded_hash(shard_manifest, "sha256"):
            reasons.append(f"{role} checksum is invalid")
        if shard_manifest.get("tier") != expected_tier:
            reasons.append(f"{role} has the wrong campaign tier")
        if shard_manifest.get("specification_sha256") != specification_hash:
            reasons.append(f"{role} does not match the frozen specification")
        expected_threshold = None if role == "calibration_shards" else threshold_hash
        if shard_manifest.get("threshold_artifact_sha256") != expected_threshold:
            reasons.append(f"{role} has the wrong threshold provenance")
        if specification is not None:
            tier_spec = specification["tiers"][expected_tier]
            expected_records = tier_spec["cells"] * tier_spec["repetitions"]
            if shard_manifest.get("record_count") != expected_records:
                reasons.append(f"{role} has the wrong record count")
        shard_commits.add(shard_manifest.get("git_commit"))
    if len(shard_commits) > 1 or None in shard_commits:
        reasons.append("campaign shard manifests do not share one git commit")
    for role, expected_tier in (
        ("validation_summary", "directional_validation"),
        ("robustness_summary", "directional_robustness"),
    ):
        summary = artifacts[role]
        if summary is None:
            continue
        if not _verify_embedded_hash(summary, "summary_sha256", allow_nan=False):
            reasons.append(f"{role} checksum is invalid")
        if not summary.get("tier_passed"):
            reasons.append(f"{role} did not pass its pooled directional criteria")
        if summary.get("tier") != expected_tier:
            reasons.append(f"{role} has the wrong campaign tier")
        if summary.get("specification_sha256") != specification_hash:
            reasons.append(f"{role} does not match the frozen specification")
        if summary.get("threshold_artifact_sha256") != threshold_hash:
            reasons.append(f"{role} does not match the locked thresholds")

    quality = manifest["quality"]
    for role, expected_version in zip(
        ("jax_minimum", "jax_maximum"), quality["required_jax_versions"], strict=True
    ):
        matrix = artifacts[role]
        if matrix is None:
            continue
        if not _verify_embedded_hash(matrix, "sha256", indent=2):
            reasons.append(f"{role} checksum is invalid")
        if matrix.get("jax_version") != expected_version:
            reasons.append(f"{role} was not run with JAX {expected_version}")
        if matrix.get("cases", 0) < quality["minimum_jax_cases_per_version"]:
            reasons.append(f"{role} has too few randomized cases")
        if matrix.get("failures"):
            reasons.append(f"{role} contains gradient failures")

    memory = artifacts["memory"]
    if memory is not None:
        if not _verify_embedded_hash(memory, "sha256"):
            reasons.append("memory benchmark checksum is invalid")
        if (
            not memory.get("peak_below_unbatched_multiplier_cube")
            or memory.get("nuisance_refit_possible_from_result")
        ):
            reasons.append("memory benchmark did not establish bounded no-refit inference")
    build = artifacts["build"]
    if build is not None:
        if not _verify_embedded_hash(build, "sha256"):
            reasons.append("package build artifact checksum is invalid")
        if not build.get("passed"):
            reasons.append("package build artifact reports failure")
    coverage = artifacts["coverage"]
    if coverage is not None:
        reasons.extend(
            coverage_failures(
                coverage,
                critical_floor=quality["minimum_critical_branch_coverage"],
                package_floor=quality["minimum_package_branch_coverage"],
            )
        )
    evidence = artifacts["evidence"]
    if evidence is not None:
        if not _verify_embedded_hash(evidence, "report_sha256", indent=2):
            reasons.append("evidence report checksum is invalid")
        if not evidence.get("complete"):
            reasons.append("evidence report is incomplete")
        if evidence.get("validation_level") != "directional":
            reasons.append("evidence report has the wrong validation level")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("release/stage3_promotion.json"),
    )
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reasons = blocking_reasons(manifest, args.manifest.parent.parent)
    if reasons:
        print("Stage 3 is not eligible for stable promotion:")
        for reason in reasons:
            print(f"- {reason}")
        if not args.allow_pending:
            raise SystemExit(1)
    else:
        print("All Stage-3 directional promotion gates are satisfied.")


if __name__ == "__main__":
    main()
