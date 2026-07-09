"""Run a restartable, reduced Stage-3 validation pilot on local CPU workers."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path

from benchmarks.calibrate_stage3_gates import calibrate
from benchmarks.stage3_campaign import run_campaign
from benchmarks.summarize_stage3 import summarize
from scripts.verify_stage3_shards import verify_shards

RELEASE_SPEC = Path("benchmarks/specs/stage3_release.json")
CANDIDATES = Path("benchmarks/specs/stage3_threshold_candidates.json")
PILOT_SPEC = Path("benchmarks/specs/stage3_local_pilot.json")


def _write_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


def _valid_existing(path: Path, *, tier: str, shard_index: int, shard_count: int) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() and not sidecar.exists():
        return False
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"incomplete existing shard output: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if (
        values.get("tier") != tier
        or values.get("shard_index") != shard_index
        or values.get("shard_count") != shard_count
    ):
        raise RuntimeError(f"existing shard {path} used different settings; rerun with --force")
    claimed = sidecar.read_text(encoding="ascii").strip()
    if claimed != sha256(path.read_bytes()).hexdigest():
        raise RuntimeError(f"existing shard checksum is invalid: {path}")
    return True


def _run_shard(
    tier: str,
    output: Path,
    shard_index: int,
    shard_count: int,
    threshold_path: Path,
) -> str:
    run_campaign(
        tier=tier,
        specification_path=PILOT_SPEC,
        output=output,
        shard_index=shard_index,
        shard_count=shard_count,
        repetitions_override=None,
        bootstrap_override=None,
        seed_set="pilot",
        threshold_path=threshold_path,
    )
    return str(output)


def _recover_thresholds(calibration_dir: Path, output_dir: Path) -> Path:
    paths = sorted(calibration_dir.rglob("calibration-*.json"))
    if not paths:
        raise RuntimeError(f"no extracted calibration JSON shards found under {calibration_dir}")
    manifest = verify_shards(
        paths,
        specification_path=RELEASE_SPEC,
        tier="calibration",
        seed_set="calibration",
        threshold_path=None,
    )
    _write_json(output_dir / "calibration-shards-manifest.json", manifest)
    campaigns = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    release_spec = json.loads(RELEASE_SPEC.read_text(encoding="utf-8"))
    artifact = calibrate(campaigns, candidates, release_spec)
    threshold_path = output_dir / "stage3-directional-thresholds.json"
    _write_json(threshold_path, artifact)
    if not artifact["calibrated"]:
        raise RuntimeError(
            "no threshold profile passed; inspect stage3-directional-thresholds.json audit"
        )
    return threshold_path


def _run_tier(
    *,
    tier: str,
    label: str,
    output_dir: Path,
    threshold_path: Path,
    workers: int,
    force: bool,
) -> tuple[dict, dict]:
    shard_dir = output_dir / f"{label}-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    paths = [shard_dir / f"{label}-{index}.json" for index in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, path in enumerate(paths):
            if force:
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
            if _valid_existing(path, tier=tier, shard_index=index, shard_count=workers):
                print(f"reusing completed {label} shard {index}", flush=True)
                continue
            tasks.append(executor.submit(_run_shard, tier, path, index, workers, threshold_path))
        for future in as_completed(tasks):
            print(f"completed {future.result()}", flush=True)
    manifest = verify_shards(
        paths,
        specification_path=PILOT_SPEC,
        tier=tier,
        seed_set="pilot",
        threshold_path=threshold_path,
    )
    specification = json.loads(PILOT_SPEC.read_text(encoding="utf-8"))
    summary = summarize(paths, specification)
    _write_json(output_dir / f"{label}-shards-manifest.json", manifest)
    _write_json(output_dir / f"{label}-summary.json", summary)
    return manifest, summary


def main() -> None:
    default_workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calibration-dir", type=Path)
    source.add_argument("--thresholds", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local-artifacts/stage3-pilot"),
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        parser.error("workers must lie between 1 and 12")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = (
        _recover_thresholds(args.calibration_dir, args.output_dir)
        if args.calibration_dir is not None
        else args.thresholds
    )
    assert threshold_path is not None
    validation_manifest, validation = _run_tier(
        tier="local_validation_pilot",
        label="validation",
        output_dir=args.output_dir,
        threshold_path=threshold_path,
        workers=args.workers,
        force=args.force,
    )
    robustness_manifest, robustness = _run_tier(
        tier="local_robustness_pilot",
        label="robustness",
        output_dir=args.output_dir,
        threshold_path=threshold_path,
        workers=args.workers,
        force=args.force,
    )
    report = {
        "schema_version": 1,
        "protocol": "stage3-local-pilot-v1",
        "validation_level": "directional-pilot",
        "promotion_eligible": False,
        "threshold_artifact_sha256": json.loads(threshold_path.read_text(encoding="utf-8"))[
            "sha256"
        ],
        "validation_manifest_sha256": validation_manifest["sha256"],
        "robustness_manifest_sha256": robustness_manifest["sha256"],
        "validation_all_cells_passed": validation["all_cells_passed"],
        "robustness_all_cells_passed": robustness["all_cells_passed"],
        "limitations": [
            "25 repetitions per cell",
            "199 bootstrap draws",
            "12 validation and 12 robustness cells",
            "not valid for stable promotion or publication claims",
        ],
    }
    encoded = json.dumps(report, sort_keys=True).encode()
    report["sha256"] = sha256(encoded).hexdigest()
    _write_json(args.output_dir / "pilot-report.json", report)
    print(f"pilot complete: {args.output_dir / 'pilot-report.json'}", flush=True)


if __name__ == "__main__":
    main()
