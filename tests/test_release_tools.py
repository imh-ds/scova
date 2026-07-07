import json
from hashlib import sha256
from pathlib import Path

from benchmarks.stage3_campaign import _failed_fit
from scripts.check_critical_coverage import CRITICAL_SUFFIXES, coverage_failures
from scripts.check_stage3_release import blocking_reasons
from scripts.generate_stage3_evidence import _artifact_record


def _coverage_report(fraction: float = 1.0) -> dict:
    summary = {
        "num_statements": 50,
        "covered_lines": round(50 * fraction),
        "num_branches": 10,
        "covered_branches": round(10 * fraction),
    }
    return {
        "files": {f"src/{suffix}": {"summary": summary} for suffix in CRITICAL_SUFFIXES},
        "totals": summary,
    }


def test_coverage_gate_accepts_and_rejects_reports() -> None:
    assert coverage_failures(_coverage_report()) == []
    failures = coverage_failures(_coverage_report(0.8))
    assert len(failures) == len(CRITICAL_SUFFIXES) + 1
    missing = _coverage_report()
    missing["files"].pop(next(iter(missing["files"])))
    assert "expected exactly one" in coverage_failures(missing)[0]


def test_promotion_checker_uses_artifacts_without_review_blockers() -> None:
    manifest = json.loads(Path("release/stage3_promotion.json").read_text(encoding="utf-8"))
    reasons = blocking_reasons(manifest)
    assert not any("review" in reason for reason in reasons)
    assert any("missing thresholds artifact" in reason for reason in reasons)
    assert any("missing validation_summary artifact" in reason for reason in reasons)


def test_evidence_and_campaign_failures_are_explicit(tmp_path: Path) -> None:
    missing = _artifact_record("simulation_summary", None)
    assert missing == {
        "role": "simulation_summary",
        "path": None,
        "present": False,
        "sha256": None,
    }
    present_path = tmp_path / "artifact.json"
    present_path.write_text("{}", encoding="utf-8")
    present = _artifact_record("artifact", present_path)
    assert present["present"] is True
    assert present["sha256"]

    failure = _failed_fit(RuntimeError("deliberate"))
    assert failure["refused"] is True
    assert failure["execution_error"] == {
        "type": "RuntimeError",
        "message": "deliberate",
    }


def _write_hashed(
    path: Path,
    values: dict,
    field: str,
    *,
    indent: int | None = None,
    allow_nan: bool = True,
) -> dict:
    payload = dict(values)
    encoded = json.dumps(
        payload, indent=indent, sort_keys=True, allow_nan=allow_nan
    ).encode()
    payload[field] = sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_artifact_backed_promotion_can_pass(tmp_path: Path) -> None:
    manifest = json.loads(Path("release/stage3_promotion.json").read_text(encoding="utf-8"))
    specification_path = tmp_path / manifest["protocol"]["specification"]
    specification_path.parent.mkdir(parents=True)
    specification_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "validation_level": "directional",
                "tiers": {
                    "calibration": {"cells": 1, "repetitions": 2},
                    "directional_validation": {"cells": 1, "repetitions": 2},
                    "directional_robustness": {"cells": 1, "repetitions": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_path = tmp_path / manifest["protocol"]["threshold_candidates"]
    candidate_path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    artifact_paths = {
        name: tmp_path / value for name, value in manifest["artifacts"].items()
    }
    threshold = _write_hashed(
        artifact_paths["thresholds"],
        {"calibrated": True, "validation_level": "directional"},
        "sha256",
    )
    artifact_paths["packaged_thresholds"].parent.mkdir(parents=True)
    artifact_paths["packaged_thresholds"].write_text(json.dumps(threshold), encoding="utf-8")
    specification_hash = sha256(specification_path.read_bytes()).hexdigest()
    for role, tier in (
        ("calibration_shards", "calibration"),
        ("validation_shards", "directional_validation"),
        ("robustness_shards", "directional_robustness"),
    ):
        _write_hashed(
            artifact_paths[role],
            {
                "tier": tier,
                "git_commit": "abc123",
                "specification_sha256": specification_hash,
                "threshold_artifact_sha256": (
                    None if role == "calibration_shards" else threshold["sha256"]
                ),
                "record_count": 2,
            },
            "sha256",
        )
    for role, tier in (
        ("validation_summary", "directional_validation"),
        ("robustness_summary", "directional_robustness"),
    ):
        _write_hashed(
            artifact_paths[role],
            {
                "all_cells_passed": True,
                "tier": tier,
                "specification_sha256": specification_hash,
                "threshold_artifact_sha256": threshold["sha256"],
            },
            "summary_sha256",
            allow_nan=False,
        )
    for role, version in (
        ("jax_minimum", "0.4.38"),
        ("jax_maximum", "0.10.2"),
    ):
        _write_hashed(
            artifact_paths[role],
            {"jax_version": version, "cases": 2000, "failures": []},
            "sha256",
            indent=2,
        )
    _write_hashed(
        artifact_paths["memory"],
        {
            "peak_below_unbatched_multiplier_cube": True,
            "nuisance_refit_possible_from_result": False,
        },
        "sha256",
    )
    _write_hashed(
        artifact_paths["build"], {"passed": True}, "sha256"
    )
    artifact_paths["coverage"].parent.mkdir(parents=True, exist_ok=True)
    artifact_paths["coverage"].write_text(json.dumps(_coverage_report()), encoding="utf-8")
    _write_hashed(
        artifact_paths["evidence"],
        {"complete": True, "validation_level": "directional"},
        "report_sha256",
        indent=2,
    )
    assert blocking_reasons(manifest, tmp_path) == []
