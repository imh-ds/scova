import json
from hashlib import sha256
from pathlib import Path

import pytest

from benchmarks.calibrate_stage3_gates import calibrate
from benchmarks.summarize_stage3 import summarize
from scova.experimental import DiagnosticThresholds
from scripts.verify_stage3_shards import verify_shards


def _metrics() -> dict[str, float]:
    return {
        "min_group_ess": 200.0,
        "target_ess_ratio": 0.8,
        "max_influence_share": 0.1,
        "max_weight_concentration": 0.03,
        "min_propensity_q01": 0.1,
        "max_calibration_error": 0.02,
        "max_balance": 0.05,
        "crossfit_instability": 0.05,
    }


def _record(index: int, overlap: str, nuisance: str = "oracle") -> dict:
    alternative = {
        "refused": False,
        "gate_status": "warning",
        "gate_metrics": _metrics(),
        "uniform_coverage": index % 20 != 0,
        "naive_uniform_coverage": index % 5 != 0,
        "false_sign_certificate": False,
        "stability_covered": index % 20 != 0,
        "scientific_target_mean_error": 0.01 if index % 2 else -0.01,
        "scientific_pseudo_target_max_drift": 0.2,
    }
    null = {
        "refused": False,
        "gate_status": "warning",
        "gate_metrics": _metrics(),
        "false_sign_certificate": False,
    }
    return {
        "spec": {
            "n_groups": 2,
            "n": 500,
            "p": 5,
            "overlap": overlap,
            "outcome": "linear",
            "imbalance": "balanced",
            "error": "normal",
            "nuisance": nuisance,
        },
        "alternative": alternative,
        "null": null,
    }


def _release_spec() -> dict:
    return json.loads(Path("benchmarks/specs/stage3_release.json").read_text(encoding="utf-8"))


def _candidates() -> dict:
    return {
        "schema_version": 2,
        "frozen": True,
        "profiles": [
            {
                "name": "directional",
                "min_group_ess": 10,
                "min_target_ess_ratio": 0.1,
                "max_influence_share": 0.75,
                "max_weight_concentration": 0.2,
                "min_propensity_q01": 0.001,
                "max_calibration_error": 0.2,
                "max_balance": 0.5,
                "max_crossfit_instability": 0.5,
            }
        ],
    }


def _campaign(tier: str, seed_namespace: int, repetitions: int = 100) -> dict:
    records = [
        _record(index, overlap)
        for overlap in ("strong", "moderate")
        for index in range(repetitions)
    ]
    return {
        "tier": tier,
        "seed_namespace": seed_namespace,
        "validation_level": "directional",
        "specification_sha256": "specification",
        "threshold_artifact_sha256": "threshold",
        "records": records,
    }


def test_directional_calibration_is_cellwise_and_checksummed() -> None:
    specification = _release_spec()
    artifact = calibrate(
        [_campaign("calibration", specification["calibration_seed_namespace"])],
        _candidates(),
        specification,
    )
    assert artifact["validation_level"] == "directional"
    assert artifact["pass_profile"]["name"] == "directional"
    assert artifact["warning_floor_profile"]["name"] == "numerical-validity-floor"
    assert len(artifact["audit"][0]["cells"]) == 2
    thresholds = DiagnosticThresholds.from_calibration_artifact(artifact)
    assert thresholds.calibrated


def test_calibration_rejects_heldout_seeds_and_trivial_refusal() -> None:
    specification = _release_spec()
    with pytest.raises(ValueError, match="non-calibration"):
        calibrate(
            [_campaign("calibration", specification["validation_seed_namespace"])],
            _candidates(),
            specification,
        )
    candidates = _candidates()
    candidates["profiles"][0]["min_group_ess"] = 10_000
    failed = calibrate(
        [_campaign("calibration", specification["calibration_seed_namespace"])],
        candidates,
        specification,
    )
    assert failed["calibrated"] is False
    assert failed["selection_status"] == "no-profile-passed"
    assert failed["audit"]


def test_directional_calibration_can_lock_ranked_provisional_profile() -> None:
    specification = _release_spec()
    campaign = _campaign("calibration", specification["calibration_seed_namespace"])
    for record in campaign["records"]:
        record["alternative"]["uniform_coverage"] = False
        record["alternative"]["stability_covered"] = False
    artifact = calibrate(
        [campaign],
        _candidates(),
        specification,
        allow_provisional=True,
    )
    assert artifact["calibrated"] is True
    assert artifact["calibration_criteria_passed"] is False
    assert artifact["selection_status"] == "provisional-ranked-for-held-out-validation"
    assert artifact["pass_profile"]["name"] == "directional"


def test_directional_summary_requires_locked_threshold_and_all_cells(tmp_path: Path) -> None:
    specification = _release_spec()
    specification_bytes = Path("benchmarks/specs/stage3_release.json").read_bytes()
    campaign = _campaign("directional_validation", specification["validation_seed_namespace"], 200)
    campaign["specification_sha256"] = sha256(specification_bytes).hexdigest()
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    result = summarize([path], specification)
    assert result["validation_level"] == "directional"
    assert result["all_cells_passed"]
    campaign["threshold_artifact_sha256"] = None
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="locked threshold"):
        summarize([path], specification)


def test_directional_summary_serializes_degenerate_bias_as_null(tmp_path: Path) -> None:
    specification = _release_spec()
    campaign = _campaign(
        "directional_validation", specification["validation_seed_namespace"], 200
    )
    campaign["specification_sha256"] = sha256(
        Path("benchmarks/specs/stage3_release.json").read_bytes()
    ).hexdigest()
    for record in campaign["records"]:
        record["alternative"]["scientific_target_mean_error"] = 0.0
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")

    result = summarize([path], specification)

    assert result["all_cells_passed"] is False
    assert result["cells"][0]["standardized_absolute_bias"] is None
    assert result["cells"][0]["standardized_bias_status"] == (
        "undefined-zero-error-variance"
    )
    json.dumps(result, allow_nan=False)


def test_shard_verifier_checks_complete_provenance(tmp_path: Path) -> None:
    specification = {
        "protocol": "test-directional",
        "frozen": True,
        "validation_level": "directional",
        "calibration_seed_namespace": 1000,
        "tiers": {
            "calibration": {
                "cells": 2,
                "repetitions": 2,
                "bootstrap": 9,
                "cell_indices": [4, 7],
            }
        },
    }
    specification_path = tmp_path / "spec.json"
    specification_path.write_text(json.dumps(specification), encoding="utf-8")
    specification_hash = sha256(specification_path.read_bytes()).hexdigest()
    manifest = [
        {"source_cell_index": 4, "spec": {"cell": "a"}},
        {"source_cell_index": 7, "spec": {"cell": "b"}},
    ]
    paths = []
    for shard_index, source_index in enumerate((4, 7)):
        records = []
        for repetition in range(2):
            records.append(
                {
                    "source_cell_index": source_index,
                    "cell_id": shard_index,
                    "repetition": repetition,
                    "seed": 1000 + shard_index * 100_000 + repetition,
                    "spec": manifest[shard_index]["spec"],
                }
            )
        payload = {
            "shard_count": 2,
            "shard_index": shard_index,
            "git_commit": "abc123",
            "cell_manifest": manifest,
            "tier": "calibration",
            "seed_set": "calibration",
            "seed_namespace": 1000,
            "specification_sha256": specification_hash,
            "validation_level": "directional",
            "repetitions": 2,
            "bootstrap": 9,
            "threshold_artifact_sha256": None,
            "records": records,
        }
        path = tmp_path / f"shard-{shard_index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.with_suffix(".json.sha256").write_text(
            sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii"
        )
        paths.append(path)
    result = verify_shards(
        paths,
        specification_path=specification_path,
        tier="calibration",
        seed_set="calibration",
        threshold_path=None,
    )
    assert result["record_count"] == 4
    assert result["shard_count"] == 2
    paths[0].with_suffix(".json.sha256").write_text("bad", encoding="ascii")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_shards(
            paths,
            specification_path=specification_path,
            tier="calibration",
            seed_set="calibration",
            threshold_path=None,
        )
