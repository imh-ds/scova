import json
from hashlib import sha256
from pathlib import Path

import pytest

from benchmarks.calibrate_stage3_gates import calibrate
from benchmarks.summarize_stage3 import summarize
from scova.experimental import DiagnosticThresholds


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
    with pytest.raises(RuntimeError, match="no candidate"):
        calibrate(
            [_campaign("calibration", specification["calibration_seed_namespace"])],
            candidates,
            specification,
        )


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
