import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from benchmarks.stage3_campaign import (
    primary_cells,
    robustness_cells,
    run_campaign,
    select_cells,
)
from scova import SCOVADeclaration
from scova.experimental import PathDeclaration
from scova.experimental.nuisance import (
    ConvexStackingClassifier,
    ConvexStackingRegressor,
    assess_crossfit_stability,
    make_learner_profile,
)
from scova.experimental.simulation import StabilizationSpec, generate_stabilization_data


def base_spec(**overrides):
    values = {
        "n_groups": 4,
        "n": 240,
        "p": 5,
        "overlap": "moderate",
        "outcome": "linear",
        "imbalance": "balanced",
        "error": "normal",
        "nuisance": "oracle",
    }
    values.update(overrides)
    return StabilizationSpec(**values)


@pytest.mark.parametrize("outcome", ["linear", "nonlinear", "threshold", "nonmonotone"])
@pytest.mark.parametrize("error", ["normal", "heteroskedastic", "heavy_tailed"])
def test_stabilization_dgps_and_targets(outcome, error) -> None:
    generated = generate_stabilization_data(base_spec(outcome=outcome, error=error), seed=14)
    assert generated.data.shape == (240, 7)
    np.testing.assert_allclose(generated.propensity.sum(axis=1), 1)
    truth = generated.target_path(np.array([0.0, 1.0]), np.array([1, 0, 0, -1]))
    pseudo = generated.target_path(
        np.array([0.0, 1.0]), np.array([1, 0, 0, -1]), pseudo=True
    )
    assert truth.shape == (2,)
    assert pseudo.shape == (2,)


@pytest.mark.parametrize(
    "regime",
    [
        "oracle",
        "gps_correct_outcome_wrong",
        "outcome_correct_gps_wrong",
        "both_wrong",
        "correct_both",
        "flexible",
        "deliberately_inadequate",
    ],
)
def test_nuisance_regime_routing(regime) -> None:
    generated = generate_stabilization_data(base_spec(), seed=5)
    nuisance = generated.nuisance_predictions(regime)
    if regime in ("correct_both", "flexible", "deliberately_inadequate"):
        assert nuisance is None
    else:
        assert nuisance is not None


@pytest.mark.parametrize("name", ["linear", "nonlinear", "ensemble", "deliberately_inadequate"])
def test_learner_profiles(name) -> None:
    profile = make_learner_profile(name, random_state=2)
    assert profile.estimator().propensity_model is profile.propensity_model
    calibrated = make_learner_profile("linear", calibration="sigmoid")
    assert calibrated.calibration == "sigmoid"
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner_profile("missing")
    with pytest.raises(ValueError, match="calibration"):
        make_learner_profile("linear", calibration="temperature")


def test_declared_ensemble_uses_inner_cv_convex_weights() -> None:
    generated = generate_stabilization_data(base_spec(n=180), seed=19)
    x = generated.data[[f"x{index}" for index in range(1, 6)]].to_numpy()
    group = generated.data["group"].to_numpy()
    outcome = generated.data["outcome"].to_numpy()
    profile = make_learner_profile("ensemble", random_state=3)
    assert isinstance(profile.propensity_model, ConvexStackingClassifier)
    assert isinstance(profile.outcome_model, ConvexStackingRegressor)
    classifier = profile.propensity_model.fit(x, group)
    regressor = profile.outcome_model.fit(x, outcome)
    assert np.all(classifier.weights_ >= 0)
    assert np.all(regressor.weights_ >= 0)
    np.testing.assert_allclose(classifier.weights_.sum(), 1)
    np.testing.assert_allclose(regressor.weights_.sum(), 1)
    np.testing.assert_allclose(classifier.predict_proba(x).sum(axis=1), 1)
    assert classifier.predict(x).shape == (180,)
    assert regressor.predict(x).shape == (180,)


def test_repeated_crossfit_stability() -> None:
    generated = generate_stabilization_data(base_spec(n=300), seed=7)
    base = SCOVADeclaration(
        "outcome", "group", tuple(f"x{i}" for i in range(1, 6)), n_splits=3
    )
    declaration = PathDeclaration(base, lambdas=(0.0, 0.5, 1.0))
    stability = assess_crossfit_stability(
        generated.data,
        declaration,
        make_learner_profile("linear"),
        seeds=(11, 13),
    )
    assert stability.estimate_paths.shape[0] == 2
    assert stability.maximum_standardized_range >= 0
    with pytest.raises(ValueError, match="distinct"):
        assess_crossfit_stability(
            generated.data, declaration, make_learner_profile("linear"), seeds=(1, 1)
        )


def test_campaign_cell_counts_and_selection() -> None:
    specification = {
        "primary_factors": {
            "n_groups": [2, 4, 8],
            "n": [500, 2000],
            "p": [5, 20],
            "overlap": ["strong", "moderate", "weak", "near_violation"],
            "outcome": ["linear", "nonlinear", "threshold"],
            "imbalance": ["balanced", "rare"],
        }
    }
    primary = primary_cells(specification)
    assert len(primary) == 288
    robustness = robustness_cells()
    assert len(robustness) == 96
    assert len(set(robustness)) == 96
    assert len(select_cells(primary, 24)) == 24
    assert select_cells(primary, 999) == primary


def test_frozen_calibration_campaign_smoke(tmp_path) -> None:
    output = tmp_path / "calibration.json"
    run_campaign(
        tier="calibration",
        specification_path=Path("benchmarks/specs/stage3_release.json"),
        output=output,
        shard_index=0,
        shard_count=24,
        repetitions_override=1,
        bootstrap_override=9,
        seed_set="calibration",
        threshold_path=None,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["tier"] == "calibration"
    assert payload["validation_level"] == "directional"
    assert payload["threshold_version"] == "stage3-calibration-ungated-v1"
    assert len(payload["records"]) == 1


def test_local_pilot_campaign_smoke(tmp_path) -> None:
    profile = {
        "name": "pilot",
        "min_group_ess": 10,
        "min_target_ess_ratio": 0.1,
        "max_influence_share": 0.75,
        "max_weight_concentration": 0.2,
        "min_propensity_q01": 0.001,
        "max_calibration_error": 0.2,
        "max_balance": 0.5,
        "max_crossfit_instability": 0.5,
    }
    artifact = {
        "version": "pilot-test",
        "calibrated": True,
        "pass_profile": profile,
        "warning_floor_profile": profile,
    }
    artifact["sha256"] = sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps(artifact), encoding="utf-8")
    output = tmp_path / "pilot.json"
    run_campaign(
        tier="local_validation_pilot",
        specification_path=Path("benchmarks/specs/stage3_local_pilot.json"),
        output=output,
        shard_index=0,
        shard_count=12,
        repetitions_override=1,
        bootstrap_override=9,
        seed_set="pilot",
        threshold_path=threshold_path,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation_level"] == "directional-pilot"
    assert payload["seed_namespace"] == 83_000_000
    assert payload["threshold_artifact_sha256"] == artifact["sha256"]


def test_stabilization_input_failures() -> None:
    with pytest.raises(ValueError, match="K"):
        generate_stabilization_data(replace(base_spec(), n_groups=3), seed=1)
    with pytest.raises(ValueError, match="n >= 100"):
        generate_stabilization_data(replace(base_spec(), n=20), seed=1)
