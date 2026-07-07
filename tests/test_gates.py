import json
from hashlib import sha256

import pytest

from scova.experimental.gates import (
    DiagnosticThresholds,
    GateDecision,
    GateStatus,
    evaluate_path_gates,
)


def good_values(**overrides):
    values = {
        "min_group_ess": 200.0,
        "target_ess_ratio": 0.8,
        "max_influence_share": 0.1,
        "max_weight_concentration": 0.03,
        "min_propensity_q01": 0.1,
        "max_calibration_error": 0.02,
        "max_balance": 0.05,
        "crossfit_instability": 0.05,
        "numerical_valid": True,
    }
    values.update(overrides)
    return values


def test_provisional_thresholds_cannot_pass() -> None:
    thresholds = DiagnosticThresholds()
    decision = evaluate_path_gates(**good_values(), thresholds=thresholds)
    assert decision.status is GateStatus.WARNING
    assert "provisional" in decision.reasons[-1]
    assert GateDecision.from_dict(decision.to_dict()) == decision


def test_calibrated_pass_warning_and_refusal() -> None:
    artifact = {
        "version": "locked-v1",
        "calibrated": True,
        "pass_profile": {
            "min_group_ess": 50,
            "min_target_ess_ratio": 0.25,
            "max_influence_share": 0.4,
            "max_weight_concentration": 0.1,
            "min_propensity_q01": 0.005,
            "max_calibration_error": 0.1,
            "max_balance": 0.25,
            "max_crossfit_instability": 0.3,
        },
        "warning_floor_profile": {
            "min_group_ess": 25,
            "min_target_ess_ratio": 0.15,
            "max_influence_share": 0.6,
            "max_weight_concentration": 0.15,
            "min_propensity_q01": 0.0025,
            "max_calibration_error": 0.15,
            "max_balance": 0.35,
            "max_crossfit_instability": 0.4,
        },
    }
    artifact["sha256"] = sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()
    thresholds = DiagnosticThresholds.from_calibration_artifact(artifact)
    assert evaluate_path_gates(**good_values(), thresholds=thresholds).status is GateStatus.PASS
    warning = evaluate_path_gates(
        **good_values(min_group_ess=40), thresholds=thresholds
    )
    assert warning.status is GateStatus.WARNING
    refusal = evaluate_path_gates(
        **good_values(max_influence_share=0.8), thresholds=thresholds
    )
    assert refusal.status is GateStatus.REFUSE
    numerical = evaluate_path_gates(
        **good_values(numerical_valid=False), thresholds=thresholds
    )
    assert numerical.status is GateStatus.REFUSE


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="group ESS"):
        DiagnosticThresholds(min_group_ess_warning=5, min_group_ess_refuse=10)
    with pytest.raises(ValueError, match="balance"):
        DiagnosticThresholds(max_balance_warning=0.5, max_balance_refuse=0.2)
    with pytest.raises(ValueError, match="propensity"):
        DiagnosticThresholds(min_propensity_q01_refuse=0)
    with pytest.raises(ValueError, match="not calibrated"):
        DiagnosticThresholds.from_calibration_artifact({"calibrated": False})
    with pytest.raises(ValueError, match="artifact hash"):
        DiagnosticThresholds(calibrated=True)
    artifact = {
        "version": "bad",
        "calibrated": True,
        "sha256": "bad",
        "pass_profile": {},
        "warning_floor_profile": {},
    }
    with pytest.raises(ValueError, match="checksum"):
        DiagnosticThresholds.from_calibration_artifact(artifact)


def test_user_thresholds_cannot_weaken_locked_defaults() -> None:
    baseline = DiagnosticThresholds()
    stricter = DiagnosticThresholds(
        min_group_ess_warning=60,
        min_group_ess_refuse=20,
        max_balance_warning=0.15,
        max_balance_refuse=0.4,
    )
    stricter.assert_at_least_as_strict_as(baseline)
    weaker = DiagnosticThresholds(max_balance_warning=0.3)
    with pytest.raises(ValueError, match="cannot weaken"):
        weaker.assert_at_least_as_strict_as(baseline)
