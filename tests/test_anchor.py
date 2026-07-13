"""Stage 5A bounded-outcome anchor contracts."""

import numpy as np
import pytest

from scova import (
    AnchoredBoundsDeclaration,
    AnchoredBoundsResult,
    DesignDeclaration,
    OutcomeFreeDesignData,
    SCOVADesign,
)
from scova.anchor import bounded_pairwise_anchor, scaled_harmonic_overlap_and_gradient
from scova.experimental.gates import DiagnosticThresholds
from scova.simulate import generate_data

DEFAULT_BOUNDS = AnchoredBoundsDeclaration(-20, 20)


def thresholds() -> DiagnosticThresholds:
    return DiagnosticThresholds(
        version="stage5a-test", calibrated=True, artifact_sha256="test", min_group_ess_warning=1,
        min_group_ess_refuse=0, min_target_ess_ratio_warning=0, min_target_ess_ratio_refuse=0,
        max_influence_share_warning=1, max_influence_share_refuse=1,
        max_weight_concentration_warning=1, max_weight_concentration_refuse=1,
        min_propensity_q01_warning=1e-12, min_propensity_q01_refuse=1e-14,
        max_calibration_error_warning=1, max_calibration_error_refuse=1,
        max_balance_warning=1_000, max_balance_refuse=10_000,
        max_crossfit_instability_warning=1, max_crossfit_instability_refuse=1,
    )


def prepared(bounds: AnchoredBoundsDeclaration | None = DEFAULT_BOUNDS):
    simulation = generate_data("observational", n=300, seed=211)
    frame = simulation.data
    data = OutcomeFreeDesignData.from_arrays(
        frame.loc[:, ["x1", "x2", "x3"]].to_numpy(),
        frame["group"].tolist(),
        row_ids=range(len(frame)),
    )
    declaration = DesignDeclaration(
        group="group", covariates=("x1", "x2", "x3"), random_state=211, n_splits=2,
        lambdas=(0.0, 1.0), anchored_bounds=bounds,
    )
    engine = SCOVADesign(thresholds=thresholds())
    return engine, engine.prepare_design(data, declaration), frame["outcome"].to_numpy()


def test_declaration_validates_and_hashes_anchor() -> None:
    with pytest.raises(ValueError, match="finite"):
        AnchoredBoundsDeclaration(1, 1)
    first = DesignDeclaration("a", ("x",), anchored_bounds=AnchoredBoundsDeclaration(0, 1))
    second = DesignDeclaration("a", ("x",), anchored_bounds=AnchoredBoundsDeclaration(0, 2))
    assert first.declaration_hash != second.declaration_hash
    assert DesignDeclaration.from_dict(first.to_dict()) == first


def test_smooth_weight_is_bounded_and_endpoints_ordered() -> None:
    probability = np.array([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    omega, gradient = scaled_harmonic_overlap_and_gradient(probability, (0, 1))
    assert np.all((omega >= 0) & (omega <= 1))
    assert gradient.shape == probability.shape
    result = bounded_pairwise_anchor(
        groups=("a", "b"), group_codes=np.array([0, 1, 0]), outcomes=np.array([0.2, 0.8, 0.4]),
        propensity=probability, outcome_predictions=np.array([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]),
        active_codes=(0, 1), outcome_lower=0, outcome_upper=1, confidence_level=0.95,
    )
    assert result.lower_endpoint <= result.upper_endpoint


def test_endpoint_algebra_and_gradient_checks() -> None:
    probability = np.array([[0.5, 0.5], [0.7, 0.3], [0.4, 0.6]])
    predictions = np.array([[0.3, 0.7], [0.2, 0.6], [0.4, 0.5]])
    common = dict(
        group_codes=np.array([0, 1, 0]),
        outcomes=np.array([0.2, 0.8, 0.4]),
        propensity=probability,
        outcome_predictions=predictions,
        active_codes=(0, 1),
        confidence_level=0.95,
    )
    narrow = bounded_pairwise_anchor(groups=("a", "b"), outcome_lower=0, outcome_upper=1, **common)
    wide = bounded_pairwise_anchor(groups=("a", "b"), outcome_lower=-1, outcome_upper=2, **common)
    reversed_result = bounded_pairwise_anchor(
        groups=("b", "a"),
        group_codes=common["group_codes"],
        outcomes=common["outcomes"],
        propensity=probability,
        outcome_predictions=predictions,
        active_codes=(1, 0),
        outcome_lower=0,
        outcome_upper=1,
        confidence_level=0.95,
    )
    wide_width = wide.upper_endpoint - wide.lower_endpoint
    narrow_width = narrow.upper_endpoint - narrow.lower_endpoint
    assert wide_width >= narrow_width
    np.testing.assert_allclose(reversed_result.lower_endpoint, -narrow.upper_endpoint)
    np.testing.assert_allclose(reversed_result.upper_endpoint, -narrow.lower_endpoint)
    omega, gradient = scaled_harmonic_overlap_and_gradient(probability, (0, 1))
    step = 1e-6
    shifted = probability.copy()
    shifted[:, 0] += step
    shifted[:, 1] -= step
    shifted_omega, _ = scaled_harmonic_overlap_and_gradient(shifted, (0, 1))
    np.testing.assert_allclose(
        (shifted_omega - omega) / step, gradient[:, 0] - gradient[:, 1], rtol=1e-5, atol=5e-6
    )


def test_locked_anchor_round_trip_and_refusals(tmp_path) -> None:
    engine, design, outcomes = prepared()
    ids = design.lock.estimation_row_ids
    result = engine.analyze_anchored_bounds(design, outcomes[list(ids)], row_ids=ids)
    assert result.verdict == "interval-only"
    assert result.contrasts
    assert all(item.lower_endpoint <= item.upper_endpoint for item in result.contrasts)
    path = tmp_path / "anchor.npz"
    result.save(path)
    assert AnchoredBoundsResult.load(path).report() == result.report()
    refused = engine.analyze_anchored_bounds(design, np.full(len(ids), 100.0), row_ids=ids)
    assert refused.verdict == "refused"
    _, missing, missing_outcomes = prepared(None)
    missing_result = engine.analyze_anchored_bounds(
        missing,
        missing_outcomes[list(missing.lock.estimation_row_ids)],
        row_ids=missing.lock.estimation_row_ids,
    )
    assert missing_result.verdict == "refused"


def test_anchor_requires_locked_row_alignment() -> None:
    engine, design, outcomes = prepared()
    ids = design.lock.estimation_row_ids
    with pytest.raises(ValueError, match="exactly match"):
        engine.analyze_anchored_bounds(design, outcomes[list(ids[:-1])], row_ids=ids[:-1])
