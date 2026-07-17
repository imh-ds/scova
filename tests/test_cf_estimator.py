from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from scova import SCOVA, ContrastSpec, NuisancePredictions, SCOVADeclaration
from scova.cf import (
    SCOVACF,
    AnalysisMode,
    ClaimClass,
    EstimatedAssignment,
    KnownAssignment,
    SCOVACFDeclaration,
    SCOVACFNuisancePredictions,
    SCOVACFRefusal,
    SCOVACFResult,
    SupportPolicy,
    SupportStatus,
)
from scova.simulate import generate_data


def declaration(
    *,
    mode: AnalysisMode = AnalysisMode.RANDOMIZED,
    sensitivity_analysis: str | None = None,
) -> SCOVACFDeclaration:
    assignment = (
        KnownAssignment(
            probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))
        )
        if mode is AnalysisMode.RANDOMIZED
        else EstimatedAssignment(nuisance_strategy="linear")
    )
    return SCOVACFDeclaration(
        outcome="outcome",
        group="group",
        covariates=("x1", "x2", "x3"),
        mode=mode,
        scientific_question="What are the standardized population means?",
        eligibility="All study units",
        target_population="Eligible study-unit population",
        group_definitions=(
            ("g0", "group zero"),
            ("g1", "group one"),
            ("g2", "group two"),
        ),
        outcome_time="follow-up",
        outcome_units="points",
        covariate_rationales=(
            ("x1", "baseline prognostic factor"),
            ("x2", "baseline prognostic factor"),
            ("x3", "baseline prognostic factor"),
        ),
        assignment=assignment,
        outcome_nuisance_strategy="linear",
        n_splits=3,
        random_state=17,
        contrasts=(
            ContrastSpec("g0 - g1", (("g0", 1.0), ("g1", -1.0))),
            ContrastSpec("g2 - g1", (("g2", 1.0), ("g1", -1.0))),
        ),
        sensitivity_analysis=sensitivity_analysis,
    )


def oracle_result(mode: AnalysisMode = AnalysisMode.RANDOMIZED) -> SCOVACFResult:
    simulation = generate_data(
        "randomized" if mode is AnalysisMode.RANDOMIZED else "observational",
        n=360,
        seed=12,
    )
    nuisance = SCOVACFNuisancePredictions(
        outcome_regression=simulation.outcome_regression,
        propensity=None if mode is AnalysisMode.RANDOMIZED else simulation.propensity,
        group_labels=simulation.group_labels,
    )
    result = SCOVACF().analyze(
        simulation.data,
        declaration(mode=mode),
        nuisance_predictions=nuisance,
    )
    assert isinstance(result, SCOVACFResult)
    return result


def test_randomized_cf_oracle_matches_base_numerical_engine() -> None:
    simulation = generate_data("randomized", n=360, seed=12)
    cf = SCOVACF().analyze(
        simulation.data,
        declaration(),
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels
        ),
    )
    assert isinstance(cf, SCOVACFResult)
    base = SCOVA().fit(
        simulation.data,
        SCOVADeclaration(
            "outcome", "group", ("x1", "x2", "x3"), n_splits=3, random_state=17
        ),
        nuisance_predictions=NuisancePredictions(
            simulation.propensity,
            simulation.outcome_regression,
            simulation.group_labels,
        ),
    )
    np.testing.assert_allclose(cf.group_means, base.group_means)
    np.testing.assert_allclose(cf.influence_values, base.influence_values)
    np.testing.assert_allclose(cf.covariance, base.covariance)
    assert cf.mode is AnalysisMode.RANDOMIZED
    assert cf.claim_class is ClaimClass.RANDOMIZATION_SUPPORTED
    assert cf.status.support is SupportStatus.UNSTABLE
    assert cf.status.confirmatory is False
    np.testing.assert_allclose(cf.propensity_predictions, 1 / 3)


def test_cf_cross_fit_uses_known_assignment_and_produces_benchmarks() -> None:
    simulation = generate_data("randomized", n=360, seed=9)
    result = SCOVACF().analyze(simulation.data, declaration())
    assert isinstance(result, SCOVACFResult)
    np.testing.assert_allclose(result.propensity_predictions, 1 / 3)
    assert result.nuisance_metadata["propensity_model"] == "known-design"
    assert result.benchmarks["unadjusted"]["name"] == "unadjusted-group-means"
    assert result.benchmarks["lin_interacted"]["name"] == "lin-fully-interacted"
    assert result.evidence_card["scientific_boundary"].startswith(
        "Population counterfactual means"
    )


def test_associational_and_observational_claim_gates() -> None:
    associational = oracle_result(AnalysisMode.STANDARDIZED_ASSOCIATIONAL)
    assert associational.claim_class is ClaimClass.ASSOCIATIONAL
    assert associational.evidence_card["claim_class"] == "associational"
    observational = oracle_result(AnalysisMode.OBSERVATIONAL_CAUSAL)
    assert observational.claim_class is ClaimClass.ASSUMPTION_DEPENDENT_CAUSAL
    assert observational.status.code == "limited/required-sensitivity-analysis"
    assert observational.status.confirmatory is False


def test_labelled_inference_and_cf_artifact_round_trip(tmp_path: Path) -> None:
    result = oracle_result()
    inference = result.infer(n_bootstrap=99, random_state=5)
    exported = inference.to_dict()
    assert exported["mode"] == "randomized"
    assert exported["claim_class"] == "randomization-supported"
    assert exported["interval_type"] == "simultaneous-max-t"
    assert exported["confirmatory"] is False
    path = tmp_path / "analysis.scova-cf"
    result.save(path)
    loaded = SCOVACFResult.load(path)
    assert loaded.declaration_hash == result.declaration_hash
    assert loaded.design_lock == result.design_lock
    assert loaded.evidence_card == result.evidence_card
    np.testing.assert_allclose(loaded.group_means, result.group_means)
    assert set(loaded.contrasts) == set(result.contrasts)

    base_path = tmp_path / "base.scova"
    simulation = generate_data("randomized", n=120, seed=3)
    base = SCOVA().fit(
        simulation.data,
        SCOVADeclaration("outcome", "group", ("x1", "x2", "x3"), n_splits=3),
        nuisance_predictions=NuisancePredictions(
            simulation.propensity,
            simulation.outcome_regression,
            simulation.group_labels,
        ),
    )
    base.save(base_path)
    with pytest.raises(ValueError, match="not a SCOVA-CF"):
        SCOVACFResult.load(base_path)


def test_typed_refusals_and_no_individual_counterfactual_api() -> None:
    simulation = generate_data("randomized", n=180, seed=7)
    post_treatment = replace(declaration(), post_treatment_covariates=("x1",))
    refused = SCOVACF().analyze(simulation.data, post_treatment)
    assert isinstance(refused, SCOVACFRefusal)
    assert refused.status.code == "refused/post-treatment-covariate"

    missing = simulation.data.copy()
    missing.loc[0, "outcome"] = np.nan
    limited = SCOVACF().analyze(missing, declaration())
    assert isinstance(limited, SCOVACFRefusal)
    assert limited.status.code == "limited/missing-outcomes"

    assert not hasattr(SCOVACF, "predict_potential_outcomes")
    assert not hasattr(SCOVACFResult, "paired_test")


def test_design_lock_and_folds_are_outcome_blind() -> None:
    simulation = generate_data("randomized", n=180, seed=14)
    first = SCOVACF().analyze(
        simulation.data,
        declaration(),
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels
        ),
    )
    changed = simulation.data.copy()
    changed["outcome"] = changed["outcome"] * -100 + 50
    second = SCOVACF().analyze(
        changed,
        declaration(),
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels
        ),
    )
    assert isinstance(first, SCOVACFResult)
    assert isinstance(second, SCOVACFResult)
    assert first.design_lock == second.design_lock
    np.testing.assert_array_equal(first.fold_assignments, second.fold_assignments)


def test_stratum_specific_known_probabilities_are_used() -> None:
    simulation = generate_data("randomized", n=180, seed=15)
    data = simulation.data.copy()
    data["block"] = np.where(data["x1"] >= 0, "positive", "negative")
    assignment = KnownAssignment(
        stratum_column="block",
        stratum_probabilities=(
            ("negative", (("g0", 0.5), ("g1", 0.25), ("g2", 0.25))),
            ("positive", (("g0", 0.2), ("g1", 0.3), ("g2", 0.5))),
        ),
    )
    declared = replace(declaration(), assignment=assignment)
    result = SCOVACF().analyze(
        data,
        declared,
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels
        ),
    )
    assert isinstance(result, SCOVACFResult)
    negative = data["block"].to_numpy() == "negative"
    np.testing.assert_allclose(result.propensity_predictions[negative, 0], 0.5)
    np.testing.assert_allclose(result.propensity_predictions[~negative, 2], 0.5)


def test_singular_omnibus_is_refused_without_hiding_other_estimates() -> None:
    simulation = generate_data("randomized", n=180, seed=8)
    data = simulation.data.copy()
    data["outcome"] = 2.0
    predictions = np.full_like(simulation.outcome_regression, 2.0)
    result = SCOVACF().analyze(
        data,
        declaration(),
        nuisance_predictions=SCOVACFNuisancePredictions(
            predictions, simulation.group_labels
        ),
    )
    assert isinstance(result, SCOVACFResult)
    assert result.omnibus.status_code == "refused/singular-omnibus"
    np.testing.assert_allclose(result.group_means, 2.0)


def test_estimator_input_refusals_are_typed() -> None:
    simulation = generate_data("randomized", n=180, seed=18)

    missing_column = SCOVACF().analyze(
        simulation.data.drop(columns="x3"), declaration()
    )
    assert isinstance(missing_column, SCOVACFRefusal)
    assert missing_column.status.code == "refused/missing-column"

    bad_covariate = simulation.data.copy()
    bad_covariate["x1"] = bad_covariate["x1"].astype(object)
    bad_covariate.loc[0, "x1"] = "not numeric"
    invalid_data = SCOVACF().analyze(bad_covariate, declaration())
    assert isinstance(invalid_data, SCOVACFRefusal)
    assert invalid_data.status.code == "refused/invalid-analysis-data"

    unexpected_group = simulation.data.copy()
    unexpected_group.loc[0, "group"] = "unexpected"
    invalid_groups = SCOVACF().analyze(unexpected_group, declaration())
    assert isinstance(invalid_groups, SCOVACFRefusal)
    assert invalid_groups.status.code == "refused/invalid-groups"

    bad_assignment = replace(
        declaration(),
        assignment=KnownAssignment(
            probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("other", 1 / 3))
        ),
    )
    invalid_assignment = SCOVACF().analyze(simulation.data, bad_assignment)
    assert isinstance(invalid_assignment, SCOVACFRefusal)
    assert invalid_assignment.status.code == "refused/invalid-assignment-mechanism"

    invalid_contrast = replace(
        declaration(),
        contrasts=(ContrastSpec("unknown", (("g0", 1), ("other", -1))),),
    )
    contrast_refusal = SCOVACF().analyze(simulation.data, invalid_contrast)
    assert isinstance(contrast_refusal, SCOVACFRefusal)
    assert contrast_refusal.status.code == "refused/invalid-contrast"

    g2_indices = simulation.data.index[simulation.data["group"] == "g2"]
    tiny = simulation.data.drop(index=g2_indices[2:]).reset_index(drop=True)
    small_sample = SCOVACF().analyze(tiny, declaration())
    assert isinstance(small_sample, SCOVACFRefusal)
    assert small_sample.status.code == "limited/small-sample-restricted-library"


def test_nuisance_prediction_and_policy_refusals() -> None:
    simulation = generate_data("observational", n=180, seed=19)
    associational = declaration(mode=AnalysisMode.STANDARDIZED_ASSOCIATIONAL)
    wrong_labels = SCOVACF().analyze(
        simulation.data,
        associational,
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, ("g0", "g2", "g1"), simulation.propensity
        ),
    )
    assert isinstance(wrong_labels, SCOVACFRefusal)
    assert wrong_labels.status.code == "refused/nuisance-labels"

    missing_propensity = SCOVACF().analyze(
        simulation.data,
        associational,
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels
        ),
    )
    assert isinstance(missing_propensity, SCOVACFRefusal)
    assert missing_propensity.status.code == "refused/missing-propensity"

    invalid_propensity = simulation.propensity.copy()
    invalid_propensity[0, 0] = 0
    positivity = SCOVACF().analyze(
        simulation.data,
        associational,
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels, invalid_propensity
        ),
    )
    assert isinstance(positivity, SCOVACFRefusal)
    assert positivity.status.code == "refused/positivity"

    bad_outcome_predictions = SCOVACF().analyze(
        simulation.data,
        associational,
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression[:, :2],
            simulation.group_labels,
            simulation.propensity,
        ),
    )
    assert isinstance(bad_outcome_predictions, SCOVACFRefusal)
    assert bad_outcome_predictions.status.code == "refused/nuisance-predictions"

    incompatible = replace(
        associational,
        assignment=EstimatedAssignment(nuisance_strategy="adaptive"),
    )
    incompatible_result = SCOVACF().analyze(simulation.data, incompatible)
    assert isinstance(incompatible_result, SCOVACFRefusal)
    assert incompatible_result.status.code == "refused/incompatible-nuisance-policy"

    custom = replace(
        associational,
        assignment=EstimatedAssignment(nuisance_strategy="custom"),
        outcome_nuisance_strategy="custom",
    )
    missing_custom = SCOVACF().analyze(simulation.data, custom)
    assert isinstance(missing_custom, SCOVACFRefusal)
    assert missing_custom.status.code == "refused/missing-custom-learner"
    with pytest.raises(ValueError, match="supplied together"):
        SCOVACF(outcome_model=Ridge())
    supplied_but_undeclared = SCOVACF(
        propensity_model=LogisticRegression(max_iter=100), outcome_model=Ridge()
    ).analyze(simulation.data, associational)
    assert isinstance(supplied_but_undeclared, SCOVACFRefusal)
    assert supplied_but_undeclared.status.code == "refused/incompatible-nuisance-policy"


def test_support_warning_details_and_rank_limited_benchmark() -> None:
    simulation = generate_data("weak_overlap", n=180, seed=20)
    data = simulation.data.copy()
    data["x2"] = data["x1"]
    strict_policy = SupportPolicy(
        min_group_count=1000,
        min_ess_ratio=0.99,
        max_normalized_weight=0.001,
        max_top_one_percent_weight_share=0.001,
    )
    declared = replace(
        declaration(mode=AnalysisMode.STANDARDIZED_ASSOCIATIONAL),
        support_policy=strict_policy,
    )
    result = SCOVACF().analyze(
        data,
        declared,
        nuisance_predictions=SCOVACFNuisancePredictions(
            simulation.outcome_regression, simulation.group_labels, simulation.propensity
        ),
    )
    assert isinstance(result, SCOVACFResult)
    assert "count" in result.status.reason
    assert "ESS ratio" in result.status.reason
    assert "maximum normalized weight" in result.status.reason
    assert "top-one-percent" in result.status.reason
    assert result.benchmarks["lin_interacted"]["status"] == (
        "limited/rank-deficient-benchmark"
    )


def test_result_validation_branches() -> None:
    result = oracle_result()
    assert result.group_confidence_intervals().shape == (3, 2)
    with pytest.raises(ValueError, match="confidence_level"):
        result.group_confidence_intervals(1.0)
    with pytest.raises(ValueError, match="unknown groups"):
        result.contrast({"unknown": 1, "g0": -1}, name="bad")
    with pytest.raises(ValueError, match="3 weights"):
        result.contrast([1, -1], name="bad")
    with pytest.raises(ValueError, match="finite"):
        result.contrast([np.nan, 0, 0], name="bad")
    with pytest.raises(ValueError, match="sum to zero"):
        result.contrast([1, 1, 0], name="bad")
    with pytest.raises(ValueError, match="nonzero"):
        result.contrast([0, 0, 0], name="bad")
    with pytest.raises(ValueError, match="name"):
        result.contrast([1, -1, 0], name="")
    with pytest.raises(ValueError, match="confidence_level"):
        result.contrast([1, -1, 0], name="bad", confidence_level=0)
    with pytest.raises(ValueError, match="unique"):
        result.infer(("g0 - g1", "g0 - g1"), n_bootstrap=10)
    with pytest.raises(ValueError, match="Unknown"):
        result.infer(("unknown",), n_bootstrap=10)
