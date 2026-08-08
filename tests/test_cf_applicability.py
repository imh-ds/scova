from scova.cf import (
    ApplicabilityClassification,
    CFValidationProtocol,
    assess_observational_applicability,
    observational_applicability_matrix,
)
from scripts.audit_cf_observational_scope import coverage_inventory

V11_SPEC = "benchmarks/specs/cf_reference_v11.json"


def test_provisional_observational_matrix_is_complete_and_has_no_scope_claim() -> None:
    matrix = observational_applicability_matrix()
    assert matrix.matrix_id == "cf-observational-provisional-v1"
    assert matrix.group_counts == (2, 3)
    assert matrix.maximum_covariate_count == 5
    assert not matrix.classifications[ApplicabilityClassification.IN_SCOPE]
    assert matrix.classifications[ApplicabilityClassification.KNOWN_LIMITATION]
    assert matrix.classifications[ApplicabilityClassification.UNTESTED]


def test_observational_matrix_marks_predictor_overreach_and_nonadaptive_strategies() -> None:
    overreach = assess_observational_applicability(
        n_groups=2, n_covariates=6, nuisance_strategy="adaptive"
    )
    assert overreach.classification is ApplicabilityClassification.UNTESTED
    assert "predictor scope" in overreach.reason

    linear = assess_observational_applicability(
        n_groups=2, n_covariates=5, nuisance_strategy="linear"
    )
    assert linear.classification is ApplicabilityClassification.KNOWN_LIMITATION
    assert "contract-ineligible" in linear.reason


def test_candidate_envelope_stays_untested_without_a_promoted_profile() -> None:
    assessment = assess_observational_applicability(
        n_groups=3, n_covariates=5, nuisance_strategy="adaptive"
    )
    assert assessment.classification is ApplicabilityClassification.UNTESTED
    assert "no observational profile is promoted" in assessment.reason


def test_coverage_inventory_is_bound_to_v11_and_reports_predictor_gap() -> None:
    inventory = coverage_inventory(CFValidationProtocol.load(V11_SPEC))
    assert inventory["matrix_id"] == "cf-observational-provisional-v1"
    assert inventory["protocol_id"].endswith("v11")
    assert "predictor-count-above-five" in inventory["untested"]
