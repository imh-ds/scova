import json
from copy import deepcopy
from pathlib import Path

import pytest

from scova.cf import (
    ApplicabilityClassification,
    CFValidationProtocol,
    assess_observational_applicability,
    observational_applicability_matrix,
)
from scova.cf.applicability import _matrix_from_dict
from scripts.audit_cf_observational_scope import coverage_inventory

V11_SPEC = "benchmarks/specs/cf_reference_v11.json"
MATRIX_PATH = Path("src/scova/cf/data/applicability_matrices.json")


def _matrix_values() -> dict:
    return deepcopy(json.loads(MATRIX_PATH.read_text(encoding="utf-8"))["matrices"][0])


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


def test_candidate_envelope_marks_unsupported_group_count_untested() -> None:
    assessment = assess_observational_applicability(
        n_groups=4, n_covariates=5, nuisance_strategy="adaptive"
    )
    assert assessment.classification is ApplicabilityClassification.UNTESTED
    assert "two- and three-group" in assessment.reason


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda values: values.pop("state"), "unexpected or missing"),
        (lambda values: values.__setitem__("mode", "randomized"), "provisional observational"),
        (
            lambda values: values["candidate_qualification_envelope"].__setitem__(
                "nuisance_strategy", "linear"
            ),
            "adaptive nuisances",
        ),
        (
            lambda values: values["candidate_qualification_envelope"].__setitem__("n_groups", [2]),
            "two/three groups",
        ),
        (
            lambda values: values["classifications"].__setitem__("in-scope", [{"entry_id": "bad"}]),
            "id and statement",
        ),
    ],
)
def test_matrix_schema_rejects_invalid_envelopes_and_entries(mutate, message: str) -> None:
    values = _matrix_values()
    mutate(values)
    with pytest.raises(ValueError, match=message):
        _matrix_from_dict(values)


def test_matrix_rejects_provisional_in_scope_and_duplicate_ids() -> None:
    in_scope = _matrix_values()
    in_scope["classifications"]["in-scope"] = [
        {
            "entry_id": "incorrect-scope",
            "statement": "Not allowed in a provisional matrix.",
            "evidence": {},
        }
    ]
    with pytest.raises(ValueError, match="cannot claim an in-scope"):
        _matrix_from_dict(in_scope)

    duplicate = _matrix_values()
    duplicate["classifications"]["untested"][0]["entry_id"] = duplicate["classifications"][
        "known-limitation"
    ][0]["entry_id"]
    with pytest.raises(ValueError, match="unique ids"):
        _matrix_from_dict(duplicate)


def test_coverage_inventory_is_bound_to_v11_and_reports_predictor_gap() -> None:
    inventory = coverage_inventory(CFValidationProtocol.load(V11_SPEC))
    assert inventory["matrix_id"] == "cf-observational-provisional-v1"
    assert inventory["protocol_id"].endswith("v11")
    assert "predictor-count-above-five" in inventory["untested"]
