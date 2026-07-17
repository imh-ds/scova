from dataclasses import FrozenInstanceError, replace

import pytest

from scova import SCOVA, ContrastSpec, SCOVADeclaration
from scova.cf import (
    AnalysisMode,
    ClaimClass,
    DeclarationAmendment,
    EstimatedAssignment,
    KnownAssignment,
    SCOVACFDeclaration,
    SupportPolicy,
)


def randomized_declaration(**updates: object) -> SCOVACFDeclaration:
    values: dict[str, object] = {
        "outcome": "outcome",
        "group": "group",
        "covariates": ("x1", "x2", "x3"),
        "mode": AnalysisMode.RANDOMIZED,
        "scientific_question": "What would the target-population mean be under each group?",
        "eligibility": "All generated study units",
        "target_population": "Eligible study-unit population",
        "group_definitions": (
            ("g0", "randomized group zero"),
            ("g1", "randomized group one"),
            ("g2", "randomized group two"),
        ),
        "outcome_time": "end of follow-up",
        "outcome_units": "points",
        "covariate_rationales": (
            ("x1", "baseline prognostic factor"),
            ("x2", "baseline prognostic factor"),
            ("x3", "baseline prognostic factor"),
        ),
        "assignment": KnownAssignment(
            probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))
        ),
        "contrasts": (
            ContrastSpec("g0 - g1", (("g0", 1.0), ("g1", -1.0))),
        ),
        "n_splits": 3,
        "outcome_nuisance_strategy": "linear",
    }
    values.update(updates)
    return SCOVACFDeclaration(**values)  # type: ignore[arg-type]


def test_cf_is_additive_and_declaration_is_governed() -> None:
    declaration = randomized_declaration()
    assert declaration.mode is AnalysisMode.RANDOMIZED
    assert declaration.claim_class is ClaimClass.RANDOMIZATION_SUPPORTED
    assert declaration.to_dict()["claim_class"] == "randomization-supported"
    assert declaration.declaration_hash == randomized_declaration().declaration_hash
    with pytest.raises(FrozenInstanceError):
        declaration.mode = AnalysisMode.STANDARDIZED_ASSOCIATIONAL  # type: ignore[misc]

    # The existing product API remains independently constructible.
    assert SCOVA is not None
    assert SCOVADeclaration("y", "g", ("x",)).interpretation == "descriptive"


def test_amendments_are_hash_visible_and_auditable() -> None:
    amendment = DeclarationAmendment(
        timestamp="2026-07-17T12:00:00-07:00",
        reason="Corrected a prespecified unit label",
        changes=(("outcome_units", "standardized points"),),
    )
    amended = randomized_declaration(amendments=(amendment,))
    assert amended.declaration_hash != randomized_declaration().declaration_hash
    assert amended.to_dict()["amendments"][0]["reason"] == amendment.reason


def test_mode_assignment_and_support_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="known assignment"):
        randomized_declaration(assignment=EstimatedAssignment())
    with pytest.raises(ValueError, match="prespecified contrast"):
        randomized_declaration(contrasts=())
    with pytest.raises(ValueError, match="No calibrated"):
        SupportPolicy(calibrated=True, version="user-claimed-calibration")
    with pytest.raises(ValueError, match="sum to one"):
        KnownAssignment(probabilities=(("a", 0.7), ("b", 0.4)))


def test_associational_claim_is_derived_not_user_selected() -> None:
    declaration = randomized_declaration(
        mode=AnalysisMode.STANDARDIZED_ASSOCIATIONAL,
        assignment=EstimatedAssignment(nuisance_strategy="linear"),
    )
    assert declaration.claim_class is ClaimClass.ASSOCIATIONAL


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: KnownAssignment(), "exactly one"),
        (
            lambda: KnownAssignment(
                probabilities=(("a", 0.5), ("b", 0.5)),
                stratum_probabilities=(("s", (("a", 0.5), ("b", 0.5))),),
                stratum_column="s",
            ),
            "exactly one",
        ),
        (lambda: KnownAssignment(probabilities=(("a", 1.0),)), "at least two"),
        (
            lambda: KnownAssignment(probabilities=(("a", 0.5), ("a", 0.5))),
            "unique",
        ),
        (
            lambda: KnownAssignment(probabilities=(("a", -0.1), ("b", 1.1))),
            "positive",
        ),
        (
            lambda: KnownAssignment(
                probabilities=(("a", 0.5), ("b", 0.5)), stratum_column="block"
            ),
            "stratum_column",
        ),
        (
            lambda: KnownAssignment(
                stratum_probabilities=(("s", (("a", 0.5), ("b", 0.5))),)
            ),
            "nonempty stratum_column",
        ),
        (
            lambda: KnownAssignment(
                stratum_column="block",
                stratum_probabilities=(
                    ("s", (("a", 0.5), ("b", 0.5))),
                    ("s", (("a", 0.5), ("b", 0.5))),
                ),
            ),
            "strata must be unique",
        ),
        (
            lambda: KnownAssignment(
                stratum_column="block",
                stratum_probabilities=(
                    ("s1", (("a", 0.5), ("b", 0.5))),
                    ("s2", (("a", 0.5), ("c", 0.5))),
                ),
            ),
            "same randomized groups",
        ),
        (lambda: EstimatedAssignment("bad"), "Unsupported"),  # type: ignore[arg-type]
    ],
)
def test_assignment_declaration_failures(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"min_group_count": 1}, "at least two"),
        ({"min_ess_ratio": 0}, "min_ess_ratio"),
        ({"max_normalized_weight": 2}, "max_normalized_weight"),
        ({"max_top_one_percent_weight_share": 0}, "top_one"),
        ({"version": ""}, "version"),
    ],
)
def test_support_policy_validation(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SupportPolicy(**updates)  # type: ignore[arg-type]


def test_amendment_validation() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        DeclarationAmendment("", "reason", (("field", "value"),))
    with pytest.raises(ValueError, match="change names"):
        DeclarationAmendment("time", "reason", ())
    with pytest.raises(ValueError, match="change names"):
        DeclarationAmendment(
            "time", "reason", (("field", "first"), ("field", "second"))
        )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"covariates": ()}, "at least one"),
        ({"outcome": "x1"}, "distinct"),
        ({"scientific_question": " "}, "fields are empty"),
        ({"n_splits": 1}, "at least two"),
        ({"outcome_nuisance_strategy": "bad"}, "outcome nuisance"),
        (
            {"group_definitions": (("g0", "first"), ("g0", "duplicate"))},
            "unique groups",
        ),
        ({"covariate_rationales": (("x1", "only one"),)}, "Every and only"),
        (
            {
                "mode": AnalysisMode.STANDARDIZED_ASSOCIATIONAL,
                "assignment": KnownAssignment(
                    probabilities=(("g0", 0.5), ("g1", 0.5))
                ),
            },
            "Nonrandomized",
        ),
        (
            {
                "contrasts": (
                    ContrastSpec("duplicate", (("g0", 1), ("g1", -1))),
                    ContrastSpec("duplicate", (("g2", 1), ("g1", -1))),
                )
            },
            "unique",
        ),
        ({"sensitivity_analysis": " "}, "must not be blank"),
    ],
)
def test_cf_declaration_failures(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        randomized_declaration(**updates)


def test_string_mode_is_canonicalized() -> None:
    declaration = randomized_declaration(mode="randomized")
    assert declaration.mode is AnalysisMode.RANDOMIZED
    assert replace(declaration, random_state=99).declaration_hash != declaration.declaration_hash
