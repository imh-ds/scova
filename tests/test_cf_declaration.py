from dataclasses import FrozenInstanceError, replace

import pytest

from scova import SCOVA, ContrastSpec, SCOVADeclaration
from scova.cf import (
    AnalysisMode,
    CFSupportProfile,
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
        "assignment": KnownAssignment(probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))),
        "contrasts": (ContrastSpec("g0 - g1", (("g0", 1.0), ("g1", -1.0))),),
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
    with pytest.raises(ValueError, match="No promoted packaged"):
        SupportPolicy.packaged("not-a-real-profile")
    with pytest.raises(ValueError, match="sum to one"):
        KnownAssignment(probabilities=(("a", 0.7), ("b", 0.4)))


def test_packaged_support_policy_requires_exact_promoted_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compatibility = {
        "mode": "randomized",
        "outcome_type": "continuous",
        "estimator": "aipw-unnormalized",
        "estimand_id": "study-population-standardized-means",
        "assignment": "known-constant",
        "independent_unit": "row",
    }
    thresholds = {
        "minimum_ess_ratio": 0.25,
        "maximum_normalized_weight": 0.20,
        "maximum_top_one_percent_weight_share": 0.35,
        "maximum_absolute_weighted_balance_difference": 1.0,
        "maximum_influence_top_one_percent_share": 0.50,
        "maximum_seed_standardized_departure": 1.5,
    }
    profile = CFSupportProfile(
        profile_id="packaged-test",
        protocol_checksum="a" * 64,
        calibration_evidence_checksum="b" * 64,
        validation_evidence_checksum="c" * 64,
        thresholds=thresholds,
        compatibility=compatibility,
        state="promoted",
    )
    monkeypatch.setattr(
        SupportPolicy,
        "_trusted_profile",
        staticmethod(lambda _profile_id: profile.to_dict()),
    )
    policy = SupportPolicy.packaged("packaged-test")
    assert policy.calibrated is True
    assert policy.profile_checksum == profile.checksum
    incompatible = replace(profile, compatibility={**compatibility, "mode": "wrong"})
    monkeypatch.setattr(
        SupportPolicy,
        "_trusted_profile",
        staticmethod(lambda _profile_id: incompatible.to_dict()),
    )
    with pytest.raises(ValueError, match="incompatible"):
        SupportPolicy.packaged("packaged-test")


def test_packaged_arm_density_enforces_the_declared_scope_not_only_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile must screen at the density it claims to have been validated at.

    Arm density lives in both halves of a v10-era profile: compatibility holds
    the scope the campaign certified, thresholds hold the floor calibration
    picked inside it. Reading only the threshold would advertise a scope of 10
    units per covariate while screening at 0.36 -- roughly thirty times thinner
    than anything the campaign ever ran.
    """
    compatibility = {
        "mode": "observational-causal",
        "outcome_type": "continuous",
        "estimator": "aipw-unnormalized",
        "estimand_id": "study-population-standardized-means",
        "assignment": "estimated",
        "nuisance_strategy": "adaptive",
        "maximum_covariate_count": 5,
        "independent_unit": "row",
        "minimum_group_count": 50,
        "minimum_arm_units_per_covariate": 10.0,
    }
    thresholds = {
        "minimum_ess_ratio": 0.25,
        "minimum_arm_units_per_covariate": 0.36,
        "maximum_normalized_weight": 0.20,
        "maximum_top_one_percent_weight_share": 0.35,
        "maximum_absolute_weighted_balance_difference": 1.0,
        "maximum_influence_top_one_percent_share": 0.50,
        "maximum_seed_standardized_departure": 1.5,
    }
    profile = CFSupportProfile(
        profile_id="density-test",
        protocol_checksum="a" * 64,
        calibration_evidence_checksum="b" * 64,
        validation_evidence_checksum="c" * 64,
        thresholds=thresholds,
        compatibility=compatibility,
        state="promoted",
    )
    monkeypatch.setattr(
        SupportPolicy,
        "_trusted_profile",
        staticmethod(lambda _profile_id: profile.to_dict()),
    )
    assert SupportPolicy.packaged("density-test").min_arm_units_per_covariate == 10.0

    # The calibrated floor still binds when it is the stricter of the two.
    stricter = replace(
        profile,
        thresholds={**thresholds, "minimum_arm_units_per_covariate": 24.0},
    )
    monkeypatch.setattr(
        SupportPolicy,
        "_trusted_profile",
        staticmethod(lambda _profile_id: stricter.to_dict()),
    )
    assert SupportPolicy.packaged("density-test").min_arm_units_per_covariate == 24.0

    # v3-v9 declare neither key, so the bound stays inactive for them.
    legacy = replace(
        profile,
        compatibility={
            name: value
            for name, value in compatibility.items()
            if name != "minimum_arm_units_per_covariate"
        },
        thresholds={
            name: value
            for name, value in thresholds.items()
            if name != "minimum_arm_units_per_covariate"
        },
    )
    monkeypatch.setattr(
        SupportPolicy,
        "_trusted_profile",
        staticmethod(lambda _profile_id: legacy.to_dict()),
    )
    assert SupportPolicy.packaged("density-test").min_arm_units_per_covariate == 0.0


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
            lambda: KnownAssignment(probabilities=(("a", 0.5), ("b", 0.5)), stratum_column="block"),
            "stratum_column",
        ),
        (
            lambda: KnownAssignment(stratum_probabilities=(("s", (("a", 0.5), ("b", 0.5))),)),
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
        DeclarationAmendment("time", "reason", (("field", "first"), ("field", "second")))


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
                "assignment": KnownAssignment(probabilities=(("g0", 0.5), ("g1", 0.5))),
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


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ((0,), "primary"),
        ((2, 2), "unique"),
        ((-1,), "nonnegative"),
        ((True,), "nonnegative"),
    ],
)
def test_stability_seed_registry_is_governed(seeds: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        randomized_declaration(stability_seeds=seeds)


def test_stability_seeds_are_part_of_declaration_identity() -> None:
    baseline = randomized_declaration()
    declared = replace(baseline, stability_seeds=(101, 211, 307, 401, 503))
    assert declared.to_dict()["stability_seeds"] == [101, 211, 307, 401, 503]
    assert declared.declaration_hash != baseline.declaration_hash


def _profile(mode: str, assignment: str, profile_id: str = "regime-test") -> CFSupportProfile:
    return CFSupportProfile(
        profile_id=profile_id,
        protocol_checksum="a" * 64,
        calibration_evidence_checksum="b" * 64,
        validation_evidence_checksum="c" * 64,
        thresholds={
            "minimum_ess_ratio": 0.25,
            "maximum_normalized_weight": 0.20,
            "maximum_top_one_percent_weight_share": 0.35,
            "maximum_absolute_weighted_balance_difference": 1.0,
            "maximum_influence_top_one_percent_share": 0.50,
            "maximum_seed_standardized_departure": 1.5,
        },
        compatibility={
            "mode": mode,
            "outcome_type": "continuous",
            "estimator": "aipw-unnormalized",
            "estimand_id": "study-population-standardized-means",
            "assignment": assignment,
            "independent_unit": "row",
            **({"nuisance_strategy": "adaptive"} if mode == "observational-causal" else {}),
            **({"maximum_covariate_count": 5} if mode == "observational-causal" else {}),
        },
        state="promoted",
    )


def _install(monkeypatch: pytest.MonkeyPatch, profile: CFSupportProfile) -> None:
    monkeypatch.setattr(
        SupportPolicy, "_trusted_profile", staticmethod(lambda _id: profile.to_dict())
    )


def test_packaged_policy_accepts_an_observational_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, _profile("observational-causal", "estimated"))
    policy = SupportPolicy.packaged("regime-test")
    assert policy.calibrated is True
    assert (policy.profile_mode, policy.profile_assignment) == (
        "observational-causal",
        "estimated",
    )
    assert policy.profile_nuisance_strategy == "adaptive"


def test_observational_packaged_profile_requires_an_adaptive_nuisance_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile("observational-causal", "estimated")
    values = profile.to_dict()
    values["compatibility"].pop("nuisance_strategy")
    # Rebuild after mutation so the fixture remains checksum-valid.
    missing = CFSupportProfile(
        profile_id=profile.profile_id,
        protocol_checksum=profile.protocol_checksum,
        calibration_evidence_checksum=profile.calibration_evidence_checksum,
        validation_evidence_checksum=profile.validation_evidence_checksum,
        thresholds=profile.thresholds,
        compatibility=values["compatibility"],
        state="promoted",
    )
    _install(monkeypatch, missing)
    with pytest.raises(ValueError, match="nuisance_strategy='adaptive'"):
        SupportPolicy.packaged("regime-test")


@pytest.mark.parametrize("maximum_covariate_count", [None, 0, 6])
def test_observational_packaged_profile_requires_a_conservative_predictor_cap(
    monkeypatch: pytest.MonkeyPatch, maximum_covariate_count: int | None
) -> None:
    profile = _profile("observational-causal", "estimated")
    compatibility = dict(profile.compatibility or {})
    if maximum_covariate_count is None:
        compatibility.pop("maximum_covariate_count")
    else:
        compatibility["maximum_covariate_count"] = maximum_covariate_count
    capped = replace(profile, compatibility=compatibility)
    _install(monkeypatch, capped)
    with pytest.raises(ValueError, match="maximum_covariate_count"):
        SupportPolicy.packaged("regime-test")


def test_packaged_policy_refuses_a_regime_the_release_cannot_govern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mode and assignment must pair coherently; a profile claiming estimated
    # assignment under randomization describes a campaign that cannot exist.
    for mode, assignment in (
        ("standardized-associational", "estimated"),
        ("randomized", "estimated"),
        ("observational-causal", "known-constant"),
    ):
        _install(monkeypatch, _profile(mode, assignment))
        with pytest.raises(ValueError, match="incompatible analysis lock"):
            SupportPolicy.packaged("regime-test")


def test_a_calibrated_profile_only_governs_its_own_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out evidence from one regime must not vouch for another."""
    _install(monkeypatch, _profile("randomized", "known-constant"))
    randomized_policy = SupportPolicy.packaged("regime-test")
    known = KnownAssignment(probabilities=(("g0", 0.5), ("g1", 0.5)))
    estimated = EstimatedAssignment()

    assert randomized_policy.governs(AnalysisMode.RANDOMIZED, known) is None
    assert "cannot govern" in (
        randomized_policy.governs(AnalysisMode.OBSERVATIONAL_CAUSAL, estimated) or ""
    )
    # Stratified randomization is a different validated object from constant.
    stratified = KnownAssignment(
        stratum_column="site",
        stratum_probabilities=(("s1", (("g0", 0.5), ("g1", 0.5))),),
    )
    assert "known-stratified" in (
        randomized_policy.governs(AnalysisMode.RANDOMIZED, stratified) or ""
    )

    _install(monkeypatch, _profile("observational-causal", "estimated"))
    observational_policy = SupportPolicy.packaged("regime-test")
    assert observational_policy.governs(AnalysisMode.OBSERVATIONAL_CAUSAL, estimated) is None
    assert "cannot govern" in (observational_policy.governs(AnalysisMode.RANDOMIZED, known) or "")


def test_uncalibrated_policy_governs_every_mode() -> None:
    provisional = SupportPolicy()
    assert provisional.calibrated is False
    for mode, assignment in (
        (AnalysisMode.RANDOMIZED, KnownAssignment(probabilities=(("g0", 0.5), ("g1", 0.5)))),
        (AnalysisMode.OBSERVATIONAL_CAUSAL, EstimatedAssignment()),
        (AnalysisMode.STANDARDIZED_ASSOCIATIONAL, EstimatedAssignment()),
    ):
        assert provisional.governs(mode, assignment) is None
