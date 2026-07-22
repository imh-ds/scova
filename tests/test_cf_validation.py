import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from benchmarks import aggregate_cf_campaign, cf_inference_campaign
from benchmarks.cf_external_validation import (
    KnownRandomizationClassifier,
    SelectedOutcomeRegressor,
    TreatmentSpecificOutcomeRegressor,
    fixed_nuisance_score,
)
from benchmarks.cf_reference_campaign import (
    plasmode_source_checksum,
    run_campaign,
    run_shard,
    simulate_plasmode_cell,
    write_deterministic_gzip,
)
from scova._aipw import assemble_aipw
from scova.cf import (
    CFSupportProfile,
    CFValidationProtocol,
    SeedPartition,
    _numerical_identity,
    canonical_checksum,
)
from scova.simulate import generate_data

sys.path.insert(0, str(Path("scripts").resolve()))

from scripts.audit_cf_pilot import audit_pilot
from scripts.calibrate_cf_support import (
    _candidate_enrichments,
    _cell_gate,
    _family_wise_multiplier,
    _screening_cell_gate,
    _unstable_enrichment,
)
from scripts.check_cf_campaign_prerequisites import prerequisite_reasons
from scripts.validate_cf_support import (
    _candidate_matches_protocol,
    _external_matches_protocol,
    _inference_matches_protocol,
)

SPEC = Path("benchmarks/specs/cf_reference_v3.json")
V4_SPEC = Path("benchmarks/specs/cf_reference_v4.json")
V6_SPEC = Path("benchmarks/specs/cf_reference_v6.json")
V7_SPEC = Path("benchmarks/specs/cf_reference_v7.json")
BLOCKED_V2 = Path("benchmarks/specs/cf_reference_v2_blocked.json")


def test_frozen_reference_protocol_has_disjoint_evidence_lanes() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    assert protocol.protocol_id == "cf-randomized-continuous-aipw-unnormalized-v3"
    assert protocol.calibration.count == 1000
    assert protocol.validation.count == 2000
    assert protocol.schema_version == 2
    assert protocol.frozen is True
    assert len(protocol.retained_cells) == 48
    assert len(protocol.plasmode_cells) == 12
    assert len(protocol.inference_cells) == 6
    assert len(protocol.external_cells) == 8
    assert protocol.external is not None and protocol.external.count == 50
    assert protocol.inference is not None and protocol.inference.count == 2000
    assert protocol.pilot.start == 1_000_000_000
    assert protocol.calibration.start == 1_100_000_000
    assert protocol.validation.start == 1_300_000_000
    assert protocol.external.start == 1_600_000_000
    assert protocol.inference.start == 1_700_000_000
    assert protocol.checksum == (
        "dfb842e6e54aff3f11a7a5a8780881bfa78e3866a230231407286ce9d9e439c0"
    )
    assert protocol.checksum == CFValidationProtocol.from_dict(
        protocol.to_dict()
    ).checksum


def test_v4_protocol_uses_new_seed_namespaces_and_calibration_screening() -> None:
    protocol = CFValidationProtocol.load(V4_SPEC)
    assert protocol.protocol_id == "cf-randomized-continuous-aipw-unnormalized-v4"
    assert protocol.checksum == "002d1b3e06f2d54bbc4f391f2e855892418275d44ae8a6cf69fee72fbdbd3cff"
    assert protocol.pilot.start == 2_000_000_000
    assert protocol.calibration.start == 2_100_000_000
    assert protocol.validation.start == 2_300_000_000
    assert protocol.calibration_candidate_retention_fraction == 0.85
    assert protocol.calibration_gate_metrics["maximum_standardized_bias"] == 0.15
    assert protocol.calibration_gate_metrics["minimum_se_ratio"] == 0.8
    assert protocol.calibration_gate_metrics["maximum_se_ratio"] == 1.25
    assert CFValidationProtocol.from_dict(protocol.to_dict()).checksum == protocol.checksum


def test_one_sided_calibration_screening_allows_conservative_inference() -> None:
    # The empirical error SD matches the reported SE, but all intervals are
    # conservative and no null is rejected.  The v4 calibration screen must
    # reserve this for held-out adjudication rather than reject it here.
    records = [
        {
            "contrasts": [
                {
                    "covered": True,
                    "estimate": error,
                    "truth": 0.0,
                    "standard_error": 1.0,
                    "null": True,
                    "rejected": False,
                }
            ]
        }
        for error in (-1.0, 1.0) * 10
    ]
    protocol = CFValidationProtocol.load(V4_SPEC)
    passed, audit = _screening_cell_gate(records, protocol.calibration_gate_metrics)
    assert passed is True
    assert audit["coverage_ok"] is True
    assert audit["type_i_ok"] is True


def test_v6_inference_amendment_binds_archived_upstream_evidence() -> None:
    protocol = CFValidationProtocol.load(V6_SPEC)
    assert protocol.protocol_id == "cf-randomized-continuous-aipw-unnormalized-v6"
    assert protocol.calibration_source == {
        "protocol_id": "cf-randomized-continuous-aipw-unnormalized-v4",
        "protocol_checksum": "002d1b3e06f2d54bbc4f391f2e855892418275d44ae8a6cf69fee72fbdbd3cff",
        "evidence_checksum": "bbfa9374c3fe3af99c73f695a163f71110ff990531fe245d6108bd3b64978bf3",
        "git_commit": "2abca2746530ba033a0e857b32f7d34edba5711c",
    }
    assert protocol.candidate_source == {
        "protocol_id": "cf-randomized-continuous-aipw-unnormalized-v5",
        "protocol_checksum": "7521cf977c51e97498ef7623c6facadfb8423a22e0740c2145d3ee7bbe68431b",
        "profile_checksum": "ea2614448b9c62b4db8c302aa56d2d8d8df4f8d6417dbc1ea65e5400c9639904",
    }
    assert protocol.external_source is not None
    assert protocol.failed_inference_source is not None
    assert protocol.reference_profile["minimum_group_count"] == 50
    assert protocol.reference_profile["maximum_group_count"] == 3
    assert protocol.inference is not None and protocol.inference.start == 3_900_000_000
    assert all("cell" in reference for reference in protocol.inference_cells)
    for reference in protocol.inference_cells:
        cell = reference["cell"]
        assert cell["support"] == "strong"
        assert cell["n_groups"] <= 3
        assert cell["n_per_group"] >= 80
    assert CFValidationProtocol.from_dict(protocol.to_dict()).checksum == protocol.checksum


def test_validation_accepts_only_the_exact_frozen_candidate_and_external_sources() -> None:
    protocol = CFValidationProtocol.load(V6_SPEC)
    candidate = CFSupportProfile(
        profile_id="source-candidate",
        protocol_checksum="source-protocol",
        calibration_evidence_checksum="calibration",
        validation_evidence_checksum=None,
        thresholds={"minimum_group_count": 50},
        compatibility={"estimand": "continuous-treatment-contrast"},
        state="candidate",
    )
    protocol = replace(
        protocol,
        candidate_source={
            "protocol_id": "source-protocol-id",
            "protocol_checksum": candidate.protocol_checksum,
            "profile_checksum": candidate.checksum,
        },
        external_source={
            "protocol_id": "external-protocol-id",
            "protocol_checksum": "external-protocol",
            "evidence_checksum": "external-evidence",
            "git_commit": "external-commit",
        },
    )
    external = {
        "protocol_checksum": "external-protocol",
        "evidence_checksum": "external-evidence",
        "git_commit": "external-commit",
    }

    assert _candidate_matches_protocol(protocol, candidate)
    assert _external_matches_protocol(protocol, external)
    assert not _candidate_matches_protocol(
        protocol, replace(candidate, calibration_evidence_checksum="tampered")
    )
    assert not _external_matches_protocol(protocol, {**external, "git_commit": "tampered"})


def test_v7_recalibrates_on_rejected_v6_evidence_and_reserves_fresh_validation() -> None:
    protocol = CFValidationProtocol.load(V7_SPEC)
    assert protocol.protocol_id == "cf-randomized-continuous-aipw-unnormalized-v7"
    assert protocol.checksum == "f393f13e40331cbf7a3de0fb336379258d1832768b4a01a25f169b61b62888c7"
    assert protocol.calibration.start == 3_300_000_000
    assert protocol.calibration.count == 2000
    assert protocol.validation.start == 4_100_000_000
    assert protocol.validation.count == 2000
    assert protocol.calibration_source is not None
    assert protocol.calibration_source["lane"] == "validation"
    assert protocol.calibration_source["evidence_checksum"] == (
        "4a6a4515456df1dd1e9943a82971d918d41db716598482d005628c7721daf7ea"
    )
    assert protocol.candidate_source is None
    assert protocol.inference_source is not None
    assert protocol.calibration_enrichment_screening is True
    assert protocol.metrics["minimum_unstable_risk_ratio"] == 2
    assert protocol.metrics["minimum_unstable_absolute_enrichment"] == 0.05
    assert CFValidationProtocol.from_dict(protocol.to_dict()).checksum == protocol.checksum


def test_v7_calibration_enrichment_gate_rejects_weak_risk_separation() -> None:
    thresholds = {
        "minimum_ess_ratio": 0.5,
        "maximum_normalized_weight": 0.5,
        "maximum_top_one_percent_weight_share": 0.5,
        "maximum_absolute_weighted_balance_difference": 0.5,
        "maximum_influence_top_one_percent_share": 0.5,
        "maximum_seed_standardized_departure": 0.5,
    }

    def record(*, supported: bool, bad: bool) -> dict:
        feature = 0.25 if supported else 0.75
        return {
            "support_features": {
                "minimum_ess_ratio": 0.75,
                **{
                    name: feature
                    for name in (
                        "maximum_normalized_weight",
                        "maximum_top_one_percent_weight_share",
                        "maximum_absolute_weighted_balance_difference",
                        "maximum_influence_top_one_percent_share",
                        "maximum_seed_standardized_departure",
                    )
                },
            },
            "contrasts": [
                {
                    "covered": not bad,
                    "estimate": 3.0 if bad else 0.0,
                    "truth": 0.0,
                    "standard_error": 1.0,
                }
            ],
        }

    records = [record(supported=True, bad=False) for _ in range(20)] + [
        record(supported=False, bad=index < 4) for index in range(20)
    ]
    result = _unstable_enrichment(
        records,
        thresholds,
        {
            "minimum_unstable_risk_ratio": 2.0,
            "minimum_unstable_absolute_enrichment": 0.05,
        },
    )
    assert result["passed"] is True
    assert result["absolute_enrichment"] == 0.2
    assert _candidate_enrichments(
        records,
        [thresholds],
        {
            "minimum_unstable_risk_ratio": 2.0,
            "minimum_unstable_absolute_enrichment": 0.05,
        },
    ) == [result]


def test_validation_accepts_only_the_exact_frozen_inference_source() -> None:
    protocol = CFValidationProtocol.load(V7_SPEC)
    assert protocol.inference_source is not None
    evidence = dict(protocol.inference_source)
    assert _inference_matches_protocol(protocol, evidence)
    assert not _inference_matches_protocol(protocol, {**evidence, "git_commit": "tampered"})


@pytest.mark.parametrize(
    ("source", "field", "message"),
    (
        ("candidate_source", "profile_checksum", "candidate source is missing fields"),
        ("external_source", "evidence_checksum", "external source is missing fields"),
        (
            "failed_inference_source",
            "git_commit",
            "failed inference source is missing fields",
        ),
    ),
)
def test_v6_protocol_rejects_incomplete_reused_evidence_sources(
    source: str, field: str, message: str
) -> None:
    values = json.loads(V6_SPEC.read_text(encoding="utf-8"))
    del values[source][field]
    with pytest.raises(ValueError, match=message):
        CFValidationProtocol.from_dict(values)


def test_v7_protocol_rejects_an_incomplete_inference_source() -> None:
    values = json.loads(V7_SPEC.read_text(encoding="utf-8"))
    del values["inference_source"]["evidence_checksum"]
    with pytest.raises(ValueError, match="inference source is missing fields"):
        CFValidationProtocol.from_dict(values)


def test_known_randomization_adapter_never_estimates_fixture_propensities() -> None:
    adapter = KnownRandomizationClassifier((0.2, 0.3, 0.5)).fit(
        np.zeros((4, 2)), np.array([0, 1, 2, 1])
    )
    assert np.allclose(
        adapter.predict_proba(np.ones((3, 2))),
        np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
    )
    assert np.array_equal(adapter.predict(np.ones((2, 2))), np.array([2, 2]))


def test_external_outcome_adapters_preserve_treatment_specific_linear_policy() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [2.0], [3.0]])
    treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    outcome = np.where(treatment == 0, 1.0 + 2.0 * features[:, 0], -3.0 + 5.0 * features[:, 0])
    design = np.column_stack([features, treatment])
    fitted = TreatmentSpecificOutcomeRegressor(n_groups=2, learner_policy="linear").fit(
        design, outcome
    )
    counterfactual_design = np.array([[4.0, 0.0], [4.0, 1.0]])
    expected = np.array(
        [
            Ridge(alpha=1.0).fit(features[:4], outcome[:4]).predict([[4.0]])[0],
            Ridge(alpha=1.0).fit(features[4:], outcome[4:]).predict([[4.0]])[0],
        ]
    )
    assert np.allclose(fitted.predict(counterfactual_design), expected)
    selected = SelectedOutcomeRegressor("linear").fit(features[:4], outcome[:4])
    assert selected.selected_name_ == "Ridge"


def test_inference_aggregate_main_creates_requested_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested" / "evidence" / "inference.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf_inference_campaign",
            "--spec",
            str(V6_SPEC),
            "--aggregate",
            "unused-shard.ndjson.gz",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(cf_inference_campaign, "aggregate", lambda *_args, **_kwargs: {"ok": True})
    cf_inference_campaign.main()
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def test_v6_inference_shard_accepts_checksum_bound_inline_cells(tmp_path: Path) -> None:
    output = tmp_path / "inference-0.ndjson.gz"
    cf_inference_campaign.run_shard(
        CFValidationProtocol.load(V6_SPEC),
        output=output,
        shard_index=0,
        shard_count=1,
        replications=1,
    )
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    assert record["simulation_cell_index"] is None
    assert record["cell"]["support"] == "strong"


def test_inference_environment_identity_ignores_only_host_platform() -> None:
    base = {
        "python": "3.12.13",
        "scova": "0.3.0.dev0",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "scikit-learn": "1.6.1",
        "platform": "Linux-6.17.0-1020-azure-x86_64-with-glibc2.39",
    }
    alternate_host = {**base, "platform": "Linux-6.17.0-1018-azure-x86_64-with-glibc2.39"}
    assert (
        cf_inference_campaign._numerical_environment_identity(base)
        == cf_inference_campaign._numerical_environment_identity(alternate_host)
    )
    assert (
        cf_inference_campaign._numerical_environment_identity({**base, "numpy": "2.3.0"})
        != cf_inference_campaign._numerical_environment_identity(base)
    )
    incomplete = dict(base)
    del incomplete["scipy"]
    with pytest.raises(ValueError, match="missing fields"):
        cf_inference_campaign._numerical_environment_identity(incomplete)


def test_inference_fwer_gate_requires_control_only_when_a_true_null_exists() -> None:
    no_null = [{"contrasts": [{"null": False}], "simultaneous": {"any_null_rejected": False}}]
    assert cf_inference_campaign._familywise_error_gate(
        no_null, alpha=0.05, multiplier=2.0
    ) == (None, True)
    conservative = [
        {"contrasts": [{"null": True}], "simultaneous": {"any_null_rejected": False}}
        for _ in range(100)
    ]
    assert cf_inference_campaign._familywise_error_gate(
        conservative, alpha=0.05, multiplier=2.0
    ) == (0.0, True)
    inflated = [
        {"contrasts": [{"null": True}], "simultaneous": {"any_null_rejected": index < 20}}
        for index in range(100)
    ]
    assert cf_inference_campaign._familywise_error_gate(
        inflated, alpha=0.05, multiplier=2.0
    ) == (0.2, False)


def test_validation_accepts_the_checksum_bound_source_candidate(tmp_path: Path) -> None:
    protocol = CFValidationProtocol.load(V6_SPEC)
    candidate = CFSupportProfile(
        profile_id="source-candidate",
        protocol_checksum="source-protocol-checksum",
        calibration_evidence_checksum="a" * 64,
        validation_evidence_checksum=None,
        thresholds={"minimum_ess_ratio": 0.25},
        compatibility=protocol.reference_profile,
    )
    sourced_protocol = replace(
        protocol,
        candidate_source={
            "protocol_id": "source-protocol",
            "protocol_checksum": candidate.protocol_checksum,
            "profile_checksum": candidate.checksum,
        },
    )
    output = tmp_path / "sourced-validation.ndjson.gz"
    run_shard(
        sourced_protocol,
        lane="validation",
        output=output,
        shard_index=0,
        shard_count=128,
        resume=False,
        replications_override=1,
        max_cells=1,
        include_stability=False,
        candidate_profile=candidate,
    )
    metadata = json.loads(
        output.with_suffix(output.suffix + ".metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["candidate_profile_checksum"] == candidate.checksum


def test_campaign_environment_identity_ignores_only_host_platform() -> None:
    base = {
        "python": "3.12.13",
        "scova": "0.3.0.dev0",
        "numpy": "2.2.6",
        "pandas": "2.2.3",
        "scipy": "1.15.3",
        "scikit-learn": "1.6.1",
        "platform": "Linux-6.17.0-1020-azure-x86_64-with-glibc2.39",
    }
    other_host = {**base, "platform": "Linux-6.17.0-1018-azure-x86_64-with-glibc2.39"}
    assert (
        aggregate_cf_campaign._numerical_environment_identity(base)
        == aggregate_cf_campaign._numerical_environment_identity(other_host)
    )
    assert (
        aggregate_cf_campaign._numerical_environment_identity({**base, "pandas": "2.3.0"})
        != aggregate_cf_campaign._numerical_environment_identity(base)
    )


def _fake_numerical_source(commit: str, path: str) -> bytes:
    if path.endswith(".json"):
        return b"{}"
    if path == "benchmarks/cf_reference_campaign.py":
        governance = "candidate_source = True" if commit == "after-governance" else "pass"
        return f"""
STABILITY_SEEDS = (1, 2)
class CampaignData: pass
def _probabilities(): pass
def _conditional_means(): pass
def _errors(): pass
def simulate_reference_cell(): pass
def _declaration(): pass
def _contrast_summary(): pass
def _support_features(): pass
def fit_campaign_record(): pass
def run_shard():
    {governance}
""".encode()
    if path == "benchmarks/cf_inference_campaign.py":
        gate = "True" if commit == "after-inference-gate" else "False"
        fingerprint = (
            "cf_numerical_fingerprint('commit', 'inference')"
            if commit == "after-identity-refactor"
            else "_cf_numerical_fingerprint('commit')"
        )
        return f"""
N_BOOTSTRAP = 999
_NUMERICAL_ENVIRONMENT_FIELDS = ('python',)
def _version(): pass
def _commit(): pass
def _numerical_environment_identity(): pass
def _familywise_error_gate(): return {gate}
def run_shard(): pass
def aggregate(): return {fingerprint}
""".encode()
    return b"value = 1\n"


def test_numerical_fingerprints_ignore_only_campaign_governance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_numerical_identity, "_committed_file", _fake_numerical_source)
    for kind in ("external", "inference"):
        assert _numerical_identity.cf_numerical_fingerprint("before-governance", kind) == (
            _numerical_identity.cf_numerical_fingerprint("after-governance", kind)
        )


def test_numerical_fingerprints_are_evidence_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_numerical_identity, "_committed_file", _fake_numerical_source)
    assert _numerical_identity.cf_numerical_fingerprint(
        "before-inference-gate", "external"
    ) == _numerical_identity.cf_numerical_fingerprint("after-inference-gate", "external")
    assert _numerical_identity.cf_numerical_fingerprint(
        "before-inference-gate", "inference"
    ) != _numerical_identity.cf_numerical_fingerprint("after-inference-gate", "inference")


def test_numerical_fingerprint_refactor_does_not_invalidate_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_numerical_identity, "_committed_file", _fake_numerical_source)
    before = _numerical_identity.cf_numerical_fingerprint(
        "before-identity-refactor", "inference"
    )
    after = _numerical_identity.cf_numerical_fingerprint(
        "after-identity-refactor", "inference"
    )
    assert before == after


def test_v2_is_machine_readably_blocked_without_using_heldout_evidence() -> None:
    blocked = json.loads(BLOCKED_V2.read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert blocked["heldout_validation_inspected"] is False
    assert blocked["profile_promoted"] is False
    assert blocked["replacement_protocol_id"] == (
        "cf-randomized-continuous-aipw-unnormalized-v3"
    )
    supplied = blocked.pop("blocking_record_checksum")
    assert supplied == canonical_checksum(blocked)


def test_protocol_rejects_overlapping_or_undersized_lanes() -> None:
    values = json.loads(SPEC.read_text(encoding="utf-8"))
    values["seed_partitions"]["validation"] = {
        "start": 1_100_000_500,
        "count": 2000,
    }
    with pytest.raises(ValueError, match="disjoint"):
        CFValidationProtocol.from_dict(values)
    values = json.loads(V7_SPEC.read_text(encoding="utf-8"))
    values["seed_partitions"]["pilot"]["start"] = 4_294_960_000
    with pytest.raises(ValueError, match="scikit-learn random_state"):
        CFValidationProtocol.from_dict(values)


def test_v3_protocol_rejects_incomplete_frozen_contract() -> None:
    original = json.loads(SPEC.read_text(encoding="utf-8"))

    def rejected(values: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            CFValidationProtocol.from_dict(values)

    values = {**original, "protocol_id": ""}
    rejected(values, "protocol_id")
    values = {**original, "frozen": False}
    rejected(values, "must be frozen")
    values = {**original, "retained_cells": original["retained_cells"][:-1]}
    rejected(values, "48 simulation")
    values = {**original, "plasmode_cells": original["plasmode_cells"][:-1]}
    rejected(values, "12 plasmode")
    values = {**original, "inference_cells": original["inference_cells"][:-1]}
    rejected(values, "six inference")
    values = {**original, "external_cells": original["external_cells"][:-1]}
    rejected(values, "eight external")
    values = json.loads(json.dumps(original))
    del values["seed_partitions"]["external"]
    rejected(values, "external and inference seeds")
    values = json.loads(json.dumps(original))
    values["seed_partitions"]["external"]["count"] = 49
    rejected(values, "50 replications")
    values = json.loads(json.dumps(original))
    values["seed_partitions"]["inference"]["count"] = 1999
    rejected(values, "2,000 replications")
    values = {**original, "dataset_checksums": {"diabetes": "x"}}
    rejected(values, "both plasmode")
    values = {**original, "dependency_lock_checksum": ""}
    rejected(values, "dependency-lock")
    values = {**original, "design_selection": {}}
    rejected(values, "pairwise-design")
    values = {**original, "calibration_fit_fraction": 1.0}
    rejected(values, "calibration_fit_fraction")
    values = json.loads(json.dumps(original))
    del values["retained_cells"][0]["learner"]
    rejected(values, "every campaign factor")
    values = json.loads(json.dumps(original))
    values["retained_cells"][0]["learner"] = "invalid"
    rejected(values, "undeclared factor")
    values = json.loads(json.dumps(original))
    del values["metrics"]["confidence_level"]
    rejected(values, "missing metrics")
    values = json.loads(SPEC.read_text(encoding="utf-8"))
    values["seed_partitions"]["calibration"] = {
        "start": 1_100_000_000,
        "count": 999,
    }
    with pytest.raises(ValueError, match="1,000"):
        CFValidationProtocol.from_dict(values)


def test_support_profile_checksum_and_promotion_evidence_are_enforced() -> None:
    candidate = CFSupportProfile(
        profile_id="candidate-v1",
        protocol_checksum="a" * 64,
        calibration_evidence_checksum="b" * 64,
        validation_evidence_checksum=None,
        thresholds={"minimum_ess_ratio": 0.25},
    )
    assert CFSupportProfile.from_dict(candidate.to_dict()) == candidate
    tampered = candidate.to_dict()
    tampered["thresholds"]["minimum_ess_ratio"] = 0.5
    with pytest.raises(ValueError, match="checksum"):
        CFSupportProfile.from_dict(tampered)
    with pytest.raises(ValueError, match="held-out"):
        CFSupportProfile(
            profile_id="invalid",
            protocol_checksum="a",
            calibration_evidence_checksum="b",
            validation_evidence_checksum=None,
            thresholds={"minimum_ess_ratio": 0.25},
            state="promoted",
        )


def test_fixed_nuisance_reference_matches_shared_engine_to_machine_precision() -> None:
    simulation = generate_data("observational", n=180, seed=41)
    labels = simulation.group_labels
    codes = np.array([labels.index(value) for value in simulation.data["group"]])
    outcome = simulation.data["outcome"].to_numpy()
    expected = assemble_aipw(
        outcome, codes, simulation.propensity, simulation.outcome_regression
    )
    observed = fixed_nuisance_score(
        outcome, codes, simulation.propensity, simulation.outcome_regression
    )
    for left, right in zip(expected, observed, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13)


def test_smoke_campaign_is_deterministic_and_cannot_promote() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    first = run_campaign(
        protocol,
        lane="pilot",
        replications=1,
        max_cells=1,
        include_stability=False,
    )
    second = run_campaign(
        protocol,
        lane="pilot",
        replications=1,
        max_cells=1,
        include_stability=False,
    )
    assert first == second
    assert first["complete_frozen_lane"] is False
    assert first["promotion_decision"] == "blocked/no-calibrated-support-profile"
    assert len(first["records"]) == 1


def test_seed_partition_requires_a_nonempty_nonnegative_range() -> None:
    assert SeedPartition(4, 3).stop == 7
    with pytest.raises(ValueError):
        SeedPartition(-1, 1)
    with pytest.raises(ValueError):
        SeedPartition(0, 0)


def test_pairwise_design_covers_every_declared_factor_level_pair() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    factor_names = tuple(protocol.factors)
    for left_index, left in enumerate(factor_names):
        for right in factor_names[left_index + 1 :]:
            observed = {(cell[left], cell[right]) for cell in protocol.retained_cells}
            expected = {
                (left_value, right_value)
                for left_value in protocol.factors[left]
                for right_value in protocol.factors[right]
            }
            assert observed == expected


def test_plasmode_uses_unique_rows_and_frozen_source_truth() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    cell = protocol.plasmode_cells[0]
    generated = simulate_plasmode_cell(cell, seed=991)
    indices = generated.source_metadata["source_row_indices"]
    assert len(indices) == len(set(indices)) == len(generated.data)
    source = str(cell["dataset"])
    assert plasmode_source_checksum(source) == protocol.dataset_checksums[source]
    assert generated.true_group_means.shape == (int(cell["n_groups"]),)
    assert np.all(np.isfinite(generated.true_group_means))


def test_compressed_campaign_payload_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    write_deterministic_gzip(first, '{"record":1}\n')
    write_deterministic_gzip(second, '{"record":1}\n')
    assert first.read_bytes() == second.read_bytes()


def test_heldout_shard_requires_and_records_candidate_lock(tmp_path: Path) -> None:
    protocol = CFValidationProtocol.load(SPEC)
    output = tmp_path / "validation-0.ndjson.gz"
    with pytest.raises(ValueError, match="frozen candidate"):
        run_shard(
            protocol,
            lane="validation",
            output=output,
            shard_index=0,
            shard_count=128,
            resume=False,
            replications_override=1,
            max_cells=1,
            include_stability=False,
        )
    candidate = CFSupportProfile(
        profile_id="locked-candidate",
        protocol_checksum=protocol.checksum,
        calibration_evidence_checksum="a" * 64,
        validation_evidence_checksum=None,
        thresholds={"minimum_ess_ratio": 0.25},
        compatibility=protocol.reference_profile,
    )
    run_shard(
        protocol,
        lane="validation",
        output=output,
        shard_index=0,
        shard_count=128,
        resume=False,
        replications_override=1,
        max_cells=1,
        include_stability=False,
        candidate_profile=candidate,
    )
    first_bytes = output.read_bytes()
    metadata = json.loads(
        output.with_suffix(output.suffix + ".metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["candidate_profile_checksum"] == candidate.checksum
    run_shard(
        protocol,
        lane="validation",
        output=output,
        shard_index=0,
        shard_count=128,
        resume=True,
        replications_override=1,
        max_cells=1,
        include_stability=False,
        candidate_profile=candidate,
    )
    assert output.read_bytes() == first_bytes


def test_full_pilot_audit_enforces_runtime_margin_and_complete_metadata(
    tmp_path: Path,
) -> None:
    protocol = CFValidationProtocol.load(SPEC)
    evidence = {
        "protocol_checksum": protocol.checksum,
        "evidence_checksum": "e" * 64,
        "lane": "pilot",
        "complete_frozen_lane": True,
        "cell_count": 60,
        "replications_per_cell": 20,
        "shard_count": 16,
        "execution_error_count": 0,
    }
    paths = []
    for index in range(16):
        values = {
            "complete_frozen_lane_configuration": True,
            "protocol_checksum": protocol.checksum,
            "shard_index": index,
            "elapsed_seconds": 60.0,
            "record_count": 75,
        }
        values["metadata_checksum"] = canonical_checksum(values)
        path = tmp_path / f"pilot-{index}.metadata.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        paths.append(path)
    result = audit_pilot(evidence, paths, protocol)
    assert result["passed"] is True
    assert result["promotion_eligible"] is False
    evidence["execution_error_count"] = 1
    assert audit_pilot(evidence, paths, protocol)["passed"] is False
    evidence["execution_error_count"] = 0
    assert audit_pilot(evidence, paths, protocol, job_limit_minutes=1)["passed"] is False


def test_campaign_prerequisites_lock_order_commit_and_evidence() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    commit = "a" * 40
    campaign = {
        "protocol_checksum": protocol.checksum,
        "git_commit": commit,
    }
    campaign["evidence_checksum"] = canonical_checksum(campaign)
    candidate = CFSupportProfile(
        profile_id="frozen-candidate",
        protocol_checksum=protocol.checksum,
        calibration_evidence_checksum=campaign["evidence_checksum"],
        validation_evidence_checksum=None,
        thresholds={"minimum_ess_ratio": 0.25},
        compatibility=protocol.reference_profile,
    ).to_dict()
    audit = {
        "protocol_checksum": protocol.checksum,
        "all_calibration_gates_passed": True,
        "calibration_evidence_checksum": campaign["evidence_checksum"],
        "candidate_profile": candidate,
    }
    audit["calibration_artifact_checksum"] = canonical_checksum(audit)
    external = {
        "protocol_checksum": protocol.checksum,
        "git_commit": commit,
        "all_numerical_agreement_gates_passed": True,
    }
    external["evidence_checksum"] = canonical_checksum(external)
    inference = {
        "protocol_checksum": protocol.checksum,
        "git_commit": commit,
        "all_inference_gates_passed": True,
    }
    inference["evidence_checksum"] = canonical_checksum(inference)
    assert prerequisite_reasons(
        "validation",
        protocol,
        calibration_campaign=campaign,
        calibration_audit=audit,
        candidate=candidate,
        expected_commit=commit,
        external=external,
        inference=inference,
    ) == []
    assert "external-agreement evidence is required" in prerequisite_reasons(
        "validation",
        protocol,
        calibration_campaign=campaign,
        calibration_audit=audit,
        candidate=candidate,
        expected_commit=commit,
        inference=inference,
    )
    assert "calibration campaign commit mismatch" in prerequisite_reasons(
        "external",
        protocol,
        calibration_campaign=campaign,
        calibration_audit=audit,
        candidate=candidate,
        expected_commit="b" * 40,
    )
    blocked = prerequisite_reasons(
        "external",
        protocol,
        calibration_campaign=campaign,
        calibration_audit={
            "all_calibration_gates_passed": False,
            "calibration_evidence_checksum": campaign["evidence_checksum"],
        },
        candidate=None,
        expected_commit=commit,
    )
    assert "calibration gates did not pass" in blocked
    assert (
        "candidate profile is missing because calibration did not promote a support policy"
        in blocked
    )


_GATE_METRICS = {
    "monte_carlo_standard_error_multiplier": 2.0,
    "maximum_standardized_bias": 0.15,
    "minimum_se_ratio": 0.9,
    "maximum_se_ratio": 1.1,
}


def _coverage_cell(covered: int, total: int, *, seed: int = 0) -> list[dict]:
    """Build one gated cell whose coverage is exactly covered/total.

    Errors are drawn ~N(0, 1) with unit standard errors so bias, empirical SD,
    and the SE ratio all sit comfortably inside the gate; only the coverage flags
    are engineered, which is the failure mode that blocked cell 56.
    """
    rng = np.random.default_rng(seed)
    errors = rng.standard_normal(total)
    errors = (errors - errors.mean()) / errors.std(ddof=1)  # bias 0, SD 1 exactly
    flags = [True] * covered + [False] * (total - covered)
    contrasts = [
        {
            "covered": flag,
            "estimate": float(err),
            "truth": 0.0,
            "standard_error": 1.0,
            "null": False,
        }
        for flag, err in zip(flags, errors)
    ]
    return [{"contrasts": contrasts}]


def test_family_wise_multiplier_matches_sidak_and_is_backward_compatible() -> None:
    # 16 cells at family-wise 5% -> ~2.95; a single cell -> the plain 1.96; and
    # an unset budget leaves the raw Monte-Carlo multiplier untouched.
    assert round(_family_wise_multiplier(0.05, 16, 2.0), 3) == 2.948
    assert round(_family_wise_multiplier(0.05, 1, 2.0), 3) == 1.960
    assert _family_wise_multiplier(None, 16, 2.0) == 2.0
    # more cells -> stricter per-cell threshold
    assert _family_wise_multiplier(0.05, 40, 2.0) > _family_wise_multiplier(0.05, 16, 2.0)
    with pytest.raises(ValueError, match="coverage_family_wise_error"):
        _family_wise_multiplier(1.5, 16, 2.0)


def test_cell_gate_multiplier_override_rescues_cell_56_coverage() -> None:
    # Cell 56's exact held-out coverage: 3771/4000 = 0.94275, which trips the raw
    # two-sided 2-sigma gate but clears the family-wise-corrected multiplier.
    records = _coverage_cell(3771, 4000)
    raw_passed, raw_audit = _cell_gate(records, _GATE_METRICS)
    assert raw_passed is False
    assert round(raw_audit["coverage"], 5) == 0.94275
    corrected = _family_wise_multiplier(0.05, 16, 2.0)
    fixed_passed, fixed_audit = _cell_gate(records, _GATE_METRICS, multiplier=corrected)
    assert fixed_passed is True
    assert fixed_audit["coverage_multiplier"] == corrected
    # a genuinely broken cell (90% coverage) still fails even after correction
    broken_passed, _ = _cell_gate(_coverage_cell(3600, 4000), _GATE_METRICS, multiplier=corrected)
    assert broken_passed is False


def test_family_wise_correction_controls_spurious_cell_failures() -> None:
    # Under perfect calibration the raw per-cell 2-sigma gate fails a family of 16
    # cells the majority of the time; the Sidak-corrected multiplier holds it near
    # the 5% budget while retaining power against a truly broken cell.
    rng = np.random.default_rng(20260721)
    counts = np.array([4000, 4000, 4000, 4000] + [2000] * 12)
    m = len(counts)
    corrected = _family_wise_multiplier(0.05, m, 2.0)
    trials = 4000
    mcse = np.sqrt(0.95 * 0.05 / counts)
    draws = rng.binomial(counts, 0.95, size=(trials, m)) / counts
    raw_family_fail = np.mean(np.any(np.abs(draws - 0.95) > 2.0 * mcse, axis=1))
    corrected_family_fail = np.mean(np.any(np.abs(draws - 0.95) > corrected * mcse, axis=1))
    assert raw_family_fail > 0.4          # the defect: majority of clean runs fail
    assert corrected_family_fail < 0.12   # budget restored (target 0.05)
    # power: a broken 0.90 cell (n=4000) is still detected essentially always
    broken = rng.binomial(4000, 0.90, size=trials) / 4000
    assert np.mean(np.abs(broken - 0.95) > corrected * np.sqrt(0.95 * 0.05 / 4000)) > 0.99
