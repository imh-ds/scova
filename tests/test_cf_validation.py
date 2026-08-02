import gzip
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from benchmarks import aggregate_cf_campaign, cf_inference_campaign
from benchmarks.cf_external_validation import (
    KnownRandomizationClassifier,
    SelectedOutcomeRegressor,
    TreatmentSpecificOutcomeRegressor,
    fixed_nuisance_score,
)
from benchmarks.cf_reference_campaign import (
    _declaration,
    plasmode_source_checksum,
    run_campaign,
    run_shard,
    simulate_plasmode_cell,
    simulate_reference_cell,
    write_deterministic_gzip,
)
from scova import SCOVA
from scova._aipw import assemble_aipw
from scova.cf import (
    AnalysisMode,
    CFSupportProfile,
    CFValidationProtocol,
    EstimatedAssignment,
    KnownAssignment,
    SeedPartition,
    _numerical_identity,
    canonical_checksum,
)
from scova.estimator import _scaled
from scova.simulate import generate_data

sys.path.insert(0, str(Path("scripts").resolve()))

from scripts.audit_cf_pilot import audit_pilot
from scripts.calibrate_cf_support import (
    _candidate_enrichments,
    _cell_gate,
    _family_wise_multiplier,
    _risk_ratio_lower_bound,
    _screening_cell_gate,
    _selection_z,
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
V8_SPEC = Path("benchmarks/specs/cf_reference_v8.json")
V9_SPEC = Path("benchmarks/specs/cf_reference_v9.json")
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


def test_v8_spec_adds_robust_enrichment_margin_selection() -> None:
    protocol = CFValidationProtocol.load(V8_SPEC)
    assert protocol.protocol_id == "cf-randomized-continuous-aipw-unnormalized-v8"
    assert protocol.checksum == (
        "9afacfbe9fb7be9968b18b47ad5a57ad2522c6baee35035d28e4dcdad56370dc"
    )
    # The v8 change is held-out-blind: it reuses the v7 frozen development,
    # external, and inference sources unchanged, and keeps the preregistered
    # acceptance thresholds.
    assert protocol.calibration_source == CFValidationProtocol.load(V7_SPEC).calibration_source
    assert protocol.external_source == CFValidationProtocol.load(V7_SPEC).external_source
    assert protocol.inference_source == CFValidationProtocol.load(V7_SPEC).inference_source
    assert protocol.metrics["minimum_unstable_risk_ratio"] == 2
    assert protocol.metrics["unstable_risk_ratio_selection_confidence"] == 0.95
    assert CFValidationProtocol.from_dict(protocol.to_dict()).checksum == protocol.checksum


def test_risk_ratio_lower_bound_is_conservative_and_tightens_with_evidence() -> None:
    z = _selection_z({"unstable_risk_ratio_selection_confidence": 0.95})
    assert z is not None
    # Same 2x point risk ratio, but ten times the evidence -> a tighter (higher)
    # lower bound, so selection prefers the better-supported rule.
    sparse = _risk_ratio_lower_bound(20.0, 100.0, 10.0, 100.0, z)
    dense = _risk_ratio_lower_bound(200.0, 1000.0, 100.0, 1000.0, z)
    assert 0.0 < sparse < 2.0  # the point estimate (2.0) is not trusted on thin evidence
    assert sparse < dense < 2.0
    # Degenerate arms never raise and never spuriously pass.
    assert _risk_ratio_lower_bound(0.0, 0.0, 0.0, 0.0, z) == 0.0


def test_candidate_enrichment_margin_ranks_robust_rule_above_a_boundary_rule() -> None:
    def record(*, supported: bool, bad: bool) -> dict:
        feature = 0.25 if supported else 0.75
        return {
            "support_features": {
                "minimum_ess_ratio": 0.75,
                "maximum_normalized_weight": feature,
                "maximum_top_one_percent_weight_share": feature,
                "maximum_absolute_weighted_balance_difference": feature,
                "maximum_influence_top_one_percent_share": feature,
                "maximum_seed_standardized_departure": feature,
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

    # Both rules share the same feature split; the difference is how much unstable
    # evidence backs the enrichment.  Both clear the 2x point gate identically.
    thresholds = {
        "minimum_ess_ratio": 0.5,
        "maximum_normalized_weight": 0.5,
        "maximum_top_one_percent_weight_share": 0.5,
        "maximum_absolute_weighted_balance_difference": 0.5,
        "maximum_influence_top_one_percent_share": 0.5,
        "maximum_seed_standardized_departure": 0.5,
    }
    def make(n_sup: int, n_sup_bad: int, n_uns: int, n_uns_bad: int) -> list[dict]:
        return [record(supported=True, bad=i < n_sup_bad) for i in range(n_sup)] + [
            record(supported=False, bad=i < n_uns_bad) for i in range(n_uns)
        ]

    # Both rules have supported bad-rate 0.05 and unstable bad-rate 0.12 (point
    # risk ratio 2.4); "thick" is backed by ten times the evidence.
    thin = make(200, 10, 25, 3)
    thick = make(2000, 100, 250, 30)
    metrics = {
        "minimum_unstable_risk_ratio": 2.0,
        "minimum_unstable_absolute_enrichment": 0.05,
        "unstable_risk_ratio_selection_confidence": 0.95,
    }
    (thin_enrichment,) = _candidate_enrichments(thin, [thresholds], metrics)
    (thick_enrichment,) = _candidate_enrichments(thick, [thresholds], metrics)
    assert thin_enrichment["passed"] and thick_enrichment["passed"]
    assert thin_enrichment["selection_confidence"] == 0.95
    # Both report a lower bound below their point estimate; the better-supported
    # rule earns the higher bound, which is the key that v8 selection ranks on.
    assert thin_enrichment["risk_ratio_lower_bound"] < thin_enrichment["risk_ratio"]
    assert (
        thin_enrichment["risk_ratio_lower_bound"]
        < thick_enrichment["risk_ratio_lower_bound"]
    )


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
    # The reference is the estimator's linear learner, which standardizes its
    # covariates; comparing against a bare Ridge would assert the adapter uses
    # a different learner from the one SCOVA actually fits.
    expected = np.array(
        [
            _scaled("Ridge", Ridge(alpha=1.0)).fit(features[:4], outcome[:4]).predict([[4.0]])[0],
            _scaled("Ridge", Ridge(alpha=1.0)).fit(features[4:], outcome[4:]).predict([[4.0]])[0],
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
    if path in _numerical_identity._EXTERNAL_NUMERICAL_PATHS:
        fitted = "None" if commit == "after-comparator-fix" else "known"
        return f"def comparator(): return {fitted}\n".encode()
    if path == "scripts/calibrate_cf_support.py":
        rule = "'v9'" if commit == "after-selection-rule" else "'v8'"
        return f"def select(): return {rule}\n".encode()
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


def test_campaign_fingerprint_excludes_the_external_comparators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comparator fix must not force a 128-shard calibration lane to re-run.

    cf_external_agreement and cf_external_validation cannot alter a calibration
    record -- they read a fitted result and compare it against DoubleML. Before
    the campaign kind existed the prerequisite check compared calibration
    commits verbatim, so editing them cost a full re-run for nothing. What DOES
    determine calibration evidence still has to bind.
    """
    monkeypatch.setattr(_numerical_identity, "_committed_file", _fake_numerical_source)
    fingerprint = _numerical_identity.cf_numerical_fingerprint
    assert fingerprint("before-comparator-fix", "campaign") == fingerprint(
        "after-comparator-fix", "campaign"
    )
    # The same edit must still invalidate external-agreement evidence.
    assert fingerprint("before-comparator-fix", "external") != fingerprint(
        "after-comparator-fix", "external"
    )
    # Selection and campaign sources remain binding for calibration.
    assert fingerprint("before-selection-rule", "campaign") != fingerprint(
        "after-selection-rule", "campaign"
    )
    assert fingerprint("before-governance", "campaign") == fingerprint(
        "after-governance", "campaign"
    )


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
        for flag, err in zip(flags, errors, strict=True)
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


def test_v9_consolidates_robust_margin_and_family_wise_coverage_gate() -> None:
    protocol = CFValidationProtocol.load(V9_SPEC)
    assert protocol.protocol_id == "cf-randomized-continuous-aipw-unnormalized-v9"
    assert protocol.checksum == (
        "c60b3780680bbbfa08c2d31ed3ff51c4a28aec0d14f013c08daeaec243d7330b"
    )
    # both improvements are active, and both reused-evidence sources are unchanged
    assert protocol.metrics["unstable_risk_ratio_selection_confidence"] == 0.95
    assert protocol.metrics["coverage_family_wise_error"] == 0.05
    v8 = CFValidationProtocol.load(V8_SPEC)
    assert protocol.calibration_source == v8.calibration_source
    assert protocol.external_source == v8.external_source
    assert protocol.inference_source == v8.inference_source
    assert CFValidationProtocol.from_dict(protocol.to_dict()).checksum == protocol.checksum


_OBSERVATIONAL_BASE = {
    "allocation": "balanced",
    "effect": "constant",
    "learner": "adaptive",
    "n_covariates": 5,
    "n_groups": 2,
    "n_per_group": 120,
    "noise": "normal",
    "support": "strong",
    "surface": "smooth-nonlinear",
}


def test_cells_without_confounding_stay_randomized() -> None:
    """The observational path must be additive: no confounding key, no change."""
    protocol = CFValidationProtocol.load(V9_SPEC)
    for cell in protocol.retained_cells[:6]:
        generated = simulate_reference_cell(cell, seed=31)
        assert generated.unit_probabilities is None
        declaration = _declaration(generated, cell, include_stability=False)
        assert declaration.mode is AnalysisMode.RANDOMIZED
        assert isinstance(declaration.assignment, KnownAssignment)


def test_confounded_cells_declare_estimated_assignment() -> None:
    cell = {**_OBSERVATIONAL_BASE, "confounding": "strong", "confounding_form": "nonlinear"}
    generated = simulate_reference_cell(cell, seed=31)
    assert generated.unit_probabilities is not None
    assert generated.unit_probabilities.shape == (len(generated.data), 2)
    np.testing.assert_allclose(generated.unit_probabilities.sum(axis=1), 1.0)
    declaration = _declaration(generated, cell, include_stability=False)
    assert declaration.mode is AnalysisMode.OBSERVATIONAL_CAUSAL
    assert isinstance(declaration.assignment, EstimatedAssignment)
    # The reference estimator refuses unless both nuisance strategies agree.
    assert declaration.assignment.nuisance_strategy == declaration.outcome_nuisance_strategy


def test_overlap_factor_is_the_only_lever_on_common_support() -> None:
    """Confounding strength must not degrade overlap on its own.

    The standardized nonlinear signal is heavy-tailed, so unbounded logits would
    drive units to propensity ~0 regardless of the overlap factor, leaving
    nothing for support screening to distinguish.
    """
    def worst_propensity(overlap: str) -> float:
        cell = {
            **_OBSERVATIONAL_BASE,
            "confounding": "strong",
            "confounding_form": "nonlinear",
            "overlap": overlap,
        }
        return min(
            float(simulate_reference_cell(cell, seed=900 + rep).unit_probabilities.min())
            for rep in range(5)
        )

    assert worst_propensity("full") > 0.05
    assert worst_propensity("partial") < 0.01
    assert worst_propensity("poor") < 0.01


V10_SPEC = Path("benchmarks/specs/cf_reference_v10.json")
V11_SPEC = Path("benchmarks/specs/cf_reference_v11.json")


def test_v10_is_a_wholly_observational_protocol_reusing_no_evidence() -> None:
    protocol = CFValidationProtocol.load(V10_SPEC)
    assert protocol.protocol_id == "cf-observational-continuous-aipw-unnormalized-v10"
    assert protocol.checksum == (
        "a1c54a76e1fb8401f8d1b7eea50c16ccd77cd9e0e1f0afdb08fe28594c6caccb"
    )
    assert protocol.reference_profile["mode"] == "observational-causal"
    assert protocol.reference_profile["assignment"] == "estimated"

    # Every lane must be confounded. A randomized cell here would let
    # randomization-supported evidence vouch for an assumption-dependent
    # causal profile, which is exactly what the profile regime lock forbids.
    lanes = (
        list(protocol.retained_cells)
        + list(protocol.plasmode_cells)
        + list(protocol.external_cells)
        + [reference["cell"] for reference in protocol.inference_cells]
    )
    assert lanes
    for cell in lanes:
        assert cell.get("confounding") not in (None, "none")
        assert cell.get("confounding_form") in {"linear", "nonlinear"}
        assert cell.get("overlap") in {"full", "partial", "poor"}

    # v9 inherited external and inference evidence from randomized v5/v6 runs.
    # Those say nothing about estimated assignment, so v10 generates its own.
    assert protocol.calibration_source is None
    assert protocol.external_source is None
    assert protocol.inference_source is None
    assert protocol.candidate_source is None
    assert CFValidationProtocol.from_dict(protocol.to_dict()).checksum == protocol.checksum


def test_v10_design_is_reproducible_and_pairwise_complete() -> None:
    from scripts.generate_cf_v10_design import build_spec

    protocol = CFValidationProtocol.load(V10_SPEC)
    regenerated = build_spec()
    assert regenerated["retained_cells"] == [dict(cell) for cell in protocol.retained_cells]
    provenance = regenerated["design_selection"]
    assert provenance["pairwise_pairs_covered"] == provenance["pairwise_pairs_total"]


def test_v10_retained_cells_can_all_be_fitted() -> None:
    """No design point may expect fewer units in its smallest arm than it can fit.

    `weak` support at small n expects 1.2-1.8 units in the smallest arm, below
    the n_splits floor, so such a cell would refuse on most draws and calibrate
    thresholds against noise. `structural-failure` is exempt in spirit -- it
    empties a group on purpose -- but it is constrained the same way here
    because the emptying happens after assignment, not through the baseline.
    """
    from scripts.generate_cf_v10_design import MINIMUM_EXPECTED_ARM, expected_smallest_arm

    protocol = CFValidationProtocol.load(V10_SPEC)
    for cell in protocol.retained_cells:
        arm = expected_smallest_arm(dict(cell))
        assert arm >= MINIMUM_EXPECTED_ARM, f"{dict(cell)} expects only {arm:.1f}"
    # The corner that prompted the constraint is gone.
    assert not [
        cell
        for cell in protocol.retained_cells
        if cell["allocation"] == "rare" and cell["n_per_group"] <= 30
    ]


def test_v10_plasmode_checksums_match_the_pinned_stack() -> None:
    """Dataset checksums hash bundled scikit-learn data, not this repository.

    Generating them on whatever the author has installed freezes a spec the
    pinned campaign can never satisfy: the source-truth check recomputes them
    under scikit-learn 1.6.1 and would mismatch on the first real run.
    """
    from scripts.generate_cf_v10_design import PINNED_DATASET_CHECKSUMS

    v10 = CFValidationProtocol.load(V10_SPEC)
    v9 = CFValidationProtocol.load(V9_SPEC)
    assert v10.dataset_checksums == PINNED_DATASET_CHECKSUMS
    # v9's were frozen in the pinned environment, so they are the reference.
    assert v10.dataset_checksums == v9.dataset_checksums


def test_v10_seed_partitions_avoid_every_earlier_campaign() -> None:
    """Seeds must stay valid scikit-learn states, and v3-v9 crowd the ceiling."""
    v10 = CFValidationProtocol.load(V10_SPEC)
    lanes = [v10.pilot, v10.calibration, v10.validation, v10.external, v10.inference]
    assert all(lane is not None for lane in lanes)
    highest = max(lane.start + 60 * lane.count for lane in lanes if lane is not None)
    assert highest < 2**32
    for spec in (SPEC, V4_SPEC, V6_SPEC, V8_SPEC, V9_SPEC):
        other = CFValidationProtocol.load(spec)
        for mine in lanes:
            for theirs in (other.pilot, other.calibration, other.validation):
                assert mine is not None
                mine_end = mine.start + 60 * mine.count
                theirs_end = theirs.start + 60 * theirs.count
                assert mine_end <= theirs.start or theirs_end <= mine.start


@pytest.mark.parametrize(
    "spec_path",
    # v2 is a deliberately blocked historical artifact that does not load.
    sorted(
        path
        for path in Path("benchmarks/specs").glob("cf_reference_v*.json")
        if "blocked" not in path.stem
    ),
    ids=lambda path: path.stem,
)
def test_every_spec_declares_quantiles_the_calibrator_can_read(spec_path: Path) -> None:
    """threshold_quantiles has a fixed two-key shape, and nothing enforced it.

    calibrate_cf_support reads quantiles["minimum_ess_ratio"] for the lower grid
    and quantiles["upper_metrics"] as one shared grid for every upper feature.
    The branch that reads them is marked no-cover, so a per-metric mapping --
    valid JSON, wrong shape -- survived freeze_check, a full pilot and 128
    calibration shards before failing with KeyError.
    """
    protocol = CFValidationProtocol.load(spec_path)
    quantiles = protocol.threshold_quantiles
    if quantiles is None:
        return  # the calibrator falls back to its built-in grids
    from scripts.calibrate_cf_support import LOWER_FEATURES

    allowed = {"upper_metrics", *LOWER_FEATURES}
    assert "minimum_ess_ratio" in quantiles and "upper_metrics" in quantiles
    assert set(quantiles) <= allowed, f"{spec_path.name} declares {sorted(quantiles)}"
    for name, grid in quantiles.items():
        assert grid, f"{name} grid is empty"
        assert all(0.0 <= float(q) <= 1.0 for q in grid), f"{name} holds non-quantiles"
        assert list(grid) == sorted(grid), f"{name} grid is not ascending"


@pytest.mark.parametrize(
    ("spec_path", "expected"),
    [(V9_SPEC, "known"), (V10_SPEC, "fitted")],
    ids=["v9-randomized", "v10-observational"],
)
def test_comparators_only_get_a_known_propensity_when_the_design_supplies_one(
    spec_path: Path, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under estimated assignment the comparators must fit their own propensity.

    `generated.probabilities` is the marginal allocation, constant across
    units. That IS the truth for a randomized design. Under v10 the propensity
    varies by unit -- on the strongest external cell the true values span
    [0.12, 0.88] against a constant 0.5 -- so passing it would misspecify both
    comparators into effectively unadjusted estimators and compare SCOVA-CF's
    confounding-adjusted answer against them.
    """
    from benchmarks import cf_external_agreement as agreement
    from benchmarks.cf_external_validation import ExternalAgreement

    seen: list[object] = []

    def _capture(*_args: object, known_probabilities: object, **_kwargs: object):
        seen.append(known_probabilities)
        return ExternalAgreement("DoubleMLAPOS", "0", "blocked/missing-dependency")

    protocol = CFValidationProtocol.load(spec_path)
    monkeypatch.setattr(agreement, "_environment", lambda: dict(protocol.software))
    monkeypatch.setattr(agreement, "doubleml_apos", _capture)
    monkeypatch.setattr(agreement, "econml_drlearner", _capture)
    agreement.run_external_agreement(protocol, replications=1, max_cells=1)

    assert seen, "comparators were never invoked"
    if expected == "fitted":
        assert all(value is None for value in seen), seen
    else:
        assert all(value is not None for value in seen), seen
        assert all(len(set(np.asarray(value).ravel().tolist())) >= 1 for value in seen)


def _records(*groups: tuple[int, str, int, float]) -> list[dict[str, object]]:
    """Build (cell, repetition) records whose per-unit mean is the given size.

    Signs alternate across repetitions so the offset averages to zero: these
    fixtures represent two implementations that differ only by fold noise,
    which is what independent splits produce when both are correct.
    """
    records: list[dict[str, object]] = []
    for cell_index, stratum, count, magnitude in groups:
        for repetition in range(count):
            sign = 1.0 if repetition % 2 else -1.0
            records.append(
                {
                    "cell_index": cell_index,
                    "repetition": repetition,
                    "stratum": stratum,
                    "differences": [sign * magnitude, sign * magnitude],
                }
            )
    return records


def test_offset_z_ignores_fold_noise_but_catches_a_systematic_shift() -> None:
    """The statistic independent folds require.

    Under different splits an individual difference carries fold noise even
    when both implementations are right -- scatter alone was measured at a
    pooled mean |d| of 0.6324 against the old 0.25 tolerance. Random noise
    averages out across replications; a systematic offset does not.
    """
    from benchmarks.cf_external_agreement import _offset_z

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.63, size=50).tolist()
    assert abs(_offset_z(noise)) < 3.0
    shifted = [value + 0.5 for value in noise]
    assert abs(_offset_z(shifted)) > 3.0
    # Unestimable rather than infinite when there is nothing to average.
    assert _offset_z([0.4]) is None
    assert _offset_z([0.4, 0.4, 0.4]) is None


def test_end_to_end_passes_on_fold_noise_and_fails_on_an_offset() -> None:
    """Scatter alone must not fail the lane, and a real shift must."""
    from benchmarks.cf_external_agreement import _summary

    noisy = _records((0, "k=2,linear", 40, 0.63), (1, "k=3,adaptive", 40, 0.63))
    passing = _summary(
        "DoubleMLAPOS", noisy, [], lane_complete=True, critical_z=3.0,
        minimum_informative_fraction=1.0,
    )
    assert passing["status"] == "complete"
    assert passing["maximum_absolute_offset_z"] < 3.0
    # The old gate scored |difference| against 0.25 and would have refused this.
    assert all(
        abs(value) > 0.25
        for record in noisy
        for value in record["differences"]  # type: ignore[attr-defined]
    )

    shifted = [
        {**record, "differences": [value + 0.5 for value in record["differences"]]}
        for record in noisy
    ]
    failing = _summary(
        "DoubleMLAPOS", shifted, [], lane_complete=True, critical_z=3.0,
        minimum_informative_fraction=1.0,
    )
    assert failing["status"] == "blocked/agreement-tolerance"
    assert failing["breaching_strata"] == ["k=2,linear", "k=3,adaptive"]


def test_end_to_end_requires_every_cell_to_be_informative() -> None:
    """Under independent folds there is no legitimate route to identity.

    A degenerate cell therefore means the fold independence did not take
    effect, which is the silent-harness failure this lane has produced twice --
    once with a constant propensity handed to both comparators, once when the
    one-vs-rest estimator change made SCOVA and DoubleMLAPOS the same fit.
    """
    from benchmarks.cf_external_agreement import _summary

    records = _records((0, "k=2,linear", 40, 0.63), (1, "k=3,adaptive", 40, 1e-15))
    summary = _summary(
        "DoubleMLAPOS", records, [], lane_complete=True, critical_z=3.0,
        minimum_informative_fraction=1.0,
    )

    assert summary["status"] == "blocked/lane-degenerate"
    assert summary["informative_cell_fraction"] == 0.5
    assert summary["degenerate_cell_count"] == 1
    # Still not judged that way on a truncated smoke run.
    truncated = _summary(
        "DoubleMLAPOS", records, [], lane_complete=False, critical_z=3.0,
        minimum_informative_fraction=1.0,
    )
    assert truncated["status"] == "incomplete/degenerate-subset"


def test_one_replication_does_not_refuse_the_smoke_lane() -> None:
    """The r1 smoke failure, pinned.

    `external_smoke` runs `--replications 1 --max-cells 1`, so every stratum
    holds one unit and no offset can be estimated. Treating unestimable the
    same as breaching refused the canary every time -- and it is the cheapest
    check in the campaign, the one that caught the r7 software block in two
    minutes. An estimable offset that is too large still fails on any lane; it
    is only the unestimable case that a truncated run is excused.
    """
    from benchmarks.cf_external_agreement import SMOKE_ADMISSIBLE_STATUSES, _summary

    single = [
        {
            "cell_index": 0,
            "repetition": 0,
            "stratum": "k=2,linear",
            "differences": [-0.198, -0.247],
        }
    ]
    smoke = _summary(
        "DoubleMLAPOS", single, [], lane_complete=False, critical_z=2.236,
        minimum_informative_fraction=1.0,
    )

    assert smoke["status"] == "incomplete/unscored-subset"
    assert smoke["status"] in SMOKE_ADMISSIBLE_STATUSES
    assert smoke["unestimable_strata"] == ["k=2,linear"]
    assert smoke["breaching_strata"] == []
    # The independent folds did their job: the cell is informative, which is
    # what the smoke run is there to demonstrate.
    assert smoke["informative_cell_fraction"] == 1.0
    assert smoke["degenerate_cell_count"] == 0

    # The same shape on the COMPLETE lane means no observed spread, and must
    # fail closed rather than be excused.
    frozen = _summary(
        "DoubleMLAPOS", single, [], lane_complete=True, critical_z=2.236,
        minimum_informative_fraction=1.0,
    )
    assert frozen["status"] == "blocked/agreement-tolerance"
    assert frozen["status"] not in SMOKE_ADMISSIBLE_STATUSES


def test_a_real_breach_still_fails_a_truncated_lane() -> None:
    """Being excused for size must not excuse a measurable disagreement."""
    from benchmarks.cf_external_agreement import _summary

    records = [
        {
            "cell_index": 0,
            "repetition": index,
            "stratum": "k=2,linear",
            "differences": [2.0 + 0.01 * index],
        }
        for index in range(12)
    ]
    summary = _summary(
        "DoubleMLAPOS", records, [], lane_complete=False, critical_z=2.236,
        minimum_informative_fraction=1.0,
    )

    assert summary["breaching_strata"] == ["k=2,linear"]
    assert summary["status"] == "blocked/agreement-tolerance"


def test_comparator_folds_are_independent_of_scova_folds() -> None:
    """Different splits, same group-stratified guarantee.

    A comparator handed a training fold missing an arm cannot fit a propensity
    for it, so the independence has to come from the same construction under a
    different seed rather than from an arbitrary reshuffle.
    """
    from benchmarks.cf_external_agreement import _agreement_policy, _comparator_folds
    from benchmarks.cf_reference_campaign import _declaration, simulate_reference_cell
    from scova.cf import SCOVACF

    protocol = CFValidationProtocol.load(V11_SPEC)
    agreement = _agreement_policy(protocol)
    assert agreement["comparator_folds"] == "independent"

    cell = dict(protocol.external_cells[4])
    generated = simulate_reference_cell(cell, seed=1_600_000_004)
    declaration = _declaration(generated, cell, include_stability=False)
    result = SCOVACF().analyze(generated.data, declaration)
    codes = np.array(
        [result.group_labels.index(value) for value in generated.data["group"]]
    )
    folds = _comparator_folds(generated.data, declaration, codes, agreement)

    assert not np.array_equal(folds, result.fold_assignments)
    assert set(np.unique(folds)) == set(np.unique(result.fold_assignments))
    # Every fold must still contain every arm on both sides of the split.
    for fold in np.unique(folds):
        assert set(np.unique(codes[folds != fold])) == set(np.unique(codes))


def test_earlier_protocols_keep_shared_folds_and_their_checksums() -> None:
    """v3-v10 sealed their evidence under shared folds; that must not move."""
    from benchmarks.cf_external_agreement import _agreement_policy

    for spec_path in (SPEC, V9_SPEC, V10_SPEC):
        protocol = CFValidationProtocol.load(spec_path)
        assert protocol.external_agreement is None
        assert _agreement_policy(protocol)["comparator_folds"] == "scova"
    assert CFValidationProtocol.load(V10_SPEC).checksum == (
        "a1c54a76e1fb8401f8d1b7eea50c16ccd77cd9e0e1f0afdb08fe28594c6caccb"
    )


def test_end_to_end_differences_stay_bound_to_the_cell_they_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cell attribution must survive the lane, not just the summary.

    DoubleMLAPOS contributes one difference per arm and EconML.DRLearner one
    per contrast, so the two accumulate at different rates within the same
    replication. If the recorded cell index drifted out of step with the
    differences, degeneracy would be judged against the wrong cell and the
    whole check would silently mis-attribute.
    """
    from benchmarks import cf_external_agreement as agreement
    from benchmarks.cf_external_validation import ExternalAgreement

    protocol = CFValidationProtocol.load(V10_SPEC)
    monkeypatch.setattr(agreement, "_environment", lambda: dict(protocol.software))

    def _arms(*args: object, **_kwargs: object) -> ExternalAgreement:
        # DoubleMLAPOS reports one estimate per arm. The value is irrelevant,
        # the bookkeeping is not.
        groups = int(np.max(np.asarray(args[2])) + 1)
        return ExternalAgreement("stub", "0", "complete", estimates=(1e3,) * groups)

    def _contrasts(*args: object, **_kwargs: object) -> ExternalAgreement:
        # EconML.DRLearner reports one estimate per contrast against arm 0.
        groups = int(np.max(np.asarray(args[2])) + 1)
        return ExternalAgreement("stub", "0", "complete", estimates=(1e3,) * (groups - 1))

    monkeypatch.setattr(agreement, "doubleml_apos", _arms)
    monkeypatch.setattr(agreement, "econml_drlearner", _contrasts)
    evidence = agreement.run_external_agreement(protocol, replications=2, max_cells=3)

    for summary in evidence["end_to_end"]["implementations"]:
        assert summary["total_unit_count"] == 3 * 2
        assert [row["cell_index"] for row in summary["cells"]] == [0, 1, 2]
        # Every cell got the same number of replications, so a drifting index
        # would show up as an uneven split.
        assert len({row["comparison_count"] for row in summary["cells"]}) == 1
        # Strata are read off the cell, so they must match the design.
        assert set(summary["strata"]) == {
            f"k={int(dict(cell)['n_groups'])},{dict(cell)['learner']}"
            for cell in protocol.external_cells[:3]
        }


@pytest.mark.parametrize(
    "spec_path",
    sorted(
        path
        for path in Path("benchmarks/specs").glob("cf_reference_v*.json")
        if "blocked" not in path.stem
    ),
    ids=lambda path: path.stem,
)
def test_every_spec_declares_the_whole_frozen_environment(spec_path: Path) -> None:
    """The software block must name every package the frozen lanes import.

    cf_external_agreement compares the entire installed environment against
    this block by dict equality, so a spec that declares a subset can never
    match and that lane fails outright -- after the freeze, after a full
    calibration. v10 shipped three of the seven packages and blocked there.
    doubleml and econml are the comparators external agreement is measured
    against, so leaving them undeclared also leaves them free to drift.
    """
    from benchmarks.cf_external_agreement import _environment

    protocol = CFValidationProtocol.load(spec_path)
    declared = dict(protocol.software)
    assert set(declared) == set(_environment()), (
        f"{spec_path.name} declares {sorted(declared)}, "
        f"but external agreement inspects {sorted(_environment())}"
    )
    pinned = dict(
        line.split("==", 1)
        for line in Path("benchmarks/requirements-cf-validation.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if "==" in line
    )
    for package, version in pinned.items():
        assert declared.get(package) == version, (
            f"{spec_path.name} declares {package} {declared.get(package)!r} "
            f"against a pinned {version!r}"
        )


def test_linear_learners_are_invariant_to_covariate_scale() -> None:
    """Rescaling a column must not change the answer.

    Unscaled, LogisticRegression and Ridge are badly conditioned on covariates
    whose columns differ by orders of magnitude: the solver never converges and
    its path depends on floating-point ordering, so identical campaigns
    disagreed on a third of the breast-cancer plasmode contrasts. The penalty
    is also expressed in each covariate's own units, so an unscaled fit depends
    on whether a column was recorded in millimetres or kilometres.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    outcome = 2.0 * x[:, 0] - x[:, 1] + rng.normal(size=200)
    # The same covariates, one column recorded in different units.
    stretched = x * np.array([1.0, 1e4, 1e-4])

    propensity = SCOVA._linear_propensity_model()
    a = clone(propensity).fit(x, y).predict_proba(x)
    b = clone(propensity).fit(stretched, y).predict_proba(stretched)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)

    regressor = SCOVA._linear_outcome_model()
    c = clone(regressor).fit(x, outcome).predict(x)
    d = clone(regressor).fit(stretched, outcome).predict(stretched)
    np.testing.assert_allclose(c, d, rtol=1e-6, atol=1e-8)


def _enrichment_record(*, supported: bool, se_ratio: float) -> dict:
    """A covered contrast whose interval is `se_ratio` times the unadjusted one."""
    features = {
        "minimum_ess_ratio": 0.9 if supported else 0.01,
        "maximum_normalized_weight": 0.1 if supported else 0.9,
        "maximum_top_one_percent_weight_share": 0.1 if supported else 0.9,
        "maximum_absolute_weighted_balance_difference": 0.1 if supported else 5.0,
        "maximum_influence_top_one_percent_share": 0.1 if supported else 0.9,
        "maximum_seed_standardized_departure": 0.1 if supported else 5.0,
    }
    return {
        "refused": False,
        "support_features": features,
        "contrasts": [
            {
                "group_code": 1,
                "covered": True,
                "estimate": 0.0,
                "truth": 0.0,
                "standard_error": se_ratio,
            }
        ],
        "benchmarks": {
            "unadjusted": {"contrasts": [{"group_code": 1, "standard_error": 1.0}]}
        },
    }


_ENRICHMENT_THRESHOLDS = {
    "minimum_ess_ratio": 0.5,
    "maximum_normalized_weight": 0.5,
    "maximum_top_one_percent_weight_share": 0.5,
    "maximum_absolute_weighted_balance_difference": 1.0,
    "maximum_influence_top_one_percent_share": 0.5,
    "maximum_seed_standardized_departure": 1.0,
}
_ENRICHMENT_METRICS = {
    "minimum_unstable_risk_ratio": 2.0,
    "minimum_unstable_absolute_enrichment": 0.05,
}


def test_enrichment_counts_a_vacuous_interval_against_the_gate() -> None:
    """Every contrast here is covered; the unstable ones are merely useless.

    Scoring error rate alone ranks them as the safest cells, which is why the
    gate inverted on the observational lane. With the metric declared, an
    interval far wider than the unadjusted benchmark's counts as bad.
    """
    records = [_enrichment_record(supported=True, se_ratio=1.0) for _ in range(20)]
    records += [_enrichment_record(supported=False, se_ratio=100.0) for _ in range(20)]

    without = _unstable_enrichment(records, _ENRICHMENT_THRESHOLDS, _ENRICHMENT_METRICS)
    assert without["unstable_bad_rate"] == 0.0
    assert without["passed"] is False

    with_metric = _unstable_enrichment(
        records,
        _ENRICHMENT_THRESHOLDS,
        {**_ENRICHMENT_METRICS, "maximum_standard_error_ratio": 10.0},
    )
    assert with_metric["unstable_bad_rate"] == 1.0
    assert with_metric["supported_bad_rate"] == 0.0
    assert with_metric["passed"] is True


def test_enrichment_ignores_contrasts_with_no_usable_benchmark() -> None:
    """A missing benchmark must make the gate harder to pass, never easier."""
    records = [_enrichment_record(supported=True, se_ratio=1.0) for _ in range(20)]
    for index in range(20):
        record = _enrichment_record(supported=False, se_ratio=100.0)
        if index < 10:  # half lose their comparator entirely
            record["benchmarks"] = {"unadjusted": {"contrasts": []}}
        records.append(record)

    result = _unstable_enrichment(
        records,
        _ENRICHMENT_THRESHOLDS,
        {**_ENRICHMENT_METRICS, "maximum_standard_error_ratio": 10.0},
    )
    # Only the ten with a benchmark are flagged, so the rate halves rather than
    # the unusable ones being counted as bad by default.
    assert result["unstable_bad_rate"] == 0.5


def test_both_enrichment_implementations_score_identically() -> None:
    """The vectorized path is what calibration calls; the scalar one is what
    most tests call. They each inlined the badness rule, so a fix applied to
    the scalar version alone passed its tests and changed nothing on a real
    run -- 590 candidates, still zero screened, with byte-identical numbers.
    """
    records = [_enrichment_record(supported=True, se_ratio=1.0) for _ in range(15)]
    records += [_enrichment_record(supported=False, se_ratio=100.0) for _ in range(15)]
    metrics = {**_ENRICHMENT_METRICS, "maximum_standard_error_ratio": 10.0}

    scalar = _unstable_enrichment(records, _ENRICHMENT_THRESHOLDS, metrics)
    vector = _candidate_enrichments(records, [_ENRICHMENT_THRESHOLDS], metrics)[0]
    for field in (
        "supported_bad_rate",
        "unstable_bad_rate",
        "absolute_enrichment",
        "risk_ratio",
        "passed",
    ):
        assert scalar[field] == vector[field], field
    assert vector["passed"] is True

    # and they must agree when the metric is absent, too
    plain_scalar = _unstable_enrichment(records, _ENRICHMENT_THRESHOLDS, _ENRICHMENT_METRICS)
    plain_vector = _candidate_enrichments(records, [_ENRICHMENT_THRESHOLDS], _ENRICHMENT_METRICS)[0]
    assert plain_scalar["unstable_bad_rate"] == plain_vector["unstable_bad_rate"] == 0.0
    assert plain_vector["passed"] is False


def test_arm_density_is_screened_and_separates_the_failing_cells() -> None:
    """The v10 blocker was bias concentrated in the smallest arm.

    Calibration finds the threshold rather than the spec asserting one, so the
    feature has to be present in the records and swept by the candidate family.
    Protocols that predate it screen on ESS alone.
    """
    from scripts.calibrate_cf_support import LOWER_FEATURES, _active_lower_features

    assert LOWER_FEATURES == ("minimum_ess_ratio", "minimum_arm_units_per_covariate")
    protocol = CFValidationProtocol.load(V10_SPEC)
    assert "minimum_arm_units_per_covariate" in protocol.threshold_quantiles
    # Density also bounds the profile's claimed scope. A threshold alone cannot
    # help: a cell screened out still counts against the usefulness criterion,
    # so only narrowing eligibility removes it from the denominator.
    assert protocol.reference_profile["minimum_arm_units_per_covariate"] == 10.0

    # records from a protocol that predates the feature must screen on ESS only
    legacy = [{"support_features": {"minimum_ess_ratio": 0.5}}]
    assert _active_lower_features(legacy) == ("minimum_ess_ratio",)

    # the failing plasmode cell and a healthy simulated one must be far apart
    from benchmarks.cf_reference_campaign import _support_features
    from scova.cf import SCOVACF, SCOVACFRefusal

    failing = dict(allocation="moderate", confounding="strong", confounding_form="linear",
                   dataset="breast-cancer", effect="null", learner="adaptive", n_groups=3,
                   n_per_group=100, noise="normal", overlap="full")
    generated = simulate_plasmode_cell(failing, seed=600_000)
    result = SCOVACF().analyze(
        generated.data, _declaration(generated, failing, include_stability=False)
    )
    assert not isinstance(result, SCOVACFRefusal)
    density = _support_features(result)["minimum_arm_units_per_covariate"]
    assert density < 3.0, density   # ~55 rows against 30 covariates


def _eligible_cell(n_groups: int, learner: str, n_per_group: int = 50) -> dict[str, object]:
    """A cell that clears every profile-eligibility term with no slack to spare.

    Balanced allocation puts 1/k of the sample in each arm, so n_per_group=50
    expects exactly 50 units in the smallest arm -- the `minimum_group_count`
    floor -- and 5 covariates put that at exactly the 10.0 density bound.
    """
    return {
        "allocation": "balanced",
        "confounding": "moderate",
        "confounding_form": "linear",
        "effect": "constant",
        "learner": learner,
        "n_covariates": 5,
        "n_groups": n_groups,
        "n_per_group": n_per_group,
        "noise": "normal",
        "overlap": "full",
        "support": "strong",
        "surface": "linear",
    }


def test_design_coverage_audit_flags_the_frozen_eligibility_gaps() -> None:
    """The v10 grid reached a full calibration with an empty claimed stratum.

    11 of 60 cells are profile-eligible and they fall 9 / 1 / 0 / 1 across
    (k, learner). The zero is (k=3, linear): no gate in the campaign could have
    detected a multi-arm defect under a linear learner, because eligibility is
    what decides the calibration denominator. This is the check that was
    missing, pinned against the design that needed it.
    """
    from scripts.generate_cf_v10_design import (
        design_coverage_failures,
        eligible_cells_by_stratum,
    )

    spec = json.loads(V10_SPEC.read_text(encoding="utf-8"))
    counts = eligible_cells_by_stratum(spec)

    assert counts == {(2, "linear"): 9, (2, "adaptive"): 1, (3, "linear"): 0, (3, "adaptive"): 1}
    # k=5 is outside maximum_group_count, so it is not a claimed stratum and
    # must not be scored as a gap.
    assert not [stratum for stratum in counts if stratum[0] == 5]
    failures = design_coverage_failures(spec)
    assert any("n_groups=3, learner=linear" in failure for failure in failures)
    assert all("pairwise" not in failure for failure in failures)


def test_design_coverage_audit_passes_when_every_claimed_stratum_is_populated() -> None:
    """The audit must be satisfiable, or it is only a way of always refusing."""
    from scripts.generate_cf_v10_design import design_coverage_failures

    spec = json.loads(V10_SPEC.read_text(encoding="utf-8"))
    retained = list(spec["retained_cells"])
    # Replace rather than append: the cell budget is part of the frozen schema.
    # Two distinct cells per claimed stratum, which is what the minimum asks for.
    retained[:8] = [
        _eligible_cell(groups, learner, n_per_group)
        for groups in (2, 3)
        for learner in ("linear", "adaptive")
        for n_per_group in (50, 80)
    ]
    spec["retained_cells"] = retained

    assert design_coverage_failures(spec) == []


def test_design_coverage_audit_requires_complete_pairwise_coverage() -> None:
    """The selection records its own coverage; nothing checked the claim held."""
    from scripts.generate_cf_v10_design import design_coverage_failures

    spec = json.loads(V10_SPEC.read_text(encoding="utf-8"))
    spec["design_selection"] = {
        **spec["design_selection"],
        "pairwise_pairs_covered": spec["design_selection"]["pairwise_pairs_total"] - 1,
    }

    assert any("pairwise coverage" in failure for failure in design_coverage_failures(spec))


def test_v11_grid_satisfies_the_coverage_audit_that_v10_fails() -> None:
    """The point of the rebuild, stated as the gate it has to pass.

    v10's eligible cells fell 9 / 1 / 0 / 1 across the claimed (n_groups,
    learner) strata because pairwise coverage optimizes marginals while
    eligibility is a conjunction, so eligible cells only ever arose by
    accident. v11 reserves the region first and spends what is left on
    coverage.
    """
    from scripts.generate_cf_v10_design import (
        MINIMUM_ELIGIBLE_CELLS_PER_STRATUM,
        design_coverage_failures,
        eligible_cells_by_stratum,
    )

    v11 = json.loads(V11_SPEC.read_text(encoding="utf-8"))
    v10 = json.loads(V10_SPEC.read_text(encoding="utf-8"))

    assert design_coverage_failures(v11) == []
    assert design_coverage_failures(v10), "v10 must still fail, or the audit proves nothing"
    counts = eligible_cells_by_stratum(v11)
    assert set(counts) == {(2, "linear"), (2, "adaptive"), (3, "linear"), (3, "adaptive")}
    assert min(counts.values()) >= MINIMUM_ELIGIBLE_CELLS_PER_STRATUM
    # The stratum that was empty is the one worth naming.
    assert counts[(3, "linear")] >= MINIMUM_ELIGIBLE_CELLS_PER_STRATUM


def test_v11_is_reproducible_and_collapses_no_factor() -> None:
    """Rebuilding the grid must not buy eligibility by narrowing the design.

    A cheap way to populate every stratum is to drop the factors that make
    cells hard to populate. That would trade a coverage gap for a blind spot,
    so the factor space and the pairwise claim both have to survive intact.
    """
    from scripts.generate_cf_v11_design import build_spec

    protocol = CFValidationProtocol.load(V11_SPEC)
    v10 = CFValidationProtocol.load(V10_SPEC)
    regenerated = build_spec()

    assert regenerated["retained_cells"] == [dict(cell) for cell in protocol.retained_cells]
    assert dict(protocol.factors) == dict(v10.factors)
    assert dict(protocol.metrics) == dict(v10.metrics)
    provenance = regenerated["design_selection"]
    assert provenance["pairwise_pairs_covered"] == provenance["pairwise_pairs_total"]
    assert len(protocol.retained_cells) == 48
    assert protocol.protocol_id.endswith("v11")
    assert protocol.checksum != v10.checksum


def test_v11_brackets_the_density_bound_it_is_meant_to_estimate() -> None:
    """v10 could not inform its own arm-density bound.

    Every eligible cell sat at or above 10.0 and five sat exactly on it, so the
    lane the bound was fitted on held nothing on the other side of it. Cells
    that clear the absolute count floor but fail on density alone are what
    separate the two terms.
    """
    from scripts.generate_cf_v10_design import expected_smallest_arm

    protocol = CFValidationProtocol.load(V11_SPEC)
    bound = float(protocol.reference_profile["minimum_arm_units_per_covariate"])
    floor = float(protocol.reference_profile["minimum_group_count"])

    isolating = [
        cell
        for cell in protocol.retained_cells
        if cell["support"] == "strong"
        and int(cell["n_groups"]) <= int(protocol.reference_profile["maximum_group_count"])
        and expected_smallest_arm(dict(cell)) >= floor
        and expected_smallest_arm(dict(cell)) / int(cell["n_covariates"]) < bound
    ]
    assert isolating, "no cell fails on density alone; the bound stays unidentified"
    for groups in (2, 3):
        below = [cell for cell in isolating if int(cell["n_groups"]) == groups]
        assert below, f"k={groups} has no sub-boundary density cell"


def _v10_with_boundary_block() -> dict[str, object]:
    v10 = json.loads(V10_SPEC.read_text(encoding="utf-8"))
    v11 = json.loads(V11_SPEC.read_text(encoding="utf-8"))
    return {**v10, "boundary_estimation": v11["boundary_estimation"]}


def test_boundary_procedure_is_identifiable_on_v11_and_not_on_v10() -> None:
    """Gate 4's check: parameter count against effective observations.

    One slope shared across strata plus one intercept per stratum is five
    parameters. v10 offers 19 admissible cells and an entirely empty (k=3,
    linear) stratum; v11 offers 28 and populates all four. Running this before
    dispatching is the point -- a design that cannot identify the procedure it
    declares still calibrates, still passes every gate, and still emits a
    boundary that is an artifact of the parameterization.
    """
    from scripts.estimate_support_boundary import identifiability_report

    v11 = identifiability_report(CFValidationProtocol.load(V11_SPEC))
    v10 = identifiability_report(CFValidationProtocol.from_dict(_v10_with_boundary_block()))

    assert v11["identifiable"], v11["failures"]
    assert v11["parameters"] == 5
    assert v11["effective_observations"] == 28
    assert v11["observations_per_parameter"] >= 5
    for stratum in v11["strata"].values():
        assert stratum["distinct_densities"] >= 3
        assert stratum["below_declared_bound"] >= 1
        assert stratum["at_or_above_declared_bound"] >= 1

    assert not v10["identifiable"]
    assert any("n_groups=3, learner=linear" in failure for failure in v10["failures"])


def test_boundary_support_set_excludes_cells_below_the_arm_count_floor() -> None:
    """Below the count floor the count term binds, not density.

    Including such a cell would charge a failure caused by having too few units
    outright against the density boundary, biasing it upward -- toward claiming
    the method needs more data per covariate than it does.
    """
    from scripts.calibrate_cf_support import _profile_scope, expected_smallest_arm
    from scripts.estimate_support_boundary import boundary_support_set

    protocol = CFValidationProtocol.load(V11_SPEC)
    floor, maximum, bound = _profile_scope(protocol)
    rows = boundary_support_set(protocol)

    for row in rows:
        cells = protocol.retained_cells if row["kind"] == "simulated" else protocol.plasmode_cells
        cell = dict(cells[row["cell_index"]])
        assert expected_smallest_arm(cell) >= floor
        assert int(cell["n_groups"]) <= int(maximum)
    # And it must keep the sub-boundary cells, or there is nothing to locate.
    assert [row for row in rows if row["density"] < bound]


def test_boundary_procedure_recovers_a_boundary_it_was_given() -> None:
    """A dry run on outcomes with a known answer, before any real evidence.

    Cells pass exactly when density clears 8.0, which is below the declared
    10.0. Each stratum's estimate must land in that stratum's gap between the
    highest failing density and the lowest passing one -- the property any
    monotone fit has to satisfy, asserted instead of a fitted constant so the
    test does not pin numerical noise.
    """
    from scripts.estimate_support_boundary import boundary_support_set, estimate_boundary

    protocol = CFValidationProtocol.load(V11_SPEC)
    rows = boundary_support_set(protocol)
    outcomes = {(row["kind"], row["cell_index"]): row["density"] >= 8.0 for row in rows}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimate_boundary(protocol, outcomes)

    assert result["status"] == "complete"
    assert result["adoption"] == "report-only"
    assert result["log10_density_slope"] > 0
    for name, values in result["strata"].items():
        groups, learner = name.removeprefix("k=").split(",")
        stratum = (int(groups), learner)
        densities = {row["density"] for row in rows if row["stratum"] == stratum}
        highest_failing = max(value for value in densities if value < 8.0)
        lowest_passing = min(value for value in densities if value >= 8.0)
        assert highest_failing < values["estimated_boundary"] < lowest_passing
    # The thinnest stratum must report the widest interval; an estimate that
    # hides how little it rests on is worse than no estimate.
    widths = {
        name: values["interval"][1] - values["interval"][0]
        for name, values in result["strata"].items()
    }
    assert max(widths, key=widths.__getitem__) == "k=3,linear"


def test_boundary_procedure_refuses_a_non_positive_density_effect() -> None:
    """A boundary is only meaningful if support improves with density.

    Inverting the outcomes gives a negative slope. Solving for the crossing
    anyway returns a finite number with the wrong sense entirely, so the
    procedure refuses instead.
    """
    from scripts.estimate_support_boundary import boundary_support_set, estimate_boundary

    protocol = CFValidationProtocol.load(V11_SPEC)
    rows = boundary_support_set(protocol)
    outcomes = {(row["kind"], row["cell_index"]): row["density"] < 8.0 for row in rows}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimate_boundary(protocol, outcomes)

    assert result["status"] == "refused/non-positive-density-effect"
    assert "estimated_boundary" not in json.dumps(result)


def test_boundary_procedure_refuses_an_unidentifiable_design_before_fitting() -> None:
    from scripts.estimate_support_boundary import estimate_boundary

    protocol = CFValidationProtocol.from_dict(_v10_with_boundary_block())
    result = estimate_boundary(protocol, {})

    assert result["status"] == "refused/unidentifiable"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"adoption": "adopt"}, "adoption"),
        ({"model": "linear-probability"}, "model"),
        ({"unit_of_observation": "replication"}, "unit_of_observation"),
        ({"pass_probability_target": 0.8}, "pass_probability_target"),
        ({"minimum_distinct_densities_per_stratum": 2}, "at least 3"),
        ({"bootstrap_resamples": 100}, "at least 1000"),
        ({"minimum_observations_per_parameter": 0}, "must be positive"),
        ({"require_bracketing_per_stratum": "yes"}, "must be boolean"),
    ],
)
def test_boundary_declaration_rejects_valid_json_with_wrong_content(
    mutation: dict[str, object], message: str
) -> None:
    """The recurring failure is a block that parses and means the wrong thing.

    `threshold_quantiles` in the wrong shape and a `software` map naming three
    of seven packages both survived a freeze and a full calibration before
    dying downstream. A pre-specified procedure that can drift into an
    unrecognized string is worth less than no procedure at all.
    """
    values = json.loads(V11_SPEC.read_text(encoding="utf-8"))
    values["boundary_estimation"] = {**values["boundary_estimation"], **mutation}

    with pytest.raises(ValueError, match=message):
        CFValidationProtocol.from_dict(values)


def test_boundary_declaration_is_absent_from_earlier_protocols() -> None:
    """Optional, so v3-v10 checksums cannot move."""
    for spec_path in (SPEC, V9_SPEC, V10_SPEC):
        assert CFValidationProtocol.load(spec_path).boundary_estimation is None
    v10 = CFValidationProtocol.load(V10_SPEC)
    assert v10.checksum == "a1c54a76e1fb8401f8d1b7eea50c16ccd77cd9e0e1f0afdb08fe28594c6caccb"


@pytest.mark.parametrize(
    ("block", "dropped"),
    [("boundary_estimation", "bootstrap_seed"), ("external_agreement", "family_wise_error")],
)
def test_preregistered_blocks_reject_a_missing_key(block: str, dropped: str) -> None:
    """A silently absent key is a procedure that was never fully declared."""
    values = json.loads(V11_SPEC.read_text(encoding="utf-8"))
    values[block] = {k: v for k, v in values[block].items() if k != dropped}

    with pytest.raises(ValueError, match="must declare exactly"):
        CFValidationProtocol.from_dict(values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"comparator_folds": "scova"}, "comparator_folds"),
        ({"statistic": "absolute-difference-in-scova-se"}, "statistic"),
        ({"unit_of_observation": "arm"}, "unit_of_observation"),
        ({"strata": "cell"}, "strata"),
        ({"comparator_fold_seed_offset": 0}, "non-zero"),
        ({"family_wise_error": 0.0}, "family_wise_error"),
        ({"minimum_informative_cell_fraction": 0.0}, "minimum_informative_cell_fraction"),
        ({"degenerate_difference_in_scova_se": 0.0}, "degenerate_difference_in_scova_se"),
    ],
)
def test_external_agreement_declaration_rejects_wrong_content(
    mutation: dict[str, object], message: str
) -> None:
    """Every degree of freedom in the lane policy has to be a declared one.

    `comparator_fold_seed_offset = 0` is the one worth naming: it parses, it
    validates as an integer, and it silently hands the comparators SCOVA's own
    folds again -- restoring the exact degeneracy the policy exists to prevent.
    """
    values = json.loads(V11_SPEC.read_text(encoding="utf-8"))
    values["external_agreement"] = {**values["external_agreement"], **mutation}

    with pytest.raises(ValueError, match=message):
        CFValidationProtocol.from_dict(values)
