import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.cf_external_validation import fixed_nuisance_score
from benchmarks.cf_reference_campaign import run_campaign
from scova._aipw import assemble_aipw
from scova.cf import (
    CFSupportProfile,
    CFValidationProtocol,
    SeedPartition,
)
from scova.simulate import generate_data

SPEC = Path("benchmarks/specs/cf_reference_v1.json")


def test_frozen_reference_protocol_has_disjoint_evidence_lanes() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    assert protocol.calibration.count == 1000
    assert protocol.validation.count == 2000
    assert len(protocol.retained_cells) == 12
    assert protocol.checksum == CFValidationProtocol.from_dict(
        protocol.to_dict()
    ).checksum


def test_protocol_rejects_overlapping_or_undersized_lanes() -> None:
    values = json.loads(SPEC.read_text(encoding="utf-8"))
    values["seed_partitions"]["validation"] = {"start": 1000500, "count": 2000}
    with pytest.raises(ValueError, match="disjoint"):
        CFValidationProtocol.from_dict(values)
    values = json.loads(SPEC.read_text(encoding="utf-8"))
    values["seed_partitions"]["calibration"] = {"start": 1000000, "count": 999}
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
    with pytest.raises(ValueError):
        SeedPartition(-1, 1)
    with pytest.raises(ValueError):
        SeedPartition(0, 0)
