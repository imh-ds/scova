import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.cf_external_validation import fixed_nuisance_score
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
)
from scova.simulate import generate_data

SPEC = Path("benchmarks/specs/cf_reference_v1.json")


def test_frozen_reference_protocol_has_disjoint_evidence_lanes() -> None:
    protocol = CFValidationProtocol.load(SPEC)
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
    assert protocol.checksum == CFValidationProtocol.from_dict(
        protocol.to_dict()
    ).checksum


def test_protocol_rejects_overlapping_or_undersized_lanes() -> None:
    values = json.loads(SPEC.read_text(encoding="utf-8"))
    values["seed_partitions"]["validation"] = {"start": 1000500, "count": 2000}
    with pytest.raises(ValueError, match="disjoint"):
        CFValidationProtocol.from_dict(values)


def test_v2_protocol_rejects_incomplete_frozen_contract() -> None:
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
