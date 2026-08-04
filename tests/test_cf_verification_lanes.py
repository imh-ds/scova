from __future__ import annotations

from benchmarks.cf_qualification_program import VERIFICATION_LANES, qualification_protocol, qualification_spec
from scripts.aggregate_qualification_verification import aggregate
from scripts.estimate_qualification_boundary import boundary_diagnostic


def test_prospective_protocol_declares_frozen_external_and_inference_lanes() -> None:
    protocol = qualification_protocol()
    assert protocol.verification_lanes == VERIFICATION_LANES
    assert len(protocol.external_cells) == 8
    assert len(protocol.inference_cells) == 6
    assert protocol.external is not None and protocol.external.count == 50
    assert protocol.inference is not None and protocol.inference.count == 2000
    assert all(cell["learner"] == "adaptive" and cell["n_covariates"] <= 5 for cell in protocol.external_cells)
    assert not protocol.verification_lanes["boundary"]["promotion_required"]
    assert all(protocol.verification_lanes[name]["promotion_required"] for name in ("external", "inference", "validation"))


def test_qualification_spec_declares_lane_claim_limits_without_reusing_boundary_scope() -> None:
    spec = qualification_spec()
    assert spec["boundary_estimation"] is None
    assert spec["boundary_diagnostic"]["scope_effect"] == "none"
    assert spec["boundary_diagnostic"]["maximum_95_interval_width"] < 0.31
    assert spec["verification_lanes"]["external"]["prohibited_claim"]


def test_boundary_diagnostic_refuses_without_candidate_and_cannot_promote() -> None:
    protocol = qualification_protocol()
    artifact = boundary_diagnostic(protocol, None, {"evidence_checksum": "no-candidate"})
    assert artifact["status"] == "unavailable/no-candidate-profile"
    assert not artifact["informative"]
    assert not artifact["promotion_required"]
    assert artifact["scope_effect"] == "none"


def test_aggregate_fails_closed_for_missing_or_uninformative_verification_evidence() -> None:
    result = aggregate(external=None, inference=None, validation=None, boundary=None)
    assert result["promotion_decision"] == "blocked"
    assert "missing external evidence" in result["reasons"]
    boundary = {"promotion_required": True, "status": "complete/informative"}
    result = aggregate(external=None, inference=None, validation=None, boundary=boundary)
    assert "boundary diagnostic must not become a promotion prerequisite" in result["reasons"]
