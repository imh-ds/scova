"""Prospective adaptive-only SCOVA-CF qualification program.

The program is executable but intentionally undispatched.  It freezes a new
protocol, supplies development evidence to the existing calibration workflow,
and never promotes a profile by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.cf_reference_campaign import _git_commit, dependency_lock_checksum, run_campaign
from scova.cf import (
    ARTIFACT_SCHEMA_VERSION,
    QUALIFICATION_PROTOCOL_ID,
    CFValidationProtocol,
    SeedPartition,
    StudyProgram,
    canonical_checksum,
    qualification_cells,
    qualification_design,
)


_METRICS = {
    "confidence_level": 0.95,
    "type_i_error": 0.05,
    "monte_carlo_standard_error_multiplier": 2.0,
    "maximum_standardized_bias": 0.15,
    "minimum_se_ratio": 0.8,
    "maximum_se_ratio": 1.25,
    "strong_support_minimum_expected_arm_count": 30.0,
    "minimum_strong_replication_pass_fraction": 0.85,
    "minimum_strong_cell_pass_fraction": 0.85,
    "coverage_family_wise_error": 0.05,
}


def qualification_protocol() -> CFValidationProtocol:
    """Return the frozen engine protocol represented by the qualification design."""
    design = qualification_design()
    cells = tuple(
        {name: value for name, value in cell.items() if name != "cell_id"}
        for cell in qualification_cells()
    )
    return CFValidationProtocol(
        schema_version=1,
        frozen=True,
        protocol_id=QUALIFICATION_PROTOCOL_ID,
        reference_profile={
            "mode": "observational-causal",
            "assignment": "estimated",
            "nuisance_strategy": "adaptive",
            "maximum_group_count": 3,
            "minimum_group_count": 50,
            "maximum_covariate_count": 5,
            "estimand_id": "study-population-standardized-means",
            "estimator": "aipw-unnormalized",
            "outcome_type": "continuous",
            "independent_unit": "row",
        },
        factors={name: tuple(levels) for name, levels in design["factors"].items()},
        retained_cells=cells,
        pilot=SeedPartition(start=610_000_000, count=20),
        calibration=SeedPartition(start=611_000_000, count=2000),
        validation=SeedPartition(start=615_000_000, count=2000),
        learners=("adaptive",),
        metrics=_METRICS,
        software={
            "python": "3.12.13", "numpy": "2.2.6", "pandas": "2.2.3",
            "scipy": "1.15.3", "scikit-learn": "1.6.1",
        },
        dependency_lock_checksum=dependency_lock_checksum(),
        design_selection={
            "method": "deterministic-mandatory-stress-then-pairwise-greedy",
            "design_checksum": design["design_checksum"],
            "cell_count": 48,
        },
        calibration_fit_fraction=0.60,
        threshold_quantiles={
            "minimum_ess_ratio": (0, 0.01, 0.025, 0.05, 0.1, 0.2),
            "upper_metrics": (0.8, 0.9, 0.95, 0.975, 0.99, 1),
        },
        calibration_screening=_METRICS,
        calibration_enrichment_screening=True,
        calibration_candidate_retention_fraction=0.85,
    )


def qualification_spec() -> dict[str, Any]:
    protocol = qualification_protocol()
    return {
        **protocol.to_dict(),
        "program_type": StudyProgram.QUALIFICATION.value,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "design_checksum": qualification_design()["design_checksum"],
        "candidate_profile_state": "unpromoted",
        "promotion_rule": "independent-held-out-validation-and-human-approval",
        "boundary_estimation": None,
        "source_evidence_ids": [],
    }


def qualification_evidence(
    *, lane: str, replications: int | None = None, decision_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a provenance-complete evidence artifact for a frozen lane."""
    protocol = qualification_protocol()
    evidence = run_campaign(protocol, lane=lane, replications=replications)
    evidence.pop("evidence_checksum")
    evidence.update(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "program_type": StudyProgram.QUALIFICATION.value,
            "design_checksum": qualification_design()["design_checksum"],
            "dependency_lock_checksum": dependency_lock_checksum(),
            "git_commit": _git_commit(),
            "planned_replications_per_cell": getattr(protocol, lane).count,
            "planned_replications": len(protocol.retained_cells) * getattr(protocol, lane).count,
            "completed_replications": len(evidence["records"]),
            "source_evidence_ids": [],
            "decision_manifest_checksum": (
                None if decision_manifest is None else decision_manifest.get("manifest_checksum")
            ),
            "required_decision_ids": (
                [] if decision_manifest is None
                else [entry["decision_id"] for entry in decision_manifest["required_decisions"]]
            ),
            "promotion_decision": "unpromoted/requires-independent-validation-and-human-approval",
        }
    )
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-spec", type=Path)
    parser.add_argument("--lane", choices=("pilot", "calibration", "validation"))
    parser.add_argument("--replications", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decision-manifest", type=Path)
    args = parser.parse_args()
    if args.write_spec:
        args.write_spec.parent.mkdir(parents=True, exist_ok=True)
        args.write_spec.write_text(json.dumps(qualification_spec(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.lane:
        if args.output is None:
            parser.error("--output is required with --lane")
        if args.decision_manifest is None:
            parser.error("--decision-manifest is required with --lane")
        artifact = qualification_evidence(
            lane=args.lane,
            replications=args.replications,
            decision_manifest=json.loads(args.decision_manifest.read_text(encoding="utf-8")),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    elif args.output is not None:
        parser.error("--output requires --lane")


if __name__ == "__main__":
    main()
