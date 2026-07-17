"""Create a checksum-bound candidate profile from the frozen calibration lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum

LOWER_FEATURE = "minimum_ess_ratio"
UPPER_FEATURES = (
    "maximum_normalized_weight",
    "maximum_top_one_percent_weight_share",
    "maximum_absolute_weighted_balance_difference",
    "maximum_influence_top_one_percent_share",
    "maximum_seed_standardized_departure",
)


def _verify_evidence(evidence: dict[str, Any]) -> None:
    supplied = evidence.get("evidence_checksum")
    payload = {name: value for name, value in evidence.items() if name != "evidence_checksum"}
    if supplied != canonical_checksum(payload):
        raise ValueError("Campaign evidence checksum does not match its payload")


def _passes(record: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return bool(
        record[LOWER_FEATURE] >= thresholds[LOWER_FEATURE]
        and all(record[name] <= thresholds[name] for name in UPPER_FEATURES)
    )


def _cell_gate(
    records: list[dict[str, Any]], metrics: dict[str, float]
) -> tuple[bool, dict[str, float | bool | None]]:
    n = len(records)
    if n < 2:
        return False, {"passed": False, "reason": "fewer-than-two-supported-replicates"}
    coverage = float(np.mean([record["covered"] for record in records]))
    coverage_mcse = np.sqrt(0.95 * 0.05 / n)
    errors = np.array([record["estimate"] - record["truth"] for record in records])
    empirical_sd = float(np.std(errors, ddof=1))
    bias = float(np.mean(errors))
    mean_se = float(np.mean([record["standard_error"] for record in records]))
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else np.inf
    null_records = [record for record in records if record["null"]]
    type_i_error = (
        None
        if not null_records
        else float(np.mean([record["rejected"] for record in null_records]))
    )
    type_i_mcse = (
        None
        if type_i_error is None
        else np.sqrt(metrics["type_i_error"] * (1 - metrics["type_i_error"]) / len(null_records))
    )
    multiplier = metrics["monte_carlo_standard_error_multiplier"]
    passed = bool(
        abs(coverage - metrics["confidence_level"]) <= multiplier * coverage_mcse
        and abs(bias) <= metrics["maximum_standardized_bias"] * empirical_sd
        and metrics["minimum_se_ratio"] <= se_ratio <= metrics["maximum_se_ratio"]
        and (
            type_i_error is None
            or abs(type_i_error - metrics["type_i_error"]) <= multiplier * type_i_mcse
        )
    )
    return passed, {
        "passed": passed,
        "replications": n,
        "coverage": coverage,
        "bias": bias,
        "empirical_standard_deviation": empirical_sd,
        "standard_error_ratio": se_ratio,
        "type_i_error": type_i_error,
    }


def calibrate(
    protocol: CFValidationProtocol, evidence: dict[str, Any]
) -> dict[str, Any]:
    _verify_evidence(evidence)
    if evidence["lane"] != "calibration" or not evidence["complete_frozen_lane"]:
        raise ValueError("Only a complete frozen calibration lane can create a profile")
    if evidence["protocol_checksum"] != protocol.checksum:
        raise ValueError("Calibration evidence uses a different protocol")
    records = [record for record in evidence["records"] if not record["refused"]]
    thresholds = {
        LOWER_FEATURE: float(np.quantile([r[LOWER_FEATURE] for r in records], 0.05)),
        **{
            name: float(np.quantile([r[name] for r in records], 0.95))
            for name in UPPER_FEATURES
        },
    }
    audits: list[dict[str, Any]] = []
    passed = True
    for summary in evidence["summaries"]:
        cell_index = summary["cell_index"]
        cell = summary["cell"]
        cell_records = [
            record
            for record in records
            if record["cell_index"] == cell_index and _passes(record, thresholds)
        ]
        if cell["support"] == "structural-failure":
            cell_passed = summary["refusal_rate"] == 1.0
            audit = {"passed": cell_passed, "structural_refusal_rate": summary["refusal_rate"]}
        else:
            cell_passed, audit = _cell_gate(cell_records, dict(protocol.metrics))
        passed &= cell_passed
        audits.append({"cell_index": cell_index, "cell": cell, **audit})
    result: dict[str, Any] = {
        "artifact_type": "scova-cf-support-calibration",
        "schema_version": 1,
        "protocol_checksum": protocol.checksum,
        "calibration_evidence_checksum": evidence["evidence_checksum"],
        "threshold_selection": "outcome-blind-5th/95th-percentile-rule-v1",
        "thresholds": thresholds,
        "all_calibration_gates_passed": passed,
        "audit": audits,
    }
    result["calibration_artifact_checksum"] = canonical_checksum(result)
    if passed:
        profile = CFSupportProfile(
            profile_id=f"{protocol.protocol_id}-candidate",
            protocol_checksum=protocol.checksum,
            calibration_evidence_checksum=evidence["evidence_checksum"],
            validation_evidence_checksum=None,
            thresholds=thresholds,
        )
        result["candidate_profile"] = profile.to_dict()
    else:
        result["candidate_profile"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--calibration-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(
        CFValidationProtocol.load(args.spec),
        json.loads(args.calibration_evidence.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
