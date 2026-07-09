"""Aggregate campaign shards and evaluate preregistered pass criteria."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import beta


def _upper_binomial(successes: int, total: int) -> float:
    if successes == total:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, total - successes))


def _primary_cell(records: list[dict], criteria: dict) -> dict:
    accepted = [record for record in records if not record["alternative"]["refused"]]
    null_accepted = [record for record in records if not record["null"]["refused"]]
    coverage = (
        float(np.mean([record["alternative"]["uniform_coverage"] for record in accepted]))
        if accepted
        else 0.0
    )
    false_signs = sum(record["null"]["false_sign_certificate"] for record in null_accepted)
    fwer_upper = _upper_binomial(false_signs, len(null_accepted)) if null_accepted else 1.0
    errors = np.asarray(
        [record["alternative"]["scientific_target_mean_error"] for record in accepted]
    )
    error_sd = float(errors.std(ddof=1)) if len(errors) > 1 else 0.0
    standardized_bias = (
        abs(float(errors.mean())) / error_sd if np.isfinite(error_sd) and error_sd > 0 else None
    )
    stability_coverage = (
        float(np.mean([record["alternative"]["stability_covered"] for record in accepted]))
        if accepted
        else 0.0
    )
    execution_failures = sum(
        "execution_error" in record["alternative"] or "execution_error" in record["null"]
        for record in records
    )
    passed = bool(
        accepted
        and execution_failures == 0
        and criteria["simultaneous_coverage_min"]
        <= coverage
        <= criteria["simultaneous_coverage_max"]
        and fwer_upper <= criteria["fwer_upper_bound_max"]
        and standardized_bias is not None
        and standardized_bias <= criteria["standardized_absolute_bias_max"]
        and stability_coverage >= criteria["simultaneous_coverage_min"]
    )
    return {
        "total": len(records),
        "accepted": len(accepted),
        "refusal_rate": 1 - len(accepted) / len(records),
        "execution_failures": execution_failures,
        "simultaneous_coverage": coverage,
        "false_sign_count": false_signs,
        "fwer_upper_95": fwer_upper,
        "standardized_absolute_bias": standardized_bias,
        "standardized_bias_status": (
            "computed" if standardized_bias is not None else "undefined-zero-error-variance"
        ),
        "stability_coverage": stability_coverage,
        "passed": passed,
    }


def _robustness_cell(records: list[dict], criteria: dict, nuisance: str) -> dict:
    if nuisance in ("oracle", "gps_correct_outcome_wrong", "flexible"):
        result = _primary_cell(records, criteria)
        result["expected_behavior"] = "scientific-target inference"
        return result
    execution_failures = sum("execution_error" in record["alternative"] for record in records)
    if nuisance == "deliberately_inadequate":
        protected = sum(
            record["alternative"]["refused"] or record["alternative"]["gate_status"] == "warning"
            for record in records
        )
        protection_rate = protected / len(records)
        return {
            "total": len(records),
            "execution_failures": execution_failures,
            "expected_behavior": "warning-or-refusal",
            "warning_or_refusal_rate": protection_rate,
            "passed": execution_failures == 0 and protection_rate >= 0.80,
        }
    drifts = [record["alternative"]["scientific_pseudo_target_max_drift"] for record in records]
    finite = all(value is not None and np.isfinite(value) for value in drifts)
    return {
        "total": len(records),
        "execution_failures": execution_failures,
        "expected_behavior": "scientific-versus-pseudo-target disclosure",
        "mean_scientific_pseudo_target_drift": float(np.mean(drifts)) if finite else None,
        "passed": (
            execution_failures == 0
            and finite
            and max((float(value) for value in drifts if value is not None), default=0.0) > 1e-6
        ),
    }


def summarize(paths: list[Path], specification: dict) -> dict:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    spec_hashes = set()
    tiers = set()
    threshold_hashes = set()
    validation_levels = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_hashes.add(payload["specification_sha256"])
        tiers.add(payload["tier"])
        threshold_hashes.add(payload.get("threshold_artifact_sha256"))
        validation_levels.add(payload.get("validation_level"))
        for record in payload["records"]:
            key = json.dumps(record["spec"], sort_keys=True)
            by_cell[key].append(record)
    if len(spec_hashes) != 1 or len(tiers) != 1 or len(validation_levels) != 1:
        raise ValueError("campaign shards do not share one specification, tier, and level")
    if len(threshold_hashes) != 1 or None in threshold_hashes:
        raise ValueError("validation shards do not share one locked threshold artifact")
    tier = next(iter(tiers))
    criteria = specification["directional_pass_criteria"]
    cells = []
    all_passed = True
    for key, records in sorted(by_cell.items()):
        spec = json.loads(key)
        if tier in ("directional_robustness", "local_robustness_pilot"):
            cell = _robustness_cell(records, criteria, spec["nuisance"])
        else:
            cell = _primary_cell(records, criteria)
        cell["spec"] = spec
        all_passed &= bool(cell["passed"])
        cells.append(cell)
    execution_failures = sum(cell["execution_failures"] for cell in cells)
    if tier in ("directional_robustness", "local_robustness_pilot"):
        inferential_nuisances = {"oracle", "gps_correct_outcome_wrong", "flexible"}
        inferential_records = [
            record
            for key, records in by_cell.items()
            if json.loads(key)["nuisance"] in inferential_nuisances
            for record in records
        ]
        inferential = _primary_cell(inferential_records, criteria)
        behavioral_cells = [
            cell for cell in cells if cell["spec"]["nuisance"] not in inferential_nuisances
        ]
        behavioral_pass_rate = (
            sum(cell["passed"] for cell in behavioral_cells) / len(behavioral_cells)
            if behavioral_cells
            else 1.0
        )
        tier_passed = bool(
            execution_failures == 0 and inferential["passed"] and behavioral_pass_rate >= 0.75
        )
        aggregate = {
            "inferential": inferential,
            "behavioral_cell_pass_rate": behavioral_pass_rate,
            "minimum_behavioral_cell_pass_rate": 0.75,
        }
    else:
        pooled = _primary_cell(
            [record for records in by_cell.values() for record in records], criteria
        )
        severe_failures = sum(
            cell["execution_failures"] > 0
            or cell["simultaneous_coverage"] < 0.80
            or cell["fwer_upper_95"] > 0.25
            or (
                cell["standardized_absolute_bias"] is not None
                and cell["standardized_absolute_bias"] > 0.50
            )
            or cell["stability_coverage"] < 0.80
            for cell in cells
        )
        cell_pass_rate = sum(cell["passed"] for cell in cells) / len(cells)
        tier_passed = bool(
            execution_failures == 0
            and pooled["passed"]
            and severe_failures == 0
            and cell_pass_rate >= 0.75
        )
        aggregate = {
            "pooled": pooled,
            "individual_cell_pass_rate": cell_pass_rate,
            "minimum_individual_cell_pass_rate": 0.75,
            "severe_cell_failures": severe_failures,
        }
    result = {
        "schema_version": 2,
        "protocol": specification["protocol"],
        "validation_level": next(iter(validation_levels)),
        "tier": tier,
        "specification_sha256": next(iter(spec_hashes)),
        "threshold_artifact_sha256": next(iter(threshold_hashes)),
        "all_cells_passed": all_passed,
        "tier_passed": tier_passed,
        "decision_rule": "pooled-directional-v1",
        "aggregate": aggregate,
        "cells": cells,
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    result["summary_sha256"] = sha256(encoded).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/specs/stage3_release.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    result = summarize(args.inputs, specification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
