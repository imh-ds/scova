"""Select and lock monotone thresholds using calibration seeds only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import beta

PROFILE_METRICS = {
    "min_group_ess": ("min_group_ess", "minimum"),
    "min_target_ess_ratio": ("target_ess_ratio", "minimum"),
    "max_influence_share": ("max_influence_share", "maximum"),
    "max_weight_concentration": ("max_weight_concentration", "maximum"),
    "min_propensity_q01": ("min_propensity_q01", "minimum"),
    "max_calibration_error": ("max_calibration_error", "maximum"),
    "max_balance": ("max_balance", "maximum"),
    "max_crossfit_instability": ("crossfit_instability", "maximum"),
}

UNBOUNDED_WARNING_FLOOR = {
    "name": "numerical-validity-floor",
    "min_group_ess": 0.0,
    "min_target_ess_ratio": 0.0,
    "max_influence_share": 1.0,
    "max_weight_concentration": 1.0,
    "min_propensity_q01": 1e-15,
    "max_calibration_error": 1.0,
    "max_balance": 1e9,
    "max_crossfit_instability": 1e9,
}

DIRECTIONAL_CALIBRATION_DEFAULTS = {
    "minimum_supported_cell_pass_rate": 0.75,
    "severe_cell_coverage_min": 0.80,
    "severe_cell_fwer_upper_max": 0.25,
    "severe_cell_standardized_bias_max": 0.50,
    "severe_cell_stability_coverage_min": 0.80,
}


def _accepted(metrics: dict[str, float], profile: dict) -> bool:
    for profile_name, (metric_name, direction) in PROFILE_METRICS.items():
        value = metrics[metric_name]
        threshold = profile[profile_name]
        if direction == "minimum" and value < threshold:
            return False
        if direction == "maximum" and value > threshold:
            return False
    return True


def _lower(successes: int, total: int) -> float:
    return 0.0 if successes == 0 else float(beta.ppf(0.05, successes, total - successes + 1))


def _upper(successes: int, total: int) -> float:
    return 1.0 if successes == total else float(beta.ppf(0.95, successes + 1, total - successes))


def _cell_audit(records: list[dict], profile: dict, criteria: dict) -> dict:
    accepted = [
        record
        for record in records
        if not record["alternative"]["refused"]
        and _accepted(record["alternative"]["gate_metrics"], profile)
    ]
    null_accepted = [
        record
        for record in records
        if not record["null"]["refused"]
        and _accepted(record["null"]["gate_metrics"], profile)
    ]
    total = len(records)
    supported = (
        len(accepted) >= criteria["minimum_accepted_repetitions"]
        and len(accepted) / total >= criteria["minimum_acceptance_rate"]
    )
    coverage = (
        float(np.mean([row["alternative"]["uniform_coverage"] for row in accepted]))
        if accepted
        else 0.0
    )
    naive_coverage = (
        float(np.mean([row["alternative"]["naive_uniform_coverage"] for row in accepted]))
        if accepted
        else 0.0
    )
    false_signs = sum(row["null"]["false_sign_certificate"] for row in null_accepted)
    fwer_upper = _upper(false_signs, len(null_accepted)) if null_accepted else 1.0
    errors = np.asarray(
        [row["alternative"]["scientific_target_mean_error"] for row in accepted]
    )
    standardized_bias = (
        abs(float(errors.mean())) / float(errors.std(ddof=1))
        if len(errors) > 1 and errors.std(ddof=1) > 0
        else float("inf")
    )
    stability_coverage = (
        float(np.mean([row["alternative"]["stability_covered"] for row in accepted]))
        if accepted
        else 0.0
    )
    passed = bool(
        supported
        and criteria["simultaneous_coverage_min"]
        <= coverage
        <= criteria["simultaneous_coverage_max"]
        and standardized_bias <= criteria["standardized_absolute_bias_max"]
        and stability_coverage >= criteria["simultaneous_coverage_min"]
    )
    severe_failure = bool(
        supported
        and (
            coverage < criteria["severe_cell_coverage_min"]
            or fwer_upper > criteria["severe_cell_fwer_upper_max"]
            or standardized_bias > criteria["severe_cell_standardized_bias_max"]
            or stability_coverage < criteria["severe_cell_stability_coverage_min"]
        )
    )
    return {
        "total": total,
        "accepted": len(accepted),
        "null_accepted": len(null_accepted),
        "acceptance_rate": len(accepted) / total,
        "supported": supported,
        "simultaneous_coverage": coverage,
        "coverage_lower_95": (
            _lower(round(coverage * len(accepted)), len(accepted)) if accepted else 0.0
        ),
        "naive_simultaneous_coverage": naive_coverage,
        "corrected_minus_naive_coverage": coverage - naive_coverage,
        "false_sign_count": false_signs,
        "fwer_upper_95": fwer_upper,
        "standardized_absolute_bias": standardized_bias,
        "stability_coverage": stability_coverage,
        "passed": passed,
        "severe_failure": severe_failure,
    }


def calibrate(campaigns: list[dict], candidates: dict, release_spec: dict) -> dict:
    seed_namespace = release_spec["calibration_seed_namespace"]
    if not release_spec.get("frozen") or not candidates.get("frozen"):
        raise ValueError("release and threshold specifications must be frozen")
    if any(campaign.get("seed_namespace") != seed_namespace for campaign in campaigns):
        raise ValueError("threshold calibration received non-calibration seeds")
    if any(campaign.get("tier") != "calibration" for campaign in campaigns):
        raise ValueError("threshold calibration requires calibration-tier artifacts")
    if any(campaign.get("validation_level") != "directional" for campaign in campaigns):
        raise ValueError("calibration artifacts have the wrong validation level")
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for campaign in campaigns:
        for record in campaign["records"]:
            by_cell[json.dumps(record["spec"], sort_keys=True)].append(record)
    criteria = {
        **DIRECTIONAL_CALIBRATION_DEFAULTS,
        **release_spec["directional_pass_criteria"],
    }
    selected = None
    profile_audits = []
    for profile in candidates["profiles"]:
        cells = []
        for key, records in sorted(by_cell.items()):
            cell = _cell_audit(records, profile, criteria)
            cell["spec"] = json.loads(key)
            cells.append(cell)
        supported = [cell for cell in cells if cell["supported"]]
        supported_keys = {
            json.dumps(cell["spec"], sort_keys=True) for cell in supported
        }
        pooled_records = [
            record
            for key in supported_keys
            for record in by_cell[key]
            if not record["alternative"]["refused"]
            and _accepted(record["alternative"]["gate_metrics"], profile)
        ]
        pooled_null_records = [
            record
            for key in supported_keys
            for record in by_cell[key]
            if not record["null"]["refused"]
            and _accepted(record["null"]["gate_metrics"], profile)
        ]
        pooled_coverage = (
            float(
                np.mean(
                    [record["alternative"]["uniform_coverage"] for record in pooled_records]
                )
            )
            if pooled_records
            else 0.0
        )
        pooled_naive_coverage = (
            float(
                np.mean(
                    [
                        record["alternative"]["naive_uniform_coverage"]
                        for record in pooled_records
                    ]
                )
            )
            if pooled_records
            else 0.0
        )
        pooled_stability_coverage = (
            float(
                np.mean(
                    [record["alternative"]["stability_covered"] for record in pooled_records]
                )
            )
            if pooled_records
            else 0.0
        )
        pooled_false_signs = sum(
            record["null"]["false_sign_certificate"] for record in pooled_null_records
        )
        pooled_fwer_upper = (
            _upper(pooled_false_signs, len(pooled_null_records))
            if pooled_null_records
            else 1.0
        )
        pooled_errors = np.asarray(
            [
                record["alternative"]["scientific_target_mean_error"]
                for record in pooled_records
            ]
        )
        pooled_standardized_bias = (
            abs(float(pooled_errors.mean())) / float(pooled_errors.std(ddof=1))
            if len(pooled_errors) > 1 and pooled_errors.std(ddof=1) > 0
            else float("inf")
        )
        supported_cell_pass_rate = (
            sum(cell["passed"] for cell in supported) / len(supported) if supported else 0.0
        )
        strong = [cell for cell in cells if cell["spec"]["overlap"] == "strong"]
        moderate = [cell for cell in cells if cell["spec"]["overlap"] == "moderate"]
        strong_support = sum(cell["supported"] for cell in strong) / len(strong) if strong else 0.0
        moderate_support = (
            sum(cell["supported"] for cell in moderate) / len(moderate) if moderate else 0.0
        )
        corrected_improves = (
            pooled_naive_coverage >= criteria["simultaneous_coverage_min"]
            or pooled_coverage - pooled_naive_coverage >= 0.02
        )
        passed = bool(
            supported
            and supported_cell_pass_rate >= criteria["minimum_supported_cell_pass_rate"]
            and not any(cell["severe_failure"] for cell in supported)
            and strong_support >= criteria["minimum_strong_cell_support_rate"]
            and moderate_support >= criteria["minimum_moderate_cell_support_rate"]
            and criteria["simultaneous_coverage_min"]
            <= pooled_coverage
            <= criteria["simultaneous_coverage_max"]
            and pooled_fwer_upper <= criteria["fwer_upper_bound_max"]
            and pooled_standardized_bias <= criteria["standardized_absolute_bias_max"]
            and pooled_stability_coverage >= criteria["simultaneous_coverage_min"]
            and corrected_improves
        )
        profile_audits.append(
            {
                "profile": profile["name"],
                "supported_cells": len(supported),
                "supported_cell_pass_rate": supported_cell_pass_rate,
                "strong_cell_support_rate": strong_support,
                "moderate_cell_support_rate": moderate_support,
                "pooled_simultaneous_coverage": pooled_coverage,
                "pooled_naive_simultaneous_coverage": pooled_naive_coverage,
                "pooled_fwer_upper_95": pooled_fwer_upper,
                "pooled_standardized_absolute_bias": pooled_standardized_bias,
                "pooled_stability_coverage": pooled_stability_coverage,
                "corrected_improves_when_naive_fails": corrected_improves,
                "passed": passed,
                "cells": cells,
            }
        )
        if passed and selected is None:
            selected = profile
    selected_index = candidates["profiles"].index(selected) if selected is not None else None
    warning_profile = None
    if selected_index is not None:
        warning_profile = (
            candidates["profiles"][selected_index - 1]
            if selected_index > 0
            else UNBOUNDED_WARNING_FLOOR
        )
    artifact = {
        "schema_version": 2,
        "version": "stage3-directional-v1",
        "protocol": release_spec["protocol"],
        "validation_level": "directional",
        "calibration_rule_version": "pooled-directional-v2",
        "calibrated": selected is not None,
        "selection_status": "passed" if selected is not None else "no-profile-passed",
        "pass_profile": selected,
        "warning_floor_profile": warning_profile,
        "criteria": criteria,
        "audit": profile_audits,
    }
    encoded = json.dumps(artifact, sort_keys=True).encode()
    artifact["sha256"] = sha256(encoded).hexdigest()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("benchmarks/specs/stage3_threshold_candidates.json"),
    )
    parser.add_argument(
        "--release-spec",
        type=Path,
        default=Path("benchmarks/specs/stage3_release.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-output", type=Path)
    args = parser.parse_args()
    campaigns = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    release_spec = json.loads(args.release_spec.read_text(encoding="utf-8"))
    artifact = calibrate(campaigns, candidates, release_spec)
    encoded = json.dumps(artifact, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    if args.package_output is not None:
        if not artifact["calibrated"]:
            raise SystemExit(
                "no candidate threshold profile satisfies directional criteria; audit written"
            )
        args.package_output.parent.mkdir(parents=True, exist_ok=True)
        args.package_output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
