"""Select a checksum-bound SCOVA-CF support profile from calibration only."""

from __future__ import annotations

import argparse
import gzip
import itertools
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


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_evidence(evidence: dict[str, Any]) -> None:
    supplied = evidence.get("evidence_checksum")
    payload = {name: value for name, value in evidence.items() if name != "evidence_checksum"}
    if supplied != canonical_checksum(payload):
        raise ValueError("Campaign evidence checksum does not match its payload")


def _passes(record: dict[str, Any], thresholds: dict[str, float]) -> bool:
    features = record["support_features"]
    return bool(
        features[LOWER_FEATURE] >= thresholds[LOWER_FEATURE]
        and all(features[name] <= thresholds[name] for name in UPPER_FEATURES)
    )


def _cell_gate(
    records: list[dict[str, Any]], metrics: MappingLike
) -> tuple[bool, dict[str, Any]]:
    contrasts = [contrast for record in records for contrast in record["contrasts"]]
    if len(contrasts) < 2:
        return False, {"passed": False, "reason": "fewer-than-two-supported-contrasts"}
    coverage = float(np.mean([value["covered"] for value in contrasts]))
    coverage_mcse = np.sqrt(0.95 * 0.05 / len(contrasts))
    errors = np.array([value["estimate"] - value["truth"] for value in contrasts])
    empirical_sd = float(errors.std(ddof=1))
    bias = float(errors.mean())
    mean_se = float(np.mean([value["standard_error"] for value in contrasts]))
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else np.inf
    nulls = [value for value in contrasts if value["null"]]
    type_i_error = None if not nulls else float(np.mean([value["rejected"] for value in nulls]))
    multiplier = float(metrics["monte_carlo_standard_error_multiplier"])
    type_i_ok = True
    if type_i_error is not None:
        type_i_mcse = np.sqrt(0.05 * 0.95 / len(nulls))
        type_i_ok = abs(type_i_error - 0.05) <= multiplier * type_i_mcse
    passed = bool(
        abs(coverage - 0.95) <= multiplier * coverage_mcse
        and abs(bias) <= float(metrics["maximum_standardized_bias"]) * empirical_sd
        and float(metrics["minimum_se_ratio"])
        <= se_ratio
        <= float(metrics["maximum_se_ratio"])
        and type_i_ok
    )
    return passed, {
        "passed": passed,
        "supported_replications": len(records),
        "contrast_count": len(contrasts),
        "coverage": coverage,
        "bias": bias,
        "empirical_standard_deviation": empirical_sd,
        "standard_error_ratio": se_ratio,
        "type_i_error": type_i_error,
    }


def _screening_cell_gate(
    records: list[dict[str, Any]], metrics: MappingLike
) -> tuple[bool, dict[str, Any]]:
    """Apply the v4 one-sided calibration safety screen.

    Candidate calibration rejects anti-conservative coverage and type-I error,
    material bias, and underestimated standard errors. Conservative inference is
    reported but is reserved for the stricter held-out promotion decision.
    """
    contrasts = [contrast for record in records for contrast in record["contrasts"]]
    if len(contrasts) < 2:
        return False, {"passed": False, "reason": "fewer-than-two-supported-contrasts"}
    coverage = float(np.mean([value["covered"] for value in contrasts]))
    coverage_mcse = np.sqrt(0.95 * 0.05 / len(contrasts))
    errors = np.array([value["estimate"] - value["truth"] for value in contrasts])
    empirical_sd = float(errors.std(ddof=1))
    bias = float(errors.mean())
    mean_se = float(np.mean([value["standard_error"] for value in contrasts]))
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else np.inf
    nulls = [value for value in contrasts if value["null"]]
    type_i_error = None if not nulls else float(np.mean([value["rejected"] for value in nulls]))
    multiplier = float(metrics["monte_carlo_standard_error_multiplier"])
    coverage_ok = bool(coverage >= 0.95 - multiplier * coverage_mcse)
    type_i_ok = True
    if type_i_error is not None:
        type_i_mcse = np.sqrt(0.05 * 0.95 / len(nulls))
        type_i_ok = bool(type_i_error <= 0.05 + multiplier * type_i_mcse)
    bias_ok = bool(abs(bias) <= float(metrics["maximum_standardized_bias"]) * empirical_sd)
    se_ok = bool(
        float(metrics["minimum_se_ratio"])
        <= se_ratio
        <= float(metrics["maximum_se_ratio"])
    )
    passed = bool(coverage_ok and bias_ok and se_ok and type_i_ok)
    return passed, {
        "passed": passed,
        "gate_regime": "one-sided-calibration-screening",
        "supported_replications": len(records),
        "contrast_count": len(contrasts),
        "coverage": coverage,
        "coverage_floor": float(0.95 - multiplier * coverage_mcse),
        "bias": bias,
        "empirical_standard_deviation": empirical_sd,
        "standard_error_ratio": se_ratio,
        "type_i_error": type_i_error,
        "coverage_ok": coverage_ok,
        "bias_ok": bias_ok,
        "standard_error_ok": se_ok,
        "type_i_ok": type_i_ok,
    }


MappingLike = dict[str, float] | Any


def _structural(cell: dict[str, Any]) -> bool:
    return cell.get("support") == "structural-failure"


def _strong(
    cell: dict[str, Any],
    kind: str,
    minimum_expected_arm_count: float = 30.0,
    maximum_group_count: int | None = None,
) -> bool:
    if kind != "plasmode" and cell.get("support") != "strong":
        return False
    k = int(cell["n_groups"])
    if maximum_group_count is not None and k > maximum_group_count:
        return False
    allocation = str(cell["allocation"])
    if allocation == "balanced":
        weights = np.ones(k)
    elif allocation == "moderate":
        weights = np.geomspace(1.0, 0.35, k)
    elif allocation == "rare":
        weights = np.geomspace(1.0, 0.08, k)
    else:
        return False
    expected_minimum = int(cell["n_per_group"]) * k * float(weights.min() / weights.sum())
    return expected_minimum >= minimum_expected_arm_count


def _profile_scope(protocol: CFValidationProtocol) -> tuple[float, int | None]:
    compatibility = dict(protocol.reference_profile)
    return (
        max(
            float(protocol.calibration_gate_metrics["strong_support_minimum_expected_arm_count"]),
            float(compatibility.get("minimum_group_count", 2)),
        ),
        (
            None
            if "maximum_group_count" not in compatibility
            else int(compatibility["maximum_group_count"])
        ),
    )


def _profile_eligible(
    protocol: CFValidationProtocol, cell: dict[str, Any], kind: str
) -> bool:
    minimum, maximum = _profile_scope(protocol)
    return _strong(cell, kind, minimum, maximum)


def _usefulness(
    records: list[dict[str, Any]], thresholds: dict[str, float], protocol: CFValidationProtocol
) -> tuple[bool, float]:
    strong_cells = {
        int(record["cell_index"])
        for record in records
        if _profile_eligible(protocol, record["cell"], record["cell_kind"])
    }
    passing_cells = 0
    supported_total = 0
    for cell_index in strong_cells:
        cell_records = [
            record
            for record in records
            if record["cell_index"] == cell_index and not record["refused"]
        ]
        supported = sum(_passes(record, thresholds) for record in cell_records)
        supported_total += supported
        if cell_records and supported / len(cell_records) >= float(
            protocol.metrics["minimum_strong_replication_pass_fraction"]
        ):
            passing_cells += 1
    useful = bool(
        strong_cells
        and passing_cells / len(strong_cells)
        >= float(protocol.metrics["minimum_strong_cell_pass_fraction"])
    )
    return useful, float(supported_total)


def calibrate(
    protocol: CFValidationProtocol, evidence: dict[str, Any]
) -> dict[str, Any]:
    _verify_evidence(evidence)
    if evidence["lane"] != "calibration" or not evidence["complete_frozen_lane"]:
        raise ValueError("Only a complete frozen calibration lane can create a profile")
    source = protocol.calibration_source
    matches_protocol = evidence["protocol_checksum"] == protocol.checksum
    matches_declared_source = bool(
        source
        and evidence["protocol_checksum"] == source["protocol_checksum"]
        and evidence.get("evidence_checksum") == source["evidence_checksum"]
        and evidence.get("git_commit") == source["git_commit"]
    )
    if not matches_protocol and not matches_declared_source:
        raise ValueError("Calibration evidence uses a different protocol or declared source")
    execution_failures = [
        record for record in evidence["records"] if record.get("status_code") == "execution-error"
    ]
    if execution_failures:
        result: dict[str, Any] = {
            "artifact_type": "scova-cf-support-calibration",
            "schema_version": 2,
            "protocol_checksum": protocol.checksum,
            "calibration_evidence_checksum": evidence["evidence_checksum"],
            "all_calibration_gates_passed": False,
            "execution_failure_count": len(execution_failures),
            "thresholds": None,
            "candidate_profile": None,
            "audit": [],
        }
        result["calibration_artifact_checksum"] = canonical_checksum(result)
        return result
    split = int(protocol.calibration.count * protocol.calibration_fit_fraction)
    usable = [
        record
        for record in evidence["records"]
        if not record["refused"] and "support_features" in record
    ]
    fit_records = [record for record in usable if int(record["repetition"]) < split]
    audit_records = [record for record in usable if int(record["repetition"]) >= split]
    quantiles = protocol.threshold_quantiles
    if quantiles is None:
        lower_q = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
        upper_q = (0.80, 0.90, 0.95, 0.975, 0.99, 1.0)
    else:  # pragma: no cover - retained for future schema extension
        lower_q = tuple(quantiles["minimum_ess_ratio"])
        upper_q = tuple(quantiles["upper_metrics"])
    grids = {
        LOWER_FEATURE: tuple(
            float(np.quantile([r["support_features"][LOWER_FEATURE] for r in fit_records], q))
            for q in lower_q
        ),
        **{
            name: tuple(
                float(np.quantile([r["support_features"][name] for r in fit_records], q))
                for q in upper_q
            )
            for name in UPPER_FEATURES
        },
    }
    feature_names = (LOWER_FEATURE, *UPPER_FEATURES)
    candidates: dict[tuple[float, ...], dict[str, float]] = {}
    # The preregistered family uses one common upper-tail quantile, plus a
    # deterministic one-feature deviation. This spans strict-to-permissive
    # rules without an impractical 6^7 Cartesian search.
    for lower, common_index in itertools.product(grids[LOWER_FEATURE], range(len(upper_q))):
        baseline = {
            LOWER_FEATURE: lower,
            **{name: grids[name][common_index] for name in UPPER_FEATURES},
        }
        candidates[tuple(baseline[name] for name in feature_names)] = baseline
        for changed_name in UPPER_FEATURES:
            for changed_index in range(len(upper_q)):
                changed = {
                    **baseline,
                    changed_name: grids[changed_name][changed_index],
                }
                candidates[tuple(changed[name] for name in feature_names)] = changed
    ranked: list[tuple[float, tuple[float, ...], dict[str, float]]] = []
    for thresholds in candidates.values():
        useful, objective = _usefulness(fit_records, thresholds, protocol)
        if useful:
            # Smaller upper limits and larger ESS floors win exact objective ties.
            conservative = (
                -thresholds[LOWER_FEATURE],
                *(thresholds[name] for name in UPPER_FEATURES),
            )
            ranked.append((-objective, conservative, thresholds))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: dict[str, float] | None = None
    selected_audit: list[dict[str, Any]] = []
    attempts: list[tuple[float, tuple[float, ...], dict[str, float], list[dict[str, Any]]]] = []
    candidate_limit = len(ranked) if protocol.calibration_screening is not None else 256
    records_by_cell = {
        cell_index: [
            record for record in evidence["records"] if int(record["cell_index"]) == cell_index
        ]
        for cell_index in sorted({int(record["cell_index"]) for record in evidence["records"]})
    }
    audit_records_by_cell = {
        cell_index: [
            record for record in audit_records if int(record["cell_index"]) == cell_index
        ]
        for cell_index in records_by_cell
    }
    for negative_objective, conservative, thresholds in ranked[:candidate_limit]:
        audits = []
        passed = True
        for cell_index, all_cell in records_by_cell.items():
            cell = all_cell[0]["cell"]
            if _structural(cell):
                audit = {
                    "passed": all(r["refused"] for r in all_cell),
                    "structural_refusal_rate": float(np.mean([r["refused"] for r in all_cell])),
                }
            else:
                supported = [
                    r
                    for r in audit_records_by_cell[cell_index]
                    if _passes(r, thresholds)
                ]
                if protocol.calibration_screening is None:
                    audit_passed, audit = _cell_gate(supported, protocol.metrics)
                    if not _strong(
                        cell,
                        all_cell[0]["cell_kind"],
                        float(protocol.metrics["strong_support_minimum_expected_arm_count"]),
                    ) and not supported:
                        audit_passed = True
                        audit = {"passed": True, "reason": "unstable-cell-no-supported-results"}
                elif not _profile_eligible(protocol, cell, all_cell[0]["cell_kind"]):
                    audit_passed = True
                    audit = {
                        "passed": True,
                        "reason": "outside-calibration-screening-eligibility",
                        "supported_replications": len(supported),
                    }
                else:
                    audit_passed, audit = _screening_cell_gate(
                        supported, protocol.calibration_gate_metrics
                    )
                audit["passed"] = audit_passed
            passed &= bool(audit["passed"])
            audits.append({"cell_index": cell_index, "cell": cell, **audit})
        attempts.append((-negative_objective, conservative, thresholds, audits))
        if protocol.calibration_screening is None and passed:
            selected = thresholds
            selected_audit = audits
            break
    screening_diagnostics: dict[str, Any] | None = None
    if protocol.calibration_screening is not None and attempts:
        fully_screened = [
            attempt for attempt in attempts if all(audit["passed"] for audit in attempt[3])
        ]
        closest = max(
            attempts,
            key=lambda attempt: (
                sum(audit["passed"] for audit in attempt[3]),
                attempt[0],
            ),
        )
        screening_diagnostics = {
            "gate_regime": "one-sided-calibration-screening",
            "evaluated_candidate_count": len(attempts),
            "fully_screened_candidate_count": len(fully_screened),
            "candidate_retention_fraction": protocol.calibration_candidate_retention_fraction,
            "closest_candidate": {
                "supported_replications": closest[0],
                "thresholds": closest[2],
                "audit": closest[3],
            },
        }
        if fully_screened:
            # Retention is relative to the best useful preregistered rule, not
            # merely the best already-screened rule.  Otherwise the retention
            # constraint becomes vacuous whenever screening rules out the most
            # permissive candidates.
            maximum_retention = max(attempt[0] for attempt in attempts)
            retained = [
                attempt
                for attempt in fully_screened
                if attempt[0]
                >= protocol.calibration_candidate_retention_fraction * maximum_retention
            ]
            screening_diagnostics["maximum_supported_replications"] = maximum_retention
            screening_diagnostics["minimum_required_supported_replications"] = (
                protocol.calibration_candidate_retention_fraction * maximum_retention
            )
            if retained:
                retained.sort(key=lambda attempt: (attempt[1], -attempt[0]))
                _, _, selected, selected_audit = retained[0]
                screening_diagnostics["selected_supported_replications"] = retained[0][0]
            else:
                screening_diagnostics["selection_refusal_reason"] = (
                    "no-fully-screened-candidate-met-retention-floor"
                )
    result: dict[str, Any] = {
        "artifact_type": "scova-cf-support-calibration",
        "schema_version": 2,
        "protocol_checksum": protocol.checksum,
        "calibration_evidence_checksum": evidence["evidence_checksum"],
        "calibration_source_protocol_checksum": evidence["protocol_checksum"],
        "fit_replications_per_cell": split,
        "audit_replications_per_cell": protocol.calibration.count - split,
        "candidate_count": len(ranked),
        "evaluated_top_candidates": candidate_limit,
        "threshold_selection": (
            "screened-conservative-within-retention-v4"
            if protocol.calibration_screening is not None
            else "preregistered-grid-usefulness-and-operating-gates-v2"
        ),
        "thresholds": selected,
        "all_calibration_gates_passed": selected is not None,
        "execution_failure_count": 0,
        "audit": selected_audit,
    }
    if screening_diagnostics is not None:
        result["screening_diagnostics"] = screening_diagnostics
    if selected is not None:
        result["candidate_profile"] = CFSupportProfile(
            profile_id=f"{protocol.protocol_id}-candidate",
            protocol_checksum=protocol.checksum,
            calibration_evidence_checksum=evidence["evidence_checksum"],
            validation_evidence_checksum=None,
            thresholds=selected,
            compatibility=protocol.reference_profile,
        ).to_dict()
    else:
        result["candidate_profile"] = None
    result["calibration_artifact_checksum"] = canonical_checksum(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--calibration-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument(
        "--require-candidate",
        action="store_true",
        help="Exit nonzero after writing the calibration report when no profile is promoted.",
    )
    args = parser.parse_args()
    result = calibrate(CFValidationProtocol.load(args.spec), read_json(args.calibration_evidence))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    if args.candidate_output is not None and result["candidate_profile"] is not None:
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(result["candidate_profile"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.require_candidate and result["candidate_profile"] is None:
        raise SystemExit(
            "Calibration did not promote a candidate support profile: no preregistered "
            "threshold rule passed the internal calibration gates. Inspect the calibration "
            "report artifact; do not dispatch external agreement, inference, or validation."
        )


if __name__ == "__main__":
    main()
