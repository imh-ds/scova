"""Select a checksum-bound SCOVA-CF support profile from calibration only."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum

LOWER_FEATURE = "minimum_ess_ratio"

# Optional selection metric.  When a spec sets it, calibration *ranks* the
# candidates that already clear the preregistered enrichment floor by the
# one-sided lower confidence bound of their unstable enrichment risk ratio, and
# selects the most robustly enriched rule (see ``calibrate``).  This is a
# held-out-blind generalization margin: v7 selected the candidate with the
# tightest thresholds, which happened to clear the point estimate by only 0.07
# (risk ratio 2.07) and then regressed to 1.95 out of sample; ranking on the
# lower bound instead prefers a rule whose enrichment is statistically well
# separated from the floor.  It never rejects a rule the protocol would accept
# and the held-out acceptance gate is unchanged (it still tests the point
# estimate).
_SELECTION_CONFIDENCE_METRIC = "unstable_risk_ratio_selection_confidence"


def _selection_z(metrics: MappingLike) -> float | None:
    """Return the one-sided normal multiplier for the selection margin, if set."""
    confidence = metrics.get(_SELECTION_CONFIDENCE_METRIC)
    if confidence is None:
        return None
    confidence = float(confidence)
    if not 0.5 <= confidence < 1.0:
        raise ValueError(
            f"{_SELECTION_CONFIDENCE_METRIC} must lie in [0.5, 1.0); got {confidence}"
        )
    return float(statistics.NormalDist().inv_cdf(confidence))


def _risk_ratio_lower_bound(
    unstable_bad: float,
    unstable_total: float,
    supported_bad: float,
    supported_total: float,
    z: float,
) -> float:
    """One-sided lower confidence bound of the risk ratio (Katz log method).

    A 0.5 continuity correction keeps the bound finite and well defined for
    zero cells.  Returns ``0.0`` when either arm has no observations at all.
    """
    if unstable_total <= 0 or supported_total <= 0:
        return 0.0
    a = unstable_bad + 0.5
    b = unstable_total - unstable_bad + 0.5
    c = supported_bad + 0.5
    d = supported_total - supported_bad + 0.5
    rr = (a / (a + b)) / (c / (c + d))
    se_log = float(np.sqrt(1.0 / a - 1.0 / (a + b) + 1.0 / c - 1.0 / (c + d)))
    return float(np.exp(np.log(rr) - z * se_log))


# Family-wise error control for the per-cell coverage/type-I gate.  The gate
# applies a two-sided Monte-Carlo test to every supported cell independently; at
# the plain 2-sigma level (~4.5% per cell) a *perfectly* calibrated campaign of
# ~16 cells trips the gate ~53% of the time by chance alone.  When a spec sets
# ``coverage_family_wise_error`` the multiplier is instead derived from a Sidak
# correction so the whole family of cells shares that error budget.
COVERAGE_FAMILY_WISE_ERROR_METRIC = "coverage_family_wise_error"


def _family_wise_multiplier(
    family_wise_error: float | None, family_size: int, base_multiplier: float
) -> float:
    """Two-sided normal multiplier that holds family-wise error across cells.

    Falls back to ``base_multiplier`` (the raw Monte-Carlo multiplier) when no
    family-wise budget is configured, so existing protocols are unchanged.
    """
    if family_wise_error is None:
        return base_multiplier
    error = float(family_wise_error)
    if not 0.0 < error < 1.0:
        raise ValueError(f"{COVERAGE_FAMILY_WISE_ERROR_METRIC} must lie in (0, 1); got {error}")
    per_cell = 1.0 - (1.0 - error) ** (1.0 / max(family_size, 1))
    return float(statistics.NormalDist().inv_cdf(1.0 - per_cell / 2.0))


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


def _unstable_enrichment(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    metrics: MappingLike,
) -> dict[str, Any]:
    supported = [record for record in records if _passes(record, thresholds)]
    unstable = [record for record in records if not _passes(record, thresholds)]

    def bad_counts(values: list[dict[str, Any]]) -> tuple[float, float]:
        contrasts = [contrast for record in values for contrast in record["contrasts"]]
        bad = float(
            sum(
                (not value["covered"])
                or abs(value["estimate"] - value["truth"]) > 2 * value["standard_error"]
                for value in contrasts
            )
        )
        return bad, float(len(contrasts))

    supported_bad_count, supported_total = bad_counts(supported)
    unstable_bad_count, unstable_total = bad_counts(unstable)
    supported_bad = supported_bad_count / supported_total if supported_total else 0.0
    unstable_bad = unstable_bad_count / unstable_total if unstable_total else 0.0
    risk_ratio = (
        np.inf
        if supported_bad == 0 and unstable_bad > 0
        else unstable_bad / supported_bad
        if supported_bad > 0
        else 0.0
    )
    passed = bool(
        unstable
        and unstable_bad - supported_bad
        >= float(metrics["minimum_unstable_absolute_enrichment"])
        and risk_ratio >= float(metrics["minimum_unstable_risk_ratio"])
    )
    selection_z = _selection_z(metrics)
    lower_bound = (
        None
        if selection_z is None
        else _risk_ratio_lower_bound(
            unstable_bad_count, unstable_total, supported_bad_count, supported_total, selection_z
        )
    )
    return {
        "passed": passed,
        "supported_count": len(supported),
        "unstable_count": len(unstable),
        "supported_bad_rate": supported_bad,
        "unstable_bad_rate": unstable_bad,
        "absolute_enrichment": unstable_bad - supported_bad,
        "risk_ratio": None if not np.isfinite(risk_ratio) else risk_ratio,
        "risk_ratio_lower_bound": lower_bound,
        "selection_confidence": (
            None if selection_z is None else float(metrics[_SELECTION_CONFIDENCE_METRIC])
        ),
    }


def _candidate_enrichments(
    records: list[dict[str, Any]],
    candidates: list[dict[str, float]],
    metrics: MappingLike,
) -> list[dict[str, Any]]:
    """Vectorize the preregistered enrichment screen across candidate rules."""
    if not candidates:
        return []
    feature_names = (LOWER_FEATURE, *UPPER_FEATURES)
    features = np.asarray(
        [[record["support_features"][name] for name in feature_names] for record in records],
        dtype=float,
    )
    limits = np.asarray(
        [[candidate[name] for name in feature_names] for candidate in candidates],
        dtype=float,
    )
    supported = features[:, 0, None] >= limits[None, :, 0]
    for index in range(1, len(feature_names)):
        supported &= features[:, index, None] <= limits[None, :, index]
    contrast_counts = np.asarray([len(record["contrasts"]) for record in records], dtype=float)
    bad_counts = np.asarray(
        [
            sum(
                (not value["covered"])
                or abs(value["estimate"] - value["truth"])
                > 2 * value["standard_error"]
                for value in record["contrasts"]
            )
            for record in records
        ],
        dtype=float,
    )
    supported_contrasts = supported.T @ contrast_counts
    supported_bad_counts = supported.T @ bad_counts
    unstable_contrasts = contrast_counts.sum() - supported_contrasts
    unstable_bad_counts = bad_counts.sum() - supported_bad_counts
    supported_bad_rates = np.divide(
        supported_bad_counts,
        supported_contrasts,
        out=np.zeros_like(supported_bad_counts),
        where=supported_contrasts > 0,
    )
    unstable_bad_rates = np.divide(
        unstable_bad_counts,
        unstable_contrasts,
        out=np.zeros_like(unstable_bad_counts),
        where=unstable_contrasts > 0,
    )
    risk_ratios = np.divide(
        unstable_bad_rates,
        supported_bad_rates,
        out=np.zeros_like(unstable_bad_rates),
        where=supported_bad_rates > 0,
    )
    risk_ratios[(supported_bad_rates == 0) & (unstable_bad_rates > 0)] = np.inf
    enrichments = unstable_bad_rates - supported_bad_rates
    supported_counts = supported.sum(axis=0)
    unstable_counts = len(records) - supported_counts
    minimum_risk_ratio = float(metrics["minimum_unstable_risk_ratio"])
    minimum_absolute = float(metrics["minimum_unstable_absolute_enrichment"])
    selection_z = _selection_z(metrics)
    lower_bounds = [
        (
            _risk_ratio_lower_bound(
                float(unstable_bad_counts[index]),
                float(unstable_contrasts[index]),
                float(supported_bad_counts[index]),
                float(supported_contrasts[index]),
                selection_z,
            )
            if selection_z is not None
            else None
        )
        for index in range(len(candidates))
    ]
    return [
        {
            # The pass gate stays exactly as preregistered (point estimate).
            # The lower confidence bound below is used only to *rank* qualifying
            # candidates during selection, so this margin never rejects a rule the
            # protocol would otherwise accept, and the held-out gate is unaffected.
            "passed": bool(
                unstable_counts[index] > 0
                and enrichments[index] >= minimum_absolute
                and risk_ratios[index] >= minimum_risk_ratio
            ),
            "supported_count": int(supported_counts[index]),
            "unstable_count": int(unstable_counts[index]),
            "supported_bad_rate": float(supported_bad_rates[index]),
            "unstable_bad_rate": float(unstable_bad_rates[index]),
            "absolute_enrichment": float(enrichments[index]),
            "risk_ratio": (
                None if not np.isfinite(risk_ratios[index]) else float(risk_ratios[index])
            ),
            "risk_ratio_lower_bound": lower_bounds[index],
            "selection_confidence": (
                None
                if selection_z is None
                else float(metrics[_SELECTION_CONFIDENCE_METRIC])
            ),
        }
        for index in range(len(candidates))
    ]


def _cell_gate(
    records: list[dict[str, Any]], metrics: MappingLike, *, multiplier: float | None = None
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
    # ``multiplier`` lets the caller inject a family-wise-corrected value across
    # the cells being adjudicated; without it the raw per-cell Monte-Carlo
    # multiplier is used (unchanged legacy behaviour).
    if multiplier is None:
        multiplier = float(metrics["monte_carlo_standard_error_multiplier"])
    multiplier = float(multiplier)
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
        "coverage_multiplier": multiplier,
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


def _candidate_usefulness(
    records: list[dict[str, Any]],
    candidates: list[dict[str, float]],
    protocol: CFValidationProtocol,
) -> list[tuple[bool, float]]:
    if not candidates:
        return []
    feature_names = (LOWER_FEATURE, *UPPER_FEATURES)
    features = np.asarray(
        [[record["support_features"][name] for name in feature_names] for record in records],
        dtype=float,
    )
    limits = np.asarray(
        [[candidate[name] for name in feature_names] for candidate in candidates],
        dtype=float,
    )
    supported = features[:, 0, None] >= limits[None, :, 0]
    for index in range(1, len(feature_names)):
        supported &= features[:, index, None] <= limits[None, :, index]
    cell_indices = np.asarray([int(record["cell_index"]) for record in records])
    strong_cells = sorted(
        {
            int(record["cell_index"])
            for record in records
            if _profile_eligible(protocol, record["cell"], record["cell_kind"])
        }
    )
    passing = np.zeros(len(candidates), dtype=int)
    supported_totals = np.zeros(len(candidates), dtype=int)
    for cell_index in strong_cells:
        cell_supported = supported[cell_indices == cell_index]
        supported_counts = cell_supported.sum(axis=0)
        supported_totals += supported_counts
        passing += supported_counts / len(cell_supported) >= float(
            protocol.metrics["minimum_strong_replication_pass_fraction"]
        )
    useful = (
        passing / len(strong_cells)
        >= float(protocol.metrics["minimum_strong_cell_pass_fraction"])
    )
    return [
        (bool(useful[index]), float(supported_totals[index]))
        for index in range(len(candidates))
    ]


def calibrate(
    protocol: CFValidationProtocol, evidence: dict[str, Any]
) -> dict[str, Any]:
    _verify_evidence(evidence)
    source = protocol.calibration_source
    source_lane = "calibration" if source is None else source.get("lane", "calibration")
    if evidence["lane"] != source_lane or not evidence["complete_frozen_lane"]:
        raise ValueError("Only the complete frozen development source can create a profile")
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
    candidate_values = list(candidates.values())
    usefulness = _candidate_usefulness(fit_records, candidate_values, protocol)
    ranked: list[tuple[float, tuple[float, ...], dict[str, float]]] = []
    for thresholds, (useful, objective) in zip(
        candidate_values, usefulness, strict=True
    ):
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
    attempts: list[
        tuple[
            float,
            tuple[float, ...],
            dict[str, float],
            list[dict[str, Any]],
            dict[str, Any],
        ]
    ] = []
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
    ranked_candidates = ranked[:candidate_limit]
    enrichments = _candidate_enrichments(
        audit_records,
        [thresholds for _, _, thresholds in ranked_candidates],
        protocol.metrics,
    )
    for (negative_objective, conservative, thresholds), enrichment in zip(
        ranked_candidates, enrichments, strict=True
    ):
        if protocol.calibration_enrichment_screening and not enrichment["passed"]:
            attempts.append(
                (-negative_objective, conservative, thresholds, [], enrichment)
            )
            continue
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
        attempts.append((-negative_objective, conservative, thresholds, audits, enrichment))
        if protocol.calibration_screening is None and passed:
            selected = thresholds
            selected_audit = audits
            break
    screening_diagnostics: dict[str, Any] | None = None
    if protocol.calibration_screening is not None and attempts:
        fully_screened = [
            attempt
            for attempt in attempts
            if all(audit["passed"] for audit in attempt[3])
            and (
                not protocol.calibration_enrichment_screening or attempt[4]["passed"]
            )
        ]
        closest = max(
            attempts,
            key=lambda attempt: (
                sum(audit["passed"] for audit in attempt[3]),
                attempt[4]["passed"],
                attempt[4]["absolute_enrichment"],
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
                "unstable_enrichment": closest[4],
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
                # v8: among candidates that meet the preregistered enrichment
                # floor and the retention floor, prefer the rule whose enrichment
                # is most robustly estimated -- the highest one-sided lower
                # confidence bound on the risk ratio -- before falling back to the
                # v7 conservative-threshold ordering.  This is held-out-blind and
                # fixes the v7 winner's-curse selection (a rule that cleared the
                # point estimate by 0.07 and then regressed below 2.0 out of
                # sample).  When no selection confidence is configured the key
                # degrades to the original v7 ordering.
                def _selection_key(attempt: Any) -> tuple[Any, ...]:
                    lower_bound = attempt[4].get("risk_ratio_lower_bound")
                    if lower_bound is None:
                        return (attempt[1], -attempt[0])
                    return (-float(lower_bound), attempt[1], -attempt[0])

                retained.sort(key=_selection_key)
                _, _, selected, selected_audit, selected_enrichment = retained[0]
                screening_diagnostics["selected_supported_replications"] = retained[0][0]
                screening_diagnostics["selected_unstable_enrichment"] = selected_enrichment
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
            "screened-enriched-robust-margin-within-retention-v8"
            if protocol.calibration_enrichment_screening
            and protocol.metrics.get(_SELECTION_CONFIDENCE_METRIC) is not None
            else "screened-enriched-conservative-within-retention-v7"
            if protocol.calibration_enrichment_screening
            else "screened-conservative-within-retention-v4"
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
