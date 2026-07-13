"""Strict Stage 4 shard aggregation and frozen directional evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from scipy.stats import beta

if __package__:
    from .stage4_campaign import frozen_cells, load_specification, work_items
else:  # ``python benchmarks/summarize_stage4.py``
    from stage4_campaign import frozen_cells, load_specification, work_items


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _upper(successes: int, total: int) -> float:
    return 1.0 if successes == total else float(beta.ppf(0.95, successes + 1, total - successes))


def _lower(successes: int, total: int) -> float:
    return 0.0 if successes == 0 else float(beta.ppf(0.05, successes, total - successes + 1))


def _rate(
    records: list[dict[str, Any]], field: str, *, true: bool = True
) -> tuple[int, int, float]:
    usable = [record for record in records if record.get(field) is not None]
    successes = sum(bool(record[field]) is true for record in usable)
    return successes, len(usable), successes / len(usable) if usable else 0.0


def _validate_shards(
    paths: list[Path], specification: dict[str, Any], tier: str
) -> list[dict[str, Any]]:
    cells = frozen_cells(tier, specification)
    expected = set(work_items(cells, int(specification["tiers"][tier]["repetitions"])))
    observed: set[tuple[int, int]] = set()
    records: list[dict[str, Any]] = []
    fingerprints: set[tuple[object, ...]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("sha256", None)
        if claimed != _canonical_digest(payload):
            raise ValueError(f"invalid shard checksum: {path}")
        fingerprint = (
            payload.get("schema_version"),
            payload.get("protocol"),
            payload.get("metric_contract", "legacy"),
            payload.get("tier"),
            payload.get("specification_sha256"),
            payload.get("catalog_sha256"),
            payload.get("threshold_artifact_sha256"),
            payload.get("shard_count"),
        )
        fingerprints.add(fingerprint)
        if payload["tier"] != tier:
            raise ValueError("shards must match the requested tier")
        if specification.get("metric_contract") == "stage4-v3" and (
            payload.get("schema_version") != 3
            or payload.get("metric_contract") != "stage4-v3"
        ):
            raise ValueError("v3 shards must use the v3 metric-contract schema")
        for record in payload["records"]:
            key = (int(record["cell_index"]), int(record["repetition"]))
            if key not in expected or key in observed:
                raise ValueError("shards contain a missing catalog record or duplicate work item")
            observed.add(key)
            records.append(record)
    if len(fingerprints) != 1:
        raise ValueError("shards disagree on protocol, catalog, threshold, or shard configuration")
    if observed != expected:
        raise ValueError(
            f"incomplete shards: expected {len(expected)} records, found {len(observed)}"
        )
    return records


def _summarize_legacy(
    paths: list[Path], specification: dict[str, Any], tier: str
) -> dict[str, Any]:
    records = _validate_shards(paths, specification, tier)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scenario[record["cell"]["scenario"]].append(record)
        by_cell[record["cell"]["id"]].append(record)
    criteria = specification["directional_pass_criteria"]
    accepted_per_cell = {
        name: sum(record["accepted"] for record in values) for name, values in by_cell.items()
    }
    expected_inferential_cells = {
        record["cell"]["id"]
        for record in records
        if record["cell"].get("expected_outcome", "inferential") == "inferential"
    }
    null = by_scenario["global_null"]
    false_rejections, null_total, _ = _rate(null, "any_rejection")
    coverage_successes, coverage_total, coverage = _rate(records, "simultaneous_coverage")
    strong = by_scenario["strong_complete_graph"]
    strong_successes = sum(bool(record["selected_edges"]) for record in strong)
    pairwise_only = by_scenario["pairwise_without_kway"]
    false_globals = sum(bool(record["selected_hyperedges"]) for record in pairwise_only)
    supported_pairs = sum(bool(record["selected_edges"]) for record in pairwise_only)
    rare = by_scenario["rare_group"]
    rare_refusals = sum(not record["accepted"] for record in rare)
    mutations = sum(
        not record["post_lock_mutation_rejected"]
        for record in records
        if record["status"] == "completed" and record["accepted"]
    )
    bounds = {
        "conditional_fwer_upper_bound_max": _upper(false_rejections, null_total),
        "simultaneous_coverage_lower_bound": _lower(coverage_successes, coverage_total),
        "simultaneous_coverage_upper_bound": _upper(coverage_successes, coverage_total),
        "strong_complete_graph_lower_bound": _lower(strong_successes, len(strong)),
        "pairwise_only_false_global_upper_bound": _upper(false_globals, len(pairwise_only)),
        "pairwise_only_supported_pair_lower_bound": _lower(supported_pairs, len(pairwise_only)),
        "rare_group_refusal_lower_bound": _lower(rare_refusals, len(rare)),
    }
    decisions = {
        "conditional_fwer_upper_bound_max": bounds["conditional_fwer_upper_bound_max"]
        <= criteria["conditional_fwer_upper_bound_max"],
        "simultaneous_coverage_min": bounds["simultaneous_coverage_lower_bound"]
        >= criteria["simultaneous_coverage_min"],
        "simultaneous_coverage_max": bounds["simultaneous_coverage_upper_bound"]
        <= criteria["simultaneous_coverage_max"],
        "minimum_strong_complete_graph_rate": bounds["strong_complete_graph_lower_bound"]
        >= criteria["minimum_strong_complete_graph_rate"],
        "maximum_false_global_claim_rate_pairwise_only": bounds[
            "pairwise_only_false_global_upper_bound"
        ]
        <= criteria["maximum_false_global_claim_rate_pairwise_only"],
        "minimum_supported_pairwise_edge_rate_pairwise_only": bounds[
            "pairwise_only_supported_pair_lower_bound"
        ]
        >= criteria["minimum_supported_pairwise_edge_rate_pairwise_only"],
        "minimum_rare_group_refusal_rate": bounds["rare_group_refusal_lower_bound"]
        >= criteria["minimum_rare_group_refusal_rate"],
        "post_lock_confirmatory_mutation_rate": mutations == 0,
        "minimum_accepted_repetitions_per_cell": all(
            accepted_per_cell[name] >= criteria["minimum_accepted_repetitions_per_cell"]
            for name in expected_inferential_cells
        ),
        "confidence_level": True,
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol": specification["protocol"],
        "tier": tier,
        "status": "pass" if all(decisions.values()) else "fail",
        "criteria": decisions,
        "bounds": bounds,
        "point_estimates": {"simultaneous_coverage": coverage, "post_lock_mutations": mutations},
        "accepted_repetitions_per_cell": accepted_per_cell,
        "record_count": len(records),
        "scenario_counts": {name: len(values) for name, values in by_scenario.items()},
    }
    payload["summary_sha256"] = _canonical_digest(payload)
    return payload


def _v3_metric(
    *,
    numerator: int,
    denominator: int,
    bound: float,
    threshold: float,
    comparison: str,
    eligible_cell_ids: list[str],
) -> dict[str, Any]:
    point_estimate = numerator / denominator if denominator else 0.0
    passed = bound >= threshold if comparison == ">=" else bound <= threshold
    return {
        "numerator": numerator,
        "denominator": denominator,
        "point_estimate": point_estimate,
        "bound": bound,
        "threshold": threshold,
        "comparison": comparison,
        "eligible_cell_ids": eligible_cell_ids,
        "passed": passed,
    }


def _summarize_v3(paths: list[Path], specification: dict[str, Any], tier: str) -> dict[str, Any]:
    records = _validate_shards(paths, specification, tier)
    criteria = specification["directional_pass_criteria"]
    inferential = [
        record
        for record in records
        if record["cell"].get("expected_outcome") == "inferential" and record["accepted"]
    ]
    inferential_cells = sorted(
        {
            record["cell"]["id"]
            for record in records
            if record["cell"].get("expected_outcome") == "inferential"
        }
    )
    expected_refusals = [
        record
        for record in records
        if record["cell"].get("expected_outcome") == "refusal"
    ]
    accepted_per_cell = {
        cell_id: sum(record["accepted"] for record in records if record["cell"]["id"] == cell_id)
        for cell_id in inferential_cells
    }
    metrics: dict[str, dict[str, Any]] = {}

    null = [record for record in inferential if record["cell"]["scenario"] == "global_null"]
    metrics["conditional_fwer_upper_bound_max"] = _v3_metric(
        numerator=sum(record["any_rejection"] for record in null),
        denominator=len(null),
        bound=_upper(sum(record["any_rejection"] for record in null), len(null)),
        threshold=criteria["conditional_fwer_upper_bound_max"],
        comparison="<=",
        eligible_cell_ids=sorted({record["cell"]["id"] for record in null}),
    )
    coverage = [record for record in inferential if record["simultaneous_coverage"] is not None]
    coverage_successes = sum(record["simultaneous_coverage"] for record in coverage)
    metrics["simultaneous_coverage_min"] = _v3_metric(
        numerator=coverage_successes,
        denominator=len(coverage),
        bound=_lower(coverage_successes, len(coverage)),
        threshold=criteria["simultaneous_coverage_min"],
        comparison=">=",
        eligible_cell_ids=sorted({record["cell"]["id"] for record in coverage}),
    )
    metrics["simultaneous_coverage_max"] = _v3_metric(
        numerator=coverage_successes,
        denominator=len(coverage),
        bound=_upper(coverage_successes, len(coverage)),
        threshold=criteria["simultaneous_coverage_max"],
        comparison="<=",
        eligible_cell_ids=metrics["simultaneous_coverage_min"]["eligible_cell_ids"],
    )
    pairwise = [
        record
        for record in inferential
        if record["cell"]["scenario"] == "pairwise_without_kway"
    ]
    pairwise_ids = sorted({record["cell"]["id"] for record in pairwise})
    false_globals = sum(bool(record["selected_hyperedges"]) for record in pairwise)
    supported_pairs = sum(bool(record["selected_edges"]) for record in pairwise)
    metrics["maximum_false_global_claim_rate_pairwise_only"] = _v3_metric(
        numerator=false_globals,
        denominator=len(pairwise),
        bound=_upper(false_globals, len(pairwise)),
        threshold=criteria["maximum_false_global_claim_rate_pairwise_only"],
        comparison="<=",
        eligible_cell_ids=pairwise_ids,
    )
    metrics["minimum_supported_pairwise_edge_rate_pairwise_only"] = _v3_metric(
        numerator=supported_pairs,
        denominator=len(pairwise),
        bound=_lower(supported_pairs, len(pairwise)),
        threshold=criteria["minimum_supported_pairwise_edge_rate_pairwise_only"],
        comparison=">=",
        eligible_cell_ids=pairwise_ids,
    )
    refusal_successes = sum(
        record.get("status") == "completed"
        and record.get("expected_refusal") == "insufficient_per_split_group_count"
        for record in expected_refusals
    )
    metrics["minimum_rare_group_refusal_rate"] = _v3_metric(
        numerator=refusal_successes,
        denominator=len(expected_refusals),
        bound=_lower(refusal_successes, len(expected_refusals)),
        threshold=criteria["minimum_rare_group_refusal_rate"],
        comparison=">=",
        eligible_cell_ids=sorted({record["cell"]["id"] for record in expected_refusals}),
    )
    strong_cells = sorted(
        {
            record["cell"]["id"]
            for record in inferential
            if record["cell"]["scenario"] == "strong_complete_graph"
        }
    )
    for cell_id in strong_cells:
        cell_records = [record for record in inferential if record["cell"]["id"] == cell_id]
        successes = sum(bool(record["complete_graph_recovered"]) for record in cell_records)
        name = f"strong_complete_graph:{cell_id}"
        metrics[name] = _v3_metric(
            numerator=successes,
            denominator=len(cell_records),
            bound=_lower(successes, len(cell_records)),
            threshold=criteria["minimum_strong_complete_graph_rate"],
            comparison=">=",
            eligible_cell_ids=[cell_id],
        )
    mutations = sum(not record["post_lock_mutation_rejected"] for record in inferential)
    metrics["post_lock_confirmatory_mutation_rate"] = _v3_metric(
        numerator=mutations,
        denominator=len(inferential),
        bound=mutations / len(inferential) if inferential else 1.0,
        threshold=0.0,
        comparison="<=",
        eligible_cell_ids=inferential_cells,
    )
    acceptance_passed = all(
        value >= criteria["minimum_accepted_repetitions_per_cell"]
        for value in accepted_per_cell.values()
    )
    decisions = {name: metric["passed"] for name, metric in metrics.items()}
    decisions["minimum_accepted_repetitions_per_cell"] = acceptance_passed
    decisions["confidence_level"] = True
    payload: dict[str, Any] = {
        "schema_version": 3,
        "protocol": specification["protocol"],
        "tier": tier,
        "status": "pass" if all(decisions.values()) else "fail",
        "criteria": decisions,
        "metrics": metrics,
        "accepted_repetitions_per_cell": accepted_per_cell,
        "record_count": len(records),
        "evaluation_population": "accepted_declared_inferential_v3",
    }
    payload["summary_sha256"] = _canonical_digest(payload)
    return payload


def summarize(paths: list[Path], specification: dict[str, Any], tier: str) -> dict[str, Any]:
    if specification.get("metric_contract") == "stage4-v3":
        return _summarize_v3(paths, specification, tier)
    return _summarize_legacy(paths, specification, tier)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--tier", required=True)
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmarks/specs/stage4_graph_release.json")
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.inputs, load_specification(args.spec), args.tier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
