"""Methods-only runner and artifact builder for the two-group comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from benchmarks.cf_comparative_estimators import score_replication
from benchmarks.cf_comparative_simulation import comparative_cells, simulate_comparative_cell
from benchmarks.cf_reference_campaign import dependency_lock_checksum
from scova.cf import canonical_checksum

_PROTOCOL_PATH = Path(__file__).with_name("specs") / "cf_two_group_comparative_methods_v2.json"
_METHODS = (
    "scova-cf",
    "linear-ancova",
    "independent-aipw",
    "psm-att",
    "econml-drlearner",
    "econml-drlearner-conservative",
)
_FORBIDDEN_FIELDS = frozenset({"profile", "calibration", "promotion", "qualification"})


def _protocol() -> dict[str, Any]:
    return json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _interval(values: np.ndarray, *, seed: int = 1701) -> list[float] | None:
    if not len(values):
        return None
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    draws = np.array(
        [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(400)]
    )
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _wilson_interval(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = float(norm.ppf(0.975))
    proportion = successes / count
    denominator = 1 + z**2 / count
    centre = (proportion + z**2 / (2 * count)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2))
    return [float(centre - half_width / denominator), float(centre + half_width / denominator)]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if row["estimate"] is not None and row["standard_error"] is not None
    ]
    completed = len(rows)
    failures = completed - len(successful)
    warnings = sum(row["status"] != "ok" for row in successful)
    if not successful:
        return {
            "completed_records": completed,
            "failure_rate": 1.0 if completed else None,
            "failure_rate_interval": _wilson_interval(failures, completed),
            "warning_rate": None,
            "warning_rate_interval": None,
            "bias": None,
            "rmse": None,
            "empirical_sd": None,
            "mean_reported_se": None,
            "se_ratio": None,
            "coverage": None,
            "coverage_interval": None,
            "treated_retained_fraction": None,
            "median_absolute_error": None,
            "absolute_error_p95": None,
            "maximum_absolute_error": None,
        }
    estimates = np.array([float(row["estimate"]) for row in successful])
    truth = np.array([float(row["truth"]) for row in successful])
    errors = estimates - truth
    absolute_errors = np.abs(errors)
    standard_errors = np.array([float(row["standard_error"]) for row in successful])
    covered = np.abs(errors) <= 1.96 * standard_errors
    empirical_sd = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    retention = [
        float(row["details"]["treated_retained_fraction"])
        for row in successful
        if "treated_retained_fraction" in row["details"]
    ]
    return {
        "completed_records": completed,
        "failure_rate": failures / completed if completed else None,
        "failure_rate_interval": _wilson_interval(failures, completed),
        "warning_rate": warnings / len(successful),
        "warning_rate_interval": _wilson_interval(warnings, len(successful)),
        "bias": float(np.mean(errors)),
        "bias_interval": _interval(errors),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "rmse_interval": _interval(errors**2),
        "empirical_sd": empirical_sd,
        "mean_reported_se": float(np.mean(standard_errors)),
        "se_ratio": float(np.mean(standard_errors) / empirical_sd) if empirical_sd else None,
        "coverage": float(np.mean(covered)),
        "coverage_interval": _wilson_interval(int(np.sum(covered)), len(covered)),
        "treated_retained_fraction": float(np.mean(retention)) if retention else None,
        "treated_retained_fraction_interval": _interval(np.asarray(retention)),
        "median_absolute_error": float(np.median(absolute_errors)),
        "absolute_error_p95": float(np.quantile(absolute_errors, 0.95)),
        "maximum_absolute_error": float(np.max(absolute_errors)),
    }


def cell_level_summaries(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return ATE and ATT method summaries separately for every observed DGP cell."""
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for record in records:
        cell_id = str(record["cell_id"])
        estimand = str(record["estimand"])
        grouped.setdefault(cell_id, {"ate": defaultdict(list), "att": defaultdict(list)})[
            estimand
        ][str(record["method"])].append(record)
    return {
        estimand: {
            cell_id: {method: _summary(rows) for method, rows in methods.items()}
            for cell_id, summaries in grouped.items()
            for methods in [summaries[estimand]]
        }
        for estimand in ("ate", "att")
    }


def comparative_artifact(records: Iterable[dict[str, Any]], replications: int) -> dict[str, Any]:
    """Build a provenance-bound descriptive artifact from per-method records."""
    if not 1 <= replications <= 1000:
        raise ValueError("replications must lie between 1 through 1000")
    values = list(records)
    for record in values:
        forbidden = _FORBIDDEN_FIELDS.intersection(record)
        if forbidden:
            raise ValueError(
                f"methods records cannot carry qualification fields: {sorted(forbidden)}"
            )
    by_estimand: dict[str, dict[str, list[dict[str, Any]]]] = {
        "ate": defaultdict(list),
        "att": defaultdict(list),
    }
    for record in values:
        if record["estimand"] not in by_estimand:
            raise ValueError("comparative records must identify ate or att")
        by_estimand[record["estimand"]][record["method"]].append(record)
    cell_summaries = cell_level_summaries(values)
    cells = comparative_cells()
    protocol = _protocol()
    final_replications = int(protocol["final_replications_per_cell"])
    expected_records = len(cells) * final_replications * len(_METHODS)
    payload: dict[str, Any] = {
        "artifact_type": "scova-cf-comparative-methods",
        "program_type": "methods",
        "protocol_id": protocol["protocol_id"],
        "protocol_checksum": canonical_checksum(protocol),
        "dependency_lock_checksum": dependency_lock_checksum(),
        "git_commit": _git_commit(),
        "planned_cells": len(cells),
        "completed_cells": len({record["cell_id"] for record in values}),
        "planned_replications_per_cell": final_replications,
        "requested_replications_per_cell": replications,
        "completed_records": len(values),
        "planned_records": expected_records,
        "complete": (
            replications == final_replications
            and len(values) == expected_records
            and {record["cell_id"] for record in values} == {cell["cell_id"] for cell in cells}
        ),
        "source_evidence_ids": [],
        "ate_summaries": {name: _summary(rows) for name, rows in by_estimand["ate"].items()},
        "att_summaries": {name: _summary(rows) for name, rows in by_estimand["att"].items()},
        "cell_ate_summaries": cell_summaries["ate"],
        "cell_att_summaries": cell_summaries["att"],
        "records": values,
        "interpretation": (
            "Descriptive methods evidence within frozen simulated DGPs only. It does not "
            "qualify SCOVA-CF, certify observational causal identification, or create a profile."
        ),
    }
    payload["artifact_checksum"] = canonical_checksum(payload)
    return payload


def aggregate_comparative_shards(shards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combine one checksum-compatible artifact per frozen v2 cell."""
    values = list(shards)
    if not values:
        raise ValueError("at least one comparative shard is required")
    protocol = _protocol()
    expected_cell_ids = {str(cell["cell_id"]) for cell in comparative_cells()}
    expected_checksum = canonical_checksum(protocol)
    replications = values[0].get("requested_replications_per_cell")
    dependency_checksum = values[0].get("dependency_lock_checksum")
    frozen_commit = values[0].get("git_commit")
    if not isinstance(replications, int):
        raise ValueError("comparative shard is missing requested replication count")
    records: list[dict[str, Any]] = []
    shard_checksums: list[str] = []
    seen_cells: set[str] = set()
    for shard in values:
        recorded_checksum = shard.get("artifact_checksum")
        computed_checksum = canonical_checksum(
            {name: value for name, value in shard.items() if name != "artifact_checksum"}
        )
        if recorded_checksum != computed_checksum:
            raise ValueError("comparative shard has a mismatched artifact checksum")
        if shard.get("program_type") != "methods":
            raise ValueError("comparative aggregation requires methods artifacts")
        if shard.get("protocol_checksum") != expected_checksum:
            raise ValueError("comparative shard has a mismatched protocol checksum")
        if shard.get("requested_replications_per_cell") != replications:
            raise ValueError("comparative shards must use the same replication count")
        if shard.get("dependency_lock_checksum") != dependency_checksum:
            raise ValueError("comparative shards must use the same dependency lock")
        if shard.get("git_commit") != frozen_commit:
            raise ValueError("comparative shards must use the same frozen commit")
        shard_cells = {str(record["cell_id"]) for record in shard.get("records", [])}
        if len(shard_cells) != 1:
            raise ValueError("each comparative shard must contain exactly one cell")
        cell_id = next(iter(shard_cells))
        if cell_id in seen_cells:
            raise ValueError("comparative aggregation received a duplicate cell shard")
        seen_cells.add(cell_id)
        records.extend(shard["records"])
        shard_checksums.append(str(shard["artifact_checksum"]))
    if seen_cells != expected_cell_ids:
        raise ValueError("comparative aggregation requires every frozen cell exactly once")
    artifact = comparative_artifact(records, replications)
    artifact["source_shard_checksums"] = sorted(shard_checksums)
    artifact["source_shard_frozen_commit"] = frozen_commit
    artifact["artifact_checksum"] = canonical_checksum(
        {name: value for name, value in artifact.items() if name != "artifact_checksum"}
    )
    return artifact


def run_comparative_study(
    replications: int,
    max_cells: int | None = None,
    cell_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Execute a bounded subset or the complete frozen 1,000-replication study."""
    if not 1 <= replications <= 1000:
        raise ValueError("replications must lie between 1 through 1000")
    cells = comparative_cells()
    if cell_ids is not None and max_cells is not None:
        raise ValueError("cell_ids and max_cells cannot be supplied together")
    if cell_ids is None:
        selected = cells if max_cells is None else cells[:max_cells]
    else:
        requested = set(cell_ids)
        known = {str(cell["cell_id"]) for cell in cells}
        if not requested or requested.difference(known):
            raise ValueError("cell_ids must name one or more frozen comparative cells")
        selected = tuple(cell for cell in cells if str(cell["cell_id"]) in requested)
    if not selected:
        raise ValueError("max_cells must select at least one frozen cell")
    cell_indices = {str(cell["cell_id"]): index for index, cell in enumerate(cells)}
    records: list[dict[str, Any]] = []
    for cell in selected:
        cell_index = cell_indices[str(cell["cell_id"])]
        for replication in range(replications):
            seed = 830_000_000 + cell_index * 10_000 + replication
            for row in score_replication(simulate_comparative_cell(cell, seed), seed):
                records.append({"cell_id": cell["cell_id"], "seed": seed, **row})
    return comparative_artifact(records, replications)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--cell-id", action="append")
    parser.add_argument("--aggregate-input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.aggregate_input:
        if args.cell_id or args.max_cells is not None:
            parser.error("aggregation cannot be combined with cell selection")
        input_paths = [
            match
            for path in args.aggregate_input
            for match in sorted(path.parent.glob(path.name))
        ]
        if not input_paths:
            parser.error("aggregation input did not match any shard artifacts")
        artifact = aggregate_comparative_shards(
            [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
        )
    else:
        artifact = run_comparative_study(
            args.replications,
            args.max_cells,
            tuple(args.cell_id) if args.cell_id else None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
