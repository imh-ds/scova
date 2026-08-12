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
    if not successful:
        return {
            "completed_records": completed,
            "failure_rate": 1.0 if completed else None,
            "failure_rate_interval": _wilson_interval(failures, completed),
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


def run_comparative_study(replications: int, max_cells: int | None = None) -> dict[str, Any]:
    """Execute a bounded subset or the complete frozen 1,000-replication study."""
    if not 1 <= replications <= 1000:
        raise ValueError("replications must lie between 1 through 1000")
    cells = comparative_cells()
    selected = cells if max_cells is None else cells[:max_cells]
    if not selected:
        raise ValueError("max_cells must select at least one frozen cell")
    records: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(selected):
        for replication in range(replications):
            seed = 830_000_000 + cell_index * 10_000 + replication
            for row in score_replication(simulate_comparative_cell(cell, seed), seed):
                records.append({"cell_id": cell["cell_id"], "seed": seed, **row})
    return comparative_artifact(records, replications)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_comparative_study(args.replications, args.max_cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
