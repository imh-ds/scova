"""Methods-only SCOVA-CF factorial study.

This module intentionally does not call a calibration gate.  It estimates
continuous performance summaries within frozen simulated DGPs and cannot emit
a support profile or a qualification conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmarks.cf_reference_campaign import (
    _git_commit,
    dependency_lock_checksum,
    fit_campaign_record,
    simulate_reference_cell,
)
from scova.cf import (
    ARTIFACT_SCHEMA_VERSION,
    METHODS_STUDY_ID,
    StudyProgram,
    canonical_checksum,
    factorial_cells,
    methods_design,
)


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    """Continuous summaries and their Monte-Carlo uncertainty inputs."""
    values = list(records)
    completed = len(values)
    refused = sum(bool(value.get("refused")) for value in values)
    contrasts = [
        contrast for value in values if not value.get("refused")
        for contrast in value.get("contrasts", [])
    ]
    if not contrasts:
        return {
            "completed_replications": completed,
            "refused_replications": refused,
            "refusal_rate": refused / completed if completed else None,
            "bias_over_sd": None,
            "bias_over_sd_mcse": None,
            "coverage": None,
            "coverage_mcse": None,
            "standard_error_ratio": None,
        }
    errors = np.array([item["estimate"] - item["truth"] for item in contrasts], dtype=float)
    empirical_sd = float(errors.std(ddof=1))
    coverage = float(np.mean([item["covered"] for item in contrasts]))
    se_ratio = float(np.mean([item["standard_error"] for item in contrasts])) / empirical_sd
    ratio = float(abs(errors.mean()) / empirical_sd) if empirical_sd else None
    ratio_mcse = float(1 / np.sqrt(len(errors))) if empirical_sd else None
    coverage_mcse = float(np.sqrt(coverage * (1 - coverage) / len(contrasts)))
    return {
        "completed_replications": completed,
        "refused_replications": refused,
        "refusal_rate": refused / completed if completed else None,
        "bias_over_sd": ratio,
        "bias_over_sd_mcse": ratio_mcse,
        "bias_over_sd_mc_interval": (
            None if ratio is None or ratio_mcse is None
            else [max(0.0, ratio - 1.96 * ratio_mcse), ratio + 1.96 * ratio_mcse]
        ),
        "coverage": coverage,
        "coverage_mcse": coverage_mcse,
        "coverage_mc_interval": [
            max(0.0, coverage - 1.96 * coverage_mcse),
            min(1.0, coverage + 1.96 * coverage_mcse),
        ],
        "standard_error_ratio": se_ratio,
    }


def _effect_table(cells: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Unaliased main-effect contrasts for the primary regular factorial."""
    result: dict[str, dict[str, float | None]] = {}
    for factor, levels in methods_design()["factors"].items():
        low = [cell["summary"]["bias_over_sd"] for cell in cells if cell["cell"][factor] == levels[0]]
        high = [cell["summary"]["bias_over_sd"] for cell in cells if cell["cell"][factor] == levels[1]]
        low_values = [float(value) for value in low if value is not None]
        high_values = [float(value) for value in high if value is not None]
        result[factor] = {
            "low_mean_bias_over_sd": float(np.mean(low_values)) if low_values else None,
            "high_mean_bias_over_sd": float(np.mean(high_values)) if high_values else None,
            "difference": (
                float(np.mean(high_values) - np.mean(low_values))
                if low_values and high_values else None
            ),
        }
    return result


def methods_artifact(
    *, surface_family: str, cell_records: dict[str, list[dict[str, Any]]],
    replications_per_cell: int = 1000,
) -> dict[str, Any]:
    """Build a checksum-bound methods artifact, rejecting qualification fields."""
    if surface_family not in {"interaction", "smooth-nonlinear", "threshold"}:
        raise ValueError("unknown methods surface family")
    cells = factorial_cells(surface_family)
    expected_ids = {cell["cell_id"] for cell in cells}
    if set(cell_records) != expected_ids:
        raise ValueError("Methods artifact must account for every frozen cell exactly once")
    summaries = [
        {"cell": cell, "summary": summarize_records(cell_records[cell["cell_id"]])}
        for cell in cells
    ]
    completed = sum(int(row["summary"]["completed_replications"]) for row in summaries)
    payload: dict[str, Any] = {
        "artifact_type": "scova-cf-methods-study",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "program_type": StudyProgram.METHODS.value,
        "study_id": METHODS_STUDY_ID,
        "surface_family": surface_family,
        "design_checksum": methods_design()["design_checksum"],
        "dependency_lock_checksum": dependency_lock_checksum(),
        "git_commit": _git_commit(),
        "planned_cells": len(cells),
        "planned_replications_per_cell": replications_per_cell,
        "planned_replications": len(cells) * replications_per_cell,
        "completed_replications": completed,
        "complete": completed == len(cells) * replications_per_cell,
        "source_evidence_ids": [],
        "cell_summaries": summaries,
        "main_effects": _effect_table(summaries),
        "interpretation": (
            "Methods evidence within the declared simulated DGPs only; it does not "
            "create a candidate profile, support threshold, or qualification claim."
        ),
    }
    payload["artifact_checksum"] = canonical_checksum(payload)
    return payload


def run(surface_family: str, *, reps: int) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for cell_index, cell in enumerate(factorial_cells(surface_family)):
        records[cell["cell_id"]] = []
        simulation_cell = {name: value for name, value in cell.items() if name != "cell_id"}
        for repetition in range(reps):
            seed = 710_000_000 + cell_index * 10_000 + repetition
            fitted = fit_campaign_record(
                simulate_reference_cell(simulation_cell, seed=seed),
                simulation_cell, include_stability=False, seed=seed,
            )
            records[cell["cell_id"]].append(fitted)
    return methods_artifact(
        surface_family=surface_family, cell_records=records, replications_per_cell=reps
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-family", choices=("interaction", "smooth-nonlinear", "threshold"), required=True)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.reps > 1000:
        parser.error("reps must lie within the frozen methods design")
    artifact = run(args.surface_family, reps=args.reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
