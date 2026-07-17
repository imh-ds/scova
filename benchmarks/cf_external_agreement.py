"""Generate the two-implementation numerical-agreement evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.cf_external_validation import (
    doubleml_apos,
    econml_drlearner,
    fixed_nuisance_score,
)
from benchmarks.cf_reference_campaign import _declaration, simulate_reference_cell
from scova._aipw import assemble_aipw
from scova.cf import SCOVACF, SCOVACFRefusal, canonical_checksum


def run_external_agreement(*, replications: int = 25) -> dict[str, object]:
    if replications < 10:
        raise ValueError("External agreement requires at least ten shared datasets")
    cell = {
        "n_per_group": 500,
        "n_groups": 3,
        "allocation": "balanced",
        "surface": "linear",
        "effect": "heterogeneous",
        "noise": "normal",
        "support": "strong",
        "learner": "linear",
    }
    doubleml_departures: list[float] = []
    econml_departures: list[float] = []
    fixed_max_error = 0.0
    implementation_rows: list[dict[str, object]] = []
    blocked: set[str] = set()
    for offset in range(replications):
        generated = simulate_reference_cell(cell, seed=700000 + offset)
        labels = generated.group_labels
        treatment = np.array([labels.index(value) for value in generated.data["group"]])
        outcome = generated.data["outcome"].to_numpy()
        result = SCOVACF().analyze(
            generated.data, _declaration(generated, cell, include_stability=False)
        )
        if isinstance(result, SCOVACFRefusal):
            raise RuntimeError(f"SCOVA-CF reference refused shared fixture: {result.status.code}")
        literal = fixed_nuisance_score(
            outcome, treatment, result.propensity_predictions, result.outcome_predictions
        )
        shared = assemble_aipw(
            outcome, treatment, result.propensity_predictions, result.outcome_predictions
        )
        fixed_max_error = max(
            fixed_max_error,
            max(
                float(np.max(np.abs(left - right)))
                for left, right in zip(literal, shared, strict=True)
            ),
        )
        doubleml = doubleml_apos(
            generated.data.loc[:, result.covariate_names].to_numpy(), outcome, treatment
        )
        econml = econml_drlearner(
            generated.data.loc[:, result.covariate_names].to_numpy(), outcome, treatment
        )
        implementation_rows.extend((doubleml.to_dict(), econml.to_dict()))
        if doubleml.status == "complete":
            scale = np.where(result.group_standard_errors > 0, result.group_standard_errors, np.nan)
            doubleml_departures.extend(
                (np.abs(np.asarray(doubleml.estimates) - result.group_means) / scale).tolist()
            )
        else:
            blocked.add("DoubleMLAPOS")
        if econml.status == "complete":
            reference = np.array(
                [
                    result.group_means[code] - result.group_means[0]
                    for code in range(1, len(labels))
                ]
            )
            reference_se = np.array(
                [
                    result.contrast(
                        {labels[code]: 1.0, labels[0]: -1.0},
                        name=f"external-{offset}-{code}",
                    ).standard_error
                    for code in range(1, len(labels))
                ]
            )
            econml_departures.extend(
                (np.abs(np.asarray(econml.estimates) - reference) / reference_se).tolist()
            )
        else:
            blocked.add("EconML.DRLearner")
    summaries = []
    for name, departures in (
        ("DoubleMLAPOS", doubleml_departures),
        ("EconML.DRLearner", econml_departures),
    ):
        values = np.asarray(departures, dtype=float)
        passed = bool(
            name not in blocked
            and values.size > 0
            and float(np.mean(values)) <= 0.25
            and float(np.max(values)) <= 1.0
        )
        summaries.append(
            {
                "implementation": name,
                "status": "complete" if passed else "blocked/agreement-tolerance",
                "mean_absolute_difference_in_scova_se": (
                    None if values.size == 0 else float(np.mean(values))
                ),
                "maximum_absolute_difference_in_scova_se": (
                    None if values.size == 0 else float(np.max(values))
                ),
            }
        )
    all_passed = fixed_max_error <= 1e-12 and all(
        row["status"] == "complete" for row in summaries
    )
    evidence: dict[str, object] = {
        "artifact_type": "scova-cf-external-agreement",
        "schema_version": 1,
        "replications": replications,
        "fixed_nuisance_maximum_absolute_error": fixed_max_error,
        "fixed_nuisance_tolerance": 1e-12,
        "end_to_end_mean_difference_tolerance_in_scova_se": 0.25,
        "end_to_end_maximum_difference_tolerance_in_scova_se": 1.0,
        "implementations": summaries,
        "run_details": implementation_rows,
        "all_numerical_agreement_gates_passed": all_passed,
    }
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=25)
    args = parser.parse_args()
    evidence = run_external_agreement(replications=args.replications)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    if not evidence["all_numerical_agreement_gates_passed"]:
        raise SystemExit("External numerical agreement did not pass")


if __name__ == "__main__":
    main()
