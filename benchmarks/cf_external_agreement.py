"""Generate frozen shared-score and end-to-end external-agreement evidence."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cf_external_validation import (
    doubleml_apos,
    doubleml_shared_score,
    econml_drlearner,
    fixed_nuisance_score,
)
from benchmarks.cf_reference_campaign import (
    _declaration,
    dependency_lock_checksum,
    simulate_reference_cell,
)
from scova._aipw import assemble_aipw
from scova.cf import (
    SCOVACF,
    CFValidationProtocol,
    SCOVACFRefusal,
    canonical_checksum,
)


def _maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _environment() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "doubleml",
        "econml",
    ):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _summary(name: str, signed: list[float], blocked: list[str]) -> dict[str, Any]:
    values = np.asarray(signed, dtype=float)
    absolute = np.abs(values)
    passed = bool(
        not blocked
        and values.size
        and float(absolute.mean()) <= 0.25
        and float(absolute.max()) <= 1.0
        and abs(float(values.mean())) <= 0.05
    )
    return {
        "implementation": name,
        "status": "complete" if passed else "blocked/agreement-tolerance",
        "mean_absolute_difference_in_scova_se": (
            None if not values.size else float(absolute.mean())
        ),
        "maximum_absolute_difference_in_scova_se": (
            None if not values.size else float(absolute.max())
        ),
        "mean_signed_difference_in_scova_se": (
            None if not values.size else float(values.mean())
        ),
        "blocked_details": blocked,
    }


def run_external_agreement(
    protocol: CFValidationProtocol,
    *,
    replications: int | None = None,
    max_cells: int | None = None,
) -> dict[str, object]:
    partition = protocol.external
    if partition is None:
        raise ValueError("Protocol has no external-comparison seed namespace")
    if dependency_lock_checksum() != protocol.dependency_lock_checksum:
        raise ValueError("Validation dependency lock does not match the frozen protocol")
    environment = _environment()
    if environment != dict(protocol.software):
        raise ValueError("Installed external-comparison environment is not the frozen environment")
    count = partition.count if replications is None else replications
    if count < 1 or count > partition.count:
        raise ValueError("replications must lie within the frozen external lane")
    cells = protocol.external_cells[:max_cells]
    fixed_max = 0.0
    shared_errors = {
        "means": 0.0,
        "influence": 0.0,
        "covariance": 0.0,
        "standard_errors": 0.0,
        "contrasts": 0.0,
        "contrast_standard_errors": 0.0,
    }
    signed = {"DoubleMLAPOS": [], "EconML.DRLearner": []}
    blocked: dict[str, list[str]] = {"DoubleMLAPOS": [], "EconML.DRLearner": []}
    details: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        for repetition in range(count):
            seed = partition.start + cell_index * partition.count + repetition
            generated = simulate_reference_cell(cell, seed=seed)
            result = SCOVACF().analyze(
                generated.data,
                _declaration(generated, cell, include_stability=False),
            )
            if isinstance(result, SCOVACFRefusal):
                raise RuntimeError(f"SCOVA-CF refused external fixture: {result.status.code}")
            labels = result.group_labels
            treatment = np.array([labels.index(value) for value in generated.data["group"]])
            outcome = generated.data["outcome"].to_numpy()
            x = generated.data.loc[:, result.covariate_names].to_numpy()
            literal = fixed_nuisance_score(
                outcome,
                treatment,
                result.propensity_predictions,
                result.outcome_predictions,
            )
            shared = assemble_aipw(
                outcome,
                treatment,
                result.propensity_predictions,
                result.outcome_predictions,
            )
            fixed_max = max(
                fixed_max,
                *(
                    float(_maximum_error(left, right))
                    for left, right in zip(literal, shared, strict=True)
                ),
            )
            exact = doubleml_shared_score(
                x,
                outcome,
                treatment,
                result.fold_assignments,
                result.propensity_predictions,
                result.outcome_predictions,
            )
            if exact.status != "complete" or exact.influence is None or exact.covariance is None:
                blocked["DoubleMLAPOS"].append(
                    f"shared cell={cell_index} rep={repetition}: {exact.detail}"
                )
            else:
                shared_errors["means"] = max(
                    shared_errors["means"],
                    _maximum_error(np.asarray(exact.estimates), result.group_means),
                )
                shared_errors["influence"] = max(
                    shared_errors["influence"],
                    _maximum_error(exact.influence, result.influence_values),
                )
                shared_errors["covariance"] = max(
                    shared_errors["covariance"],
                    _maximum_error(exact.covariance, result.covariance),
                )
                shared_errors["standard_errors"] = max(
                    shared_errors["standard_errors"],
                    _maximum_error(
                        np.asarray(exact.standard_errors),
                        result.group_standard_errors,
                    ),
                )
                exact_means = np.asarray(exact.estimates)
                exact_contrasts = exact_means[1:] - exact_means[0]
                scova_contrasts = result.group_means[1:] - result.group_means[0]
                shared_errors["contrasts"] = max(
                    shared_errors["contrasts"],
                    _maximum_error(exact_contrasts, scova_contrasts),
                )
                contrast_weights = np.column_stack(
                    [
                        np.eye(len(labels))[code] - np.eye(len(labels))[0]
                        for code in range(1, len(labels))
                    ]
                )
                exact_contrast_se = np.sqrt(
                    np.diag(contrast_weights.T @ exact.covariance @ contrast_weights)
                )
                scova_contrast_se = np.array(
                    [
                        result.contrasts[f"g{code} - g0"].standard_error
                        for code in range(1, len(labels))
                    ]
                )
                shared_errors["contrast_standard_errors"] = max(
                    shared_errors["contrast_standard_errors"],
                    _maximum_error(exact_contrast_se, scova_contrast_se),
                )
            dml = doubleml_apos(
                x,
                outcome,
                treatment,
                result.fold_assignments,
                learner_policy=str(cell["learner"]),
            )
            econ = econml_drlearner(
                x,
                outcome,
                treatment,
                result.fold_assignments,
                learner_policy=str(cell["learner"]),
            )
            if dml.status == "complete":
                scale = np.where(
                    result.group_standard_errors > 0,
                    result.group_standard_errors,
                    np.nan,
                )
                signed["DoubleMLAPOS"].extend(
                    ((np.asarray(dml.estimates) - result.group_means) / scale).tolist()
                )
            else:
                blocked["DoubleMLAPOS"].append(
                    f"fitted cell={cell_index} rep={repetition}: {dml.detail}"
                )
            reference = result.group_means[1:] - result.group_means[0]
            reference_se = np.array(
                [result.contrasts[f"g{code} - g0"].standard_error for code in range(1, len(labels))]
            )
            if econ.status == "complete":
                signed["EconML.DRLearner"].extend(
                    ((np.asarray(econ.estimates) - reference) / reference_se).tolist()
                )
            else:
                blocked["EconML.DRLearner"].append(
                    f"fitted cell={cell_index} rep={repetition}: {econ.detail}"
                )
            details.append(
                {
                    "cell_index": cell_index,
                    "repetition": repetition,
                    "seed": seed,
                    "doubleml": dml.to_dict(),
                    "econml": econ.to_dict(),
                }
            )
    shared_passed = bool(
        fixed_max <= 1e-12
        and shared_errors["means"] <= 1e-10
        and shared_errors["influence"] <= 1e-10
        and shared_errors["covariance"] <= 1e-10
        and shared_errors["standard_errors"] <= 1e-10
        and shared_errors["contrasts"] <= 1e-10
        and shared_errors["contrast_standard_errors"] <= 1e-10
    )
    summaries = [
        _summary(name, signed[name], blocked[name])
        for name in ("DoubleMLAPOS", "EconML.DRLearner")
    ]
    complete = count == partition.count and len(cells) == len(protocol.external_cells)
    evidence: dict[str, Any] = {
        "artifact_type": "scova-cf-external-agreement",
        "schema_version": 2,
        "protocol_checksum": protocol.checksum,
        "dependency_lock_checksum": dependency_lock_checksum(),
        "environment": environment,
        "complete_frozen_lane": complete,
        "replications_per_cell": count,
        "cell_count": len(cells),
        "shared_score": {
            "literal_maximum_absolute_error": fixed_max,
            "doubleml_maximum_absolute_errors": shared_errors,
            "literal_tolerance": 1e-12,
            "doubleml_tolerance": 1e-10,
            "passed": shared_passed,
            "variance_convention": "DoubleML raw SE uses n; aligned comparison uses n-1",
        },
        "end_to_end": {
            "mean_absolute_tolerance_in_scova_se": 0.25,
            "maximum_absolute_tolerance_in_scova_se": 1.0,
            "mean_signed_tolerance_in_scova_se": 0.05,
            "implementations": summaries,
        },
        "run_details": details,
        "all_numerical_agreement_gates_passed": bool(
            complete
            and shared_passed
            and all(row["status"] == "complete" for row in summaries)
        ),
    }
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    evidence = run_external_agreement(
        CFValidationProtocol.load(args.spec),
        replications=args.replications,
        max_cells=args.max_cells,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if not evidence["all_numerical_agreement_gates_passed"]:
        raise SystemExit("External numerical agreement did not pass")


if __name__ == "__main__":
    main()
