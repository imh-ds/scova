"""Run an independent, diagnostic audit of one SCOVA-CF inference cell."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.cf_reference_campaign import (
    _contrast_summary,
    _declaration,
    dependency_lock_checksum,
    simulate_reference_cell,
    write_deterministic_gzip,
)
from scova.cf import SCOVACF, CFValidationProtocol, SCOVACFRefusal, canonical_checksum

DEFAULT_CELL_INDEX = 4
DEFAULT_SEED_START = 4_500_000_000
DEFAULT_REPLICATIONS = 2_048
DEFAULT_BOOTSTRAP_REPLICATIONS = 999
CONFIDENCE_LEVEL = 0.95


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def inference_cell(protocol: CFValidationProtocol, cell_index: int) -> dict[str, Any]:
    """Return the simulation cell addressed by a frozen inference cell."""
    if protocol.inference is None:
        raise ValueError("The protocol does not declare an inference lane")
    if not 0 <= cell_index < len(protocol.inference_cells):
        raise ValueError(
            f"cell_index must lie in [0, {len(protocol.inference_cells) - 1}]"
        )
    reference = protocol.inference_cells[cell_index]
    simulation_cell_index = reference.get("simulation_cell_index")
    if simulation_cell_index is None:
        cell = reference.get("cell")
    else:
        cell = protocol.retained_cells[int(simulation_cell_index)]
    if not isinstance(cell, dict):
        raise ValueError("The selected inference cell does not contain a simulation cell")
    return dict(cell)


def audit_record(
    *,
    cell: dict[str, Any],
    seed: int,
    bootstrap_replications: int,
) -> dict[str, Any]:
    """Run one replication while retaining the interval diagnostics."""
    generated = simulate_reference_cell(cell, seed=seed)
    result = SCOVACF().analyze(
        generated.data,
        _declaration(generated, cell, include_stability=False),
    )
    if isinstance(result, SCOVACFRefusal):
        return {
            "seed": seed,
            "refused": True,
            "status_code": result.status.code,
            "reason": result.status.reason,
        }

    contrasts = _contrast_summary(
        result.group_means,
        result.covariance,
        generated.true_group_means,
    )
    inference = result.infer(
        confidence_level=CONFIDENCE_LEVEL,
        n_bootstrap=bootstrap_replications,
        random_state=seed + 7_919,
    ).core
    truths = np.asarray([item["truth"] for item in contrasts], dtype=float)
    intervals = np.asarray(
        [item.simultaneous_confidence_interval for item in inference.contrasts],
        dtype=float,
    )
    return {
        "seed": seed,
        "refused": False,
        "status_code": result.status.code,
        "contrasts": contrasts,
        "simultaneous": {
            "critical_value": float(inference.critical_value),
            "covered_family": bool(
                np.all((intervals[:, 0] <= truths) & (truths <= intervals[:, 1]))
            ),
            "any_null_rejected": bool(
                any(
                    contrast["null"] and value.adjusted_p_value < 0.05
                    for contrast, value in zip(
                        contrasts,
                        inference.contrasts,
                        strict=True,
                    )
                )
            ),
            "max_t_p_value": float(inference.global_test.max_t_p_value),
            "adjusted_p_values": [
                float(value.adjusted_p_value) for value in inference.contrasts
            ],
        },
    }


def _critical_value_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    values = np.asarray(
        [record["simultaneous"]["critical_value"] for record in records],
        dtype=float,
    )
    if values.size == 0:
        return {key: None for key in ("minimum", "mean", "median", "p95", "maximum")}
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95, method="higher")),
        "maximum": float(np.max(values)),
    }


def _standard_error_calibration(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    summaries: list[dict[str, float | None]] = []
    for contrast_index in range(len(records[0]["contrasts"])):
        errors = np.asarray(
            [
                record["contrasts"][contrast_index]["estimate"]
                - record["contrasts"][contrast_index]["truth"]
                for record in records
            ],
            dtype=float,
        )
        standard_errors = np.asarray(
            [record["contrasts"][contrast_index]["standard_error"] for record in records],
            dtype=float,
        )
        bias = float(np.mean(errors))
        empirical_sd = None if len(errors) < 2 else float(np.std(errors, ddof=1))
        mean_standard_error = float(np.mean(standard_errors))
        summaries.append(
            {
                "contrast_index": contrast_index,
                "bias": bias,
                "empirical_sd": empirical_sd,
                "mean_standard_error": mean_standard_error,
                "sd_over_mean_standard_error": (
                    None
                    if empirical_sd is None or mean_standard_error == 0
                    else empirical_sd / mean_standard_error
                ),
            }
        )
    return summaries


def summarize_audit(
    protocol: CFValidationProtocol,
    *,
    cell_index: int,
    cell: dict[str, Any],
    seed_start: int,
    expected_replications: int,
    bootstrap_replications: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize an independent audit without turning it into release evidence."""
    valid_records = [record for record in records if not record.get("refused", False)]
    refusal_count = len(records) - len(valid_records)
    alpha = float(protocol.metrics["coverage_family_wise_error"])
    target = 1.0 - alpha
    multiplier = float(protocol.metrics["monte_carlo_standard_error_multiplier"])
    mcse = float(np.sqrt(alpha * (1.0 - alpha) / expected_replications))
    simultaneous_coverage = (
        None
        if not valid_records
        else float(
            np.mean(
                [record["simultaneous"]["covered_family"] for record in valid_records]
            )
        )
    )
    pointwise_coverage = (
        []
        if not valid_records
        else [
            float(
                np.mean(
                    [
                        record["contrasts"][contrast_index]["covered"]
                        for record in valid_records
                    ]
                )
            )
            for contrast_index in range(len(valid_records[0]["contrasts"]))
        ]
    )
    familywise_error = (
        None
        if not valid_records
        else float(
            np.mean(
                [record["simultaneous"]["any_null_rejected"] for record in valid_records]
            )
        )
    )
    all_replications_completed = (
        len(records) == expected_replications and refusal_count == 0
    )
    report = {
        "artifact_type": "scova-cf-inference-cell-audit",
        "schema_version": 1,
        "audit_role": "independent-diagnostic-only",
        "protocol_checksum": protocol.checksum,
        "git_commit": _commit(),
        "dependency_lock_checksum": dependency_lock_checksum(),
        "environment": {
            "python": platform.python_version(),
            "scova": _version("scova"),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "scikit-learn": _version("scikit-learn"),
            "platform": platform.platform(),
        },
        "cell_index": cell_index,
        "cell": cell,
        "seed_start": seed_start,
        "seed_end": seed_start + expected_replications - 1,
        "expected_replications": expected_replications,
        "completed_replications": len(valid_records),
        "refusal_count": refusal_count,
        "all_replications_completed": all_replications_completed,
        "bootstrap_replications": bootstrap_replications,
        "confidence_level": CONFIDENCE_LEVEL,
        "pointwise_coverage": pointwise_coverage,
        "simultaneous_coverage": simultaneous_coverage,
        "familywise_error": familywise_error,
        "critical_value_summary": _critical_value_summary(valid_records),
        "standard_error_calibration": _standard_error_calibration(valid_records),
        "coverage_gate": {
            "target": target,
            "monte_carlo_standard_error": mcse,
            "multiplier": multiplier,
            "lower_bound": target - multiplier * mcse,
            "passed": bool(
                all_replications_completed
                and simultaneous_coverage is not None
                and simultaneous_coverage >= target - multiplier * mcse
            ),
        },
        "records": records,
    }
    report["evidence_checksum"] = canonical_checksum(report)
    return report


def run_audit(
    protocol: CFValidationProtocol,
    *,
    cell_index: int = DEFAULT_CELL_INDEX,
    seed_start: int = DEFAULT_SEED_START,
    replications: int = DEFAULT_REPLICATIONS,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
) -> dict[str, Any]:
    """Run the independent cell audit with an explicit non-release seed range."""
    if seed_start < 0:
        raise ValueError("seed_start must be nonnegative")
    if replications < 1:
        raise ValueError("replications must be positive")
    if bootstrap_replications < 1:
        raise ValueError("bootstrap_replications must be positive")
    cell = inference_cell(protocol, cell_index)
    records = [
        audit_record(
            cell=cell,
            seed=seed_start + repetition,
            bootstrap_replications=bootstrap_replications,
        )
        for repetition in range(replications)
    ]
    return summarize_audit(
        protocol,
        cell_index=cell_index,
        cell=cell,
        seed_start=seed_start,
        expected_replications=replications,
        bootstrap_replications=bootstrap_replications,
        records=records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, default=DEFAULT_CELL_INDEX)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument(
        "--bootstrap-replications",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATIONS,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        CFValidationProtocol.load(args.spec),
        cell_index=args.cell_index,
        seed_start=args.seed_start,
        replications=args.replications,
        bootstrap_replications=args.bootstrap_replications,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_gzip(
        args.output,
        json.dumps(report, indent=2, sort_keys=True),
    )
    digest = sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        digest + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
