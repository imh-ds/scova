"""Render a reproducible v11 coverage inventory for the applicability matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scova.cf import (
    ApplicabilityClassification,
    CFValidationProtocol,
    observational_applicability_matrix,
)
from scripts.calibrate_cf_support import _profile_eligible


def _counts(cells: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [cell.get(field, "not-declared") for cell in cells]
    return {str(value): count for value, count in sorted(Counter(values).items())}


def coverage_inventory(protocol: CFValidationProtocol) -> dict[str, Any]:
    """Summarize cells against the matrix without treating it as pass/fail evidence."""
    matrix = observational_applicability_matrix()
    if protocol.protocol_id != matrix.source_protocol:
        raise ValueError("Applicability matrix is bound to a different frozen protocol")
    simulated = [dict(cell) for cell in protocol.retained_cells]
    plasmode = [dict(cell) for cell in protocol.plasmode_cells]
    eligible_simulated = [
        cell for cell in simulated if _profile_eligible(protocol, cell, "simulated")
    ]
    eligible_plasmode = [
        cell for cell in plasmode if _profile_eligible(protocol, cell, "plasmode")
    ]
    return {
        "matrix_id": matrix.matrix_id,
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "candidate_qualification_envelope": {
            "n_groups": list(matrix.group_counts),
            "maximum_covariate_count": matrix.maximum_covariate_count,
            "nuisance_strategy": matrix.nuisance_strategy,
        },
        "simulated_cells_by_covariate_count": _counts(simulated, "n_covariates"),
        "simulated_cells_by_group_count": _counts(simulated, "n_groups"),
        "profile_eligible_simulated_by_covariate_count": _counts(
            eligible_simulated, "n_covariates"
        ),
        "profile_eligible_simulated_by_group_count": _counts(eligible_simulated, "n_groups"),
        "profile_eligible_plasmode_by_covariate_count": _counts(
            eligible_plasmode, "n_covariates"
        ),
        "known_limitations": [
            entry["entry_id"]
            for entry in matrix.classifications[ApplicabilityClassification.KNOWN_LIMITATION]
        ],
        "untested": [
            entry["entry_id"]
            for entry in matrix.classifications[ApplicabilityClassification.UNTESTED]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = coverage_inventory(CFValidationProtocol.load(args.spec))
    rendered = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
