"""Aggregate campaign shards and evaluate preregistered pass criteria."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import beta


def _upper_binomial(successes: int, total: int) -> float:
    if successes == total:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, total - successes))


def summarize(paths: list[Path], specification: dict) -> dict:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    spec_hashes = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_hashes.add(payload["specification_sha256"])
        for record in payload["records"]:
            key = json.dumps(record["spec"], sort_keys=True)
            by_cell[key].append(record)
    if len(spec_hashes) != 1:
        raise ValueError("campaign shards do not share one specification hash")
    criteria = specification["pass_criteria"]
    cells = []
    all_passed = True
    for key, records in sorted(by_cell.items()):
        accepted = [record for record in records if not record["alternative"]["refused"]]
        null_accepted = [record for record in records if not record["null"]["refused"]]
        execution_failures = sum(
            "execution_error" in record["alternative"] or "execution_error" in record["null"]
            for record in records
        )
        coverage = (
            np.mean([record["alternative"]["uniform_coverage"] for record in accepted])
            if accepted
            else np.nan
        )
        false_signs = sum(record["null"]["false_sign_certificate"] for record in null_accepted)
        fwer_upper = _upper_binomial(false_signs, len(null_accepted)) if null_accepted else np.nan
        errors = np.array(
            [record["alternative"]["scientific_target_mean_error"] for record in accepted]
        )
        standardized_bias = (
            abs(float(errors.mean())) / float(errors.std(ddof=1))
            if len(errors) > 1 and errors.std(ddof=1) > 0
            else np.nan
        )
        passed = bool(
            accepted
            and execution_failures == 0
            and criteria["simultaneous_coverage_min"]
            <= coverage
            <= criteria["simultaneous_coverage_max"]
            and fwer_upper <= criteria["fwer_upper_bound_max"]
            and standardized_bias <= criteria["standardized_absolute_bias_max"]
        )
        all_passed &= passed
        cells.append(
            {
                "spec": json.loads(key),
                "total": len(records),
                "accepted": len(accepted),
                "execution_failures": execution_failures,
                "refusal_rate": 1 - len(accepted) / len(records),
                "simultaneous_coverage": coverage,
                "false_sign_count": false_signs,
                "fwer_upper_95": fwer_upper,
                "standardized_absolute_bias": standardized_bias,
                "passed": passed,
            }
        )
    result = {
        "schema_version": 1,
        "specification_sha256": next(iter(spec_hashes)),
        "all_cells_passed": all_passed,
        "cells": cells,
    }
    encoded = json.dumps(result, sort_keys=True, allow_nan=True).encode()
    result["summary_sha256"] = sha256(encoded).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/specs/stage3_release.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    result = summarize(args.inputs, specification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
