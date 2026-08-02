"""Estimate where arm density stops supporting the SCOVA-CF reference profile.

The procedure is declared in the protocol's `boundary_estimation` block and is
fixed before any evidence exists. This module only executes what that block
says; every choice it could otherwise make quietly is read from the spec.

What is being estimated. v10 ASSERTED `minimum_arm_units_per_covariate = 10.0`
and then ran a lane that could not inform it -- every profile-eligible cell sat
at or above the bound and five sat exactly on it. A bound fitted where all the
data lies on one side of it is not fitted at all.

The unit of observation is the cell. Replications inside a cell share a design
point, so they sharpen that point's pass rate rather than locating the
boundary; pooling them as independent would inflate the effective sample by
roughly the replication count and produce an interval that means nothing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scova.cf import CFValidationProtocol
from scripts.calibrate_cf_support import _profile_scope, arm_density, expected_smallest_arm

CellKey = tuple[str, int]
Stratum = tuple[int, str]


def claimed_strata(protocol: CFValidationProtocol) -> list[Stratum]:
    """(n_groups, learner) combinations the reference profile claims to serve."""
    maximum = protocol.reference_profile.get("maximum_group_count")
    return [
        (int(groups), str(learner))
        for groups in protocol.factors["n_groups"]
        if maximum is None or int(groups) <= int(maximum)
        for learner in protocol.factors["learner"]
    ]


def boundary_support_set(protocol: CFValidationProtocol) -> list[dict[str, Any]]:
    """Cells admissible as evidence about the density boundary.

    Includes cells BELOW the declared bound, which are profile-ineligible by
    definition -- a boundary cannot be located from one side of itself.

    Excludes cells below the absolute arm-count floor. There the count term is
    what binds, so treating such a cell as a density observation would blame
    density for a failure caused by having too few units outright.
    """
    minimum, maximum, _bound = _profile_scope(protocol)
    rows: list[dict[str, Any]] = []
    sources = (
        ("simulated", protocol.retained_cells),
        ("plasmode", protocol.plasmode_cells),
    )
    for kind, cells in sources:
        for index, raw in enumerate(cells):
            cell = dict(raw)
            groups = int(cell["n_groups"])
            if maximum is not None and groups > int(maximum):
                continue
            if kind != "plasmode" and cell.get("support") != "strong":
                continue
            smallest = expected_smallest_arm(cell)
            if smallest is None or smallest < minimum:
                continue
            rows.append(
                {
                    "kind": kind,
                    "cell_index": index,
                    "stratum": (groups, str(cell["learner"])),
                    "density": float(arm_density(cell, kind)),
                }
            )
    return rows


def identifiability_report(protocol: CFValidationProtocol) -> dict[str, Any]:
    """Can the declared procedure be fitted on this design at all?

    Run before dispatching anything. A design that cannot identify the
    procedure it declares will still calibrate, still pass every gate, and
    still produce a boundary number that is an artifact of the parameterization
    rather than a measurement.
    """
    declared = protocol.boundary_estimation
    if declared is None:
        raise ValueError("Protocol declares no boundary_estimation procedure")
    bound = float(protocol.reference_profile["minimum_arm_units_per_covariate"])
    minimum_distinct = int(declared["minimum_distinct_densities_per_stratum"])
    per_parameter = int(declared["minimum_observations_per_parameter"])
    require_bracketing = bool(declared["require_bracketing_per_stratum"])

    rows = boundary_support_set(protocol)
    strata = claimed_strata(protocol)
    # One slope shared across strata, one intercept per stratum.
    parameters = 1 + len(strata)
    failures: list[str] = []
    detail: dict[str, Any] = {}
    for stratum in strata:
        densities = sorted({row["density"] for row in rows if row["stratum"] == stratum})
        below = [value for value in densities if value < bound]
        at_or_above = [value for value in densities if value >= bound]
        detail[f"k={stratum[0]},{stratum[1]}"] = {
            "cells": sum(1 for row in rows if row["stratum"] == stratum),
            "distinct_densities": len(densities),
            "below_declared_bound": len(below),
            "at_or_above_declared_bound": len(at_or_above),
        }
        if len(densities) < minimum_distinct:
            failures.append(
                f"stratum (n_groups={stratum[0]}, learner={stratum[1]}) has "
                f"{len(densities)} distinct densities, below the required {minimum_distinct}"
            )
        if require_bracketing and not (below and at_or_above):
            failures.append(
                f"stratum (n_groups={stratum[0]}, learner={stratum[1]}) does not bracket "
                f"the declared bound {bound} ({len(below)} below, {len(at_or_above)} at or above)"
            )
    if len(rows) < per_parameter * parameters:
        failures.append(
            f"{len(rows)} admissible cells against {parameters} parameters, below the "
            f"required {per_parameter} per parameter"
        )
    return {
        "declared_bound": bound,
        "parameters": parameters,
        "effective_observations": len(rows),
        "observations_per_parameter": round(len(rows) / parameters, 2),
        "strata": detail,
        "failures": failures,
        "identifiable": not failures,
    }


def _fit(rows: list[dict[str, Any]], passes: list[bool], strata: list[Stratum]):
    """Unpenalized logistic fit: shared slope on log10 density, stratum intercepts."""
    from sklearn.linear_model import LogisticRegression

    outcome = np.asarray(passes, dtype=int)
    if outcome.min() == outcome.max():
        return None
    design = np.zeros((len(rows), 1 + len(strata)), dtype=float)
    for position, row in enumerate(rows):
        design[position, 0] = math.log10(row["density"])
        design[position, 1 + strata.index(row["stratum"])] = 1.0
    model = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000)
    model.fit(design, outcome)
    coefficients = np.asarray(model.coef_, dtype=float).ravel()
    if not np.all(np.isfinite(coefficients)):
        return None
    return coefficients


def _boundaries(
    coefficients: np.ndarray, strata: list[Stratum], target: float
) -> dict[Stratum, float] | None:
    slope = float(coefficients[0])
    # A boundary is only meaningful if support IMPROVES with density. A
    # non-positive slope means the design saw no density effect, and inverting
    # it would report a number with the wrong sign of meaning.
    if not slope > 0:
        return None
    logit = math.log(target / (1 - target))
    return {
        stratum: float(10 ** ((logit - float(coefficients[1 + index])) / slope))
        for index, stratum in enumerate(strata)
    }


def estimate_boundary(
    protocol: CFValidationProtocol, outcomes: dict[CellKey, bool]
) -> dict[str, Any]:
    """Fit the declared procedure and report, never adopt.

    `adoption` is `report-only`: moving the profile's declared bound stays a
    human decision requiring a new freeze. A campaign that can rescope itself
    to whichever cells passed is the failure this protocol has already been
    burned by.
    """
    declared = protocol.boundary_estimation
    if declared is None:
        raise ValueError("Protocol declares no boundary_estimation procedure")
    if declared["adoption"] != "report-only":
        raise ValueError(f"Unsupported adoption rule {declared['adoption']!r}")
    report = identifiability_report(protocol)
    if not report["identifiable"]:
        return {"status": "refused/unidentifiable", "identifiability": report}

    target = float(protocol.metrics["minimum_strong_cell_pass_fraction"])
    strata = claimed_strata(protocol)
    rows = [
        row
        for row in boundary_support_set(protocol)
        if (row["kind"], row["cell_index"]) in outcomes
    ]
    missing = len(boundary_support_set(protocol)) - len(rows)
    if missing:
        return {
            "status": "refused/incomplete-outcomes",
            "missing_cells": missing,
            "identifiability": report,
        }
    passes = [bool(outcomes[(row["kind"], row["cell_index"])]) for row in rows]
    coefficients = _fit(rows, passes, strata)
    if coefficients is None:
        return {"status": "refused/degenerate-fit", "identifiability": report}
    point = _boundaries(coefficients, strata, target)
    if point is None:
        return {"status": "refused/non-positive-density-effect", "identifiability": report}

    rng = np.random.default_rng(int(declared["bootstrap_seed"]))
    draws: dict[Stratum, list[float]] = {stratum: [] for stratum in strata}
    resamples = int(declared["bootstrap_resamples"])
    for _ in range(resamples):
        index = rng.integers(0, len(rows), size=len(rows))
        sampled = [rows[position] for position in index]
        sampled_passes = [passes[position] for position in index]
        if {row["stratum"] for row in sampled} != set(strata):
            continue
        fitted = _fit(sampled, sampled_passes, strata)
        if fitted is None:
            continue
        values = _boundaries(fitted, strata, target)
        if values is None:
            continue
        for stratum, value in values.items():
            draws[stratum].append(value)
    return {
        "status": "complete",
        "adoption": "report-only",
        "declared_bound": float(protocol.reference_profile["minimum_arm_units_per_covariate"]),
        "pass_probability_target": target,
        "log10_density_slope": float(coefficients[0]),
        "identifiability": report,
        "strata": {
            f"k={stratum[0]},{stratum[1]}": {
                "estimated_boundary": point[stratum],
                "bootstrap_resamples_used": len(draws[stratum]),
                "interval": (
                    None
                    if len(draws[stratum]) < resamples // 2
                    else [
                        float(np.percentile(draws[stratum], 2.5)),
                        float(np.percentile(draws[stratum], 97.5)),
                    ]
                ),
            }
            for stratum in strata
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = CFValidationProtocol.load(args.spec)
    report = identifiability_report(protocol)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not report["identifiable"]:
        raise SystemExit(f"{len(report['failures'])} identifiability failure(s)")


if __name__ == "__main__":
    main()
