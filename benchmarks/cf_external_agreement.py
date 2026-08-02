"""Generate frozen shared-score and end-to-end external-agreement evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import replace
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
    _git_commit,
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
from scripts.calibrate_cf_support import _family_wise_multiplier

# An end-to-end cell is only evidence if the outside implementation is free to
# disagree. When every difference in a cell sits at the precision the SHARED
# score lane already certifies as identity, that cell is not an independent
# comparison: it is `shared_score` recomputed through a second API, and it can
# no longer detect anything that lane does not already detect.
#
# Set to the shared lane's own DoubleML tolerance, which is the honest reading
# of "indistinguishable from recomputing SCOVA". Nine orders of magnitude
# separate the two regimes in practice -- r9 measured degenerate cells at
# 2.5e-15 to 2.3e-14 and the weakest informative cell at 1.1e-1 -- so the exact
# value is not delicate.
DEGENERATE_DIFFERENCE_IN_SCOVA_SE = 1e-10

# Used only when a protocol declares no family-wise budget. Two-sided 5% on a
# single test; the declared budget supersedes it for every protocol that has
# one.
_BASE_OFFSET_Z = 1.959963984540054

# What a non-authoritative `--allow-incomplete` smoke run may end on. The
# authoritative gate is `all_numerical_agreement_gates_passed`, which requires
# the complete frozen lane and "complete" from both implementations.
SMOKE_ADMISSIBLE_STATUSES = frozenset(
    {"complete", "incomplete/degenerate-subset", "incomplete/unscored-subset"}
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


def _agreement_policy(protocol: CFValidationProtocol) -> dict[str, Any]:
    """Read the preregistered external-agreement policy, or the v3-v10 default.

    Protocols without the block keep shared folds and the raw SE-difference
    tolerances they were frozen with; nothing about their evidence moves.
    """
    declared = getattr(protocol, "external_agreement", None)
    if declared is None:
        return {
            "comparator_folds": "scova",
            "statistic": "absolute-difference-in-scova-se",
            "minimum_informative_cell_fraction": 0.0,
        }
    return dict(declared)


def _comparator_folds(
    data: Any, declaration: Any, group_codes: np.ndarray, agreement: Mapping[str, Any]
) -> np.ndarray:
    """Fold assignments for the comparators.

    Built with SCOVA-CF's own fold construction under a different seed rather
    than a second splitter: the same group-stratified guarantee has to hold, or
    a comparator can be handed a training fold missing an arm. The seed salt is
    run through a SplitMix64 avalanche precisely so a changed seed reorders
    everything rather than perturbing a few ties.
    """
    if agreement.get("comparator_folds") != "independent":
        raise ValueError("comparator folds requested without an independent-fold policy")
    offset = int(agreement["comparator_fold_seed_offset"])
    shifted = replace(declaration, random_state=int(declaration.random_state) + offset)
    folds, _stratified = SCOVACF._design_folds(data, shifted, group_codes)
    return folds


def _record(
    cell: Mapping[str, Any], cell_index: int, repetition: int, differences: list[float]
) -> dict[str, Any]:
    return {
        "cell_index": cell_index,
        "repetition": repetition,
        "stratum": f"k={int(cell['n_groups'])},{cell['learner']}",
        "differences": [float(value) for value in differences],
    }


def _offset_z(unit_means: list[float]) -> float | None:
    """Standardized offset between two implementations, or None if unestimable.

    Under independent folds the comparators no longer share SCOVA's sample
    splits, so an individual difference carries fold noise even when both
    implementations are exactly right. Scoring |difference| against SCOVA's
    standard error therefore measures the split, not the software: fold-induced
    scatter alone was measured at a pooled mean |d| of 0.6324 against a 0.25
    tolerance and a maximum of 6.70 against 1.0.

    What survives that noise is a SYSTEMATIC offset. Random fold noise averages
    to zero across replications; a genuine implementation difference does not.
    So the statistic is the mean unit difference over its own standard error,
    and the same measurement that breached the raw tolerances kept this within
    +/-1.5 across every stratum.
    """
    values = np.asarray(unit_means, dtype=float)
    if values.size < 2:
        return None
    deviation = float(values.std(ddof=1))
    # Zero observed spread gives no usable denominator -- the float residue of
    # identical values would otherwise divide out to a z of ~1e16. Returning
    # None fails closed: the caller counts an unestimable stratum as breaching,
    # which is right either way, since identical units mean either a systematic
    # offset with no way to size it or a harness feeding the same numbers twice.
    if deviation <= 1e-12 * max(1.0, float(np.abs(values).max())):
        return None
    return float(values.mean() / (deviation / math.sqrt(values.size)))


def _summary(
    name: str,
    records: list[dict[str, Any]],
    blocked: list[str],
    *,
    lane_complete: bool,
    critical_z: float,
    minimum_informative_fraction: float,
) -> dict[str, Any]:
    """Score an implementation by its systematic offset, per stratum.

    The comparators no longer share SCOVA's folds, so agreement is a claim
    about the two implementations rather than about their arithmetic --
    `shared_score` already certifies the arithmetic at 1e-13 and is the place
    for that. What independent folds cost is the raw difference: fold-induced
    scatter alone breaches the old tolerances. What they buy is that a
    difference means something. See `_offset_z`.

    Degeneracy detection is retained even though independent folds should make
    it unreachable. That is the point of keeping it: under different splits
    there is no legitimate route to identity, so a degenerate cell now means the
    independence did not take effect, which is exactly the silent-harness
    failure this lane keeps producing. Hence a required informative FRACTION
    rather than merely "at least one".

    A truncated lane is still not judged on informativeness, because that is a
    property of the frozen cell set and not of an arbitrary prefix of it. That
    keeps `external_smoke` -- one replication of one cell -- working as the
    two-minute canary it exists to be.
    """
    per_cell: list[dict[str, Any]] = []
    by_cell: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_cell.setdefault(int(record["cell_index"]), []).append(record)
    degenerate_cells: set[int] = set()
    for cell_index in sorted(by_cell):
        flat = [
            abs(value) for record in by_cell[cell_index] for value in record["differences"]
        ]
        cell_maximum = max(flat) if flat else 0.0
        degenerate = cell_maximum <= DEGENERATE_DIFFERENCE_IN_SCOVA_SE
        if degenerate:
            degenerate_cells.add(cell_index)
        per_cell.append(
            {
                "cell_index": cell_index,
                "comparison_count": len(flat),
                "maximum_absolute_difference_in_scova_se": cell_maximum,
                "degenerate": degenerate,
            }
        )
    informative_cells = len(by_cell) - len(degenerate_cells)
    fraction = informative_cells / len(by_cell) if by_cell else 0.0

    scored = [
        record for record in records if int(record["cell_index"]) not in degenerate_cells
    ]
    by_stratum: dict[str, list[float]] = {}
    for record in scored:
        # One value per (cell, repetition): the differences inside a replication
        # share data and nuisances, so they are one observation, not several.
        by_stratum.setdefault(str(record["stratum"]), []).append(
            float(np.mean(record["differences"]))
        )
    strata = {
        name: {
            "unit_count": len(values),
            "mean_difference_in_scova_se": float(np.mean(values)),
            "offset_z": _offset_z(values),
        }
        for name, values in sorted(by_stratum.items())
    }
    # Two different failures, and conflating them broke the smoke tier. A
    # stratum whose offset is estimable and too large is a real signal on any
    # lane. A stratum with too few units to estimate an offset at all is a
    # statement about the run's size: on the complete lane that means no
    # observed spread and must fail closed, but `external_smoke` deliberately
    # runs ONE replication, so every stratum is unestimable by construction and
    # a combined check refuses the canary every time.
    breaching = sorted(
        name
        for name, values in strata.items()
        if values["offset_z"] is not None and abs(values["offset_z"]) > critical_z
    )
    unestimable = sorted(
        name for name, values in strata.items() if values["offset_z"] is None
    )

    if not records:
        status = "blocked/agreement-tolerance"
    elif fraction < minimum_informative_fraction:
        status = (
            "blocked/lane-degenerate" if lane_complete else "incomplete/degenerate-subset"
        )
    elif blocked or breaching:
        status = "blocked/agreement-tolerance"
    elif unestimable:
        status = (
            "blocked/agreement-tolerance" if lane_complete else "incomplete/unscored-subset"
        )
    else:
        status = "complete"
    return {
        "implementation": name,
        "status": status,
        "statistic": "standardized-offset-z",
        "critical_offset_z": critical_z,
        "maximum_absolute_offset_z": (
            None
            if not strata or any(row["offset_z"] is None for row in strata.values())
            else max(abs(row["offset_z"]) for row in strata.values())
        ),
        "breaching_strata": breaching,
        "unestimable_strata": unestimable,
        "strata": strata,
        # Everything below describes the denominator the figures above were
        # computed on, so a reader can tell a lane that agreed from a lane that
        # could not disagree without re-deriving it from run_details.
        "scored_unit_count": len(scored),
        "total_unit_count": len(records),
        "informative_cell_count": informative_cells,
        "degenerate_cell_count": len(degenerate_cells),
        "informative_cell_fraction": fraction,
        "minimum_informative_cell_fraction": minimum_informative_fraction,
        "cells": per_cell,
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
    # A known constant propensity is a property of the DESIGN, not of the
    # harness. v3-v9 randomize, so the design probabilities are the truth and
    # supplying them keeps the comparison to the outcome nuisance alone. v10
    # assigns on covariates, where `generated.probabilities` is only the
    # marginal allocation -- passing it would hand both comparators a flat,
    # misspecified propensity on confounded data and compare SCOVA-CF's
    # adjusted estimate against two effectively unadjusted ones.
    known_design = str(protocol.reference_profile.get("assignment")) == "known-constant"
    agreement = _agreement_policy(protocol)
    fixed_max = 0.0
    shared_errors = {
        "means": 0.0,
        "influence": 0.0,
        "covariance": 0.0,
        "standard_errors": 0.0,
        "contrasts": 0.0,
        "contrast_standard_errors": 0.0,
    }
    # One record per (implementation, cell, repetition). That tuple is the
    # independent unit: differences inside a replication share the same data and
    # the same fitted nuisances, so they are not separate observations of the
    # offset between two implementations. Everything the gate needs -- per-cell
    # degeneracy and the per-stratum offset -- is derived from these.
    records: dict[str, list[dict[str, Any]]] = {"DoubleMLAPOS": [], "EconML.DRLearner": []}
    blocked: dict[str, list[str]] = {"DoubleMLAPOS": [], "EconML.DRLearner": []}
    details: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        for repetition in range(count):
            seed = partition.start + cell_index * partition.count + repetition
            generated = simulate_reference_cell(cell, seed=seed)
            declaration = _declaration(generated, cell, include_stability=False)
            result = SCOVACF().analyze(generated.data, declaration)
            if isinstance(result, SCOVACFRefusal):
                raise RuntimeError(f"SCOVA-CF refused external fixture: {result.status.code}")
            labels = result.group_labels
            treatment = np.array([labels.index(value) for value in generated.data["group"]])
            comparator_folds = (
                _comparator_folds(generated.data, declaration, treatment, agreement)
                if agreement["comparator_folds"] == "independent"
                # v3-v10 froze their evidence with shared folds. Changing what
                # they compare would silently redefine agreement for protocols
                # that are already sealed.
                else result.fold_assignments
            )
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
            # The comparators get their OWN folds. Handed SCOVA's splits they
            # reproduce SCOVA's nuisances exactly -- both fit the declared
            # learner family, and since v11 both fit the propensity one arm at a
            # time -- so every cell collapsed to identity and the lane could
            # only ever report agreement with itself. `shared_score` above is
            # where the arithmetic is checked, on SCOVA's folds and SCOVA's
            # nuisances, and it is unaffected by this.
            dml = doubleml_apos(
                x,
                outcome,
                treatment,
                comparator_folds,
                learner_policy=str(cell["learner"]),
                known_probabilities=generated.probabilities if known_design else None,
            )
            econ = econml_drlearner(
                x,
                outcome,
                treatment,
                comparator_folds,
                learner_policy=str(cell["learner"]),
                known_probabilities=generated.probabilities if known_design else None,
            )
            if dml.status == "complete":
                scale = np.where(
                    result.group_standard_errors > 0,
                    result.group_standard_errors,
                    np.nan,
                )
                differences = (
                    (np.asarray(dml.estimates) - result.group_means) / scale
                ).tolist()
                records["DoubleMLAPOS"].append(
                    _record(cell, cell_index, repetition, differences)
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
                differences = (
                    (np.asarray(econ.estimates) - reference) / reference_se
                ).tolist()
                records["EconML.DRLearner"].append(
                    _record(cell, cell_index, repetition, differences)
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
    complete = count == partition.count and len(cells) == len(protocol.external_cells)
    # One test per (implementation, stratum). Scoring each against an
    # uncorrected 5% would inflate the family-wise error with the number of
    # strata -- the same uncorrected multiplicity that made the v8 per-cell
    # coverage gate unusable.
    family_size = sum(
        len({record["stratum"] for record in records[name]})
        for name in ("DoubleMLAPOS", "EconML.DRLearner")
    )
    critical_z = _family_wise_multiplier(
        agreement.get("family_wise_error"), max(family_size, 1), _BASE_OFFSET_Z
    )
    summaries = [
        _summary(
            name,
            records[name],
            blocked[name],
            lane_complete=complete,
            critical_z=critical_z,
            minimum_informative_fraction=float(
                agreement.get("minimum_informative_cell_fraction", 0.0)
            ),
        )
        for name in ("DoubleMLAPOS", "EconML.DRLearner")
    ]
    evidence: dict[str, Any] = {
        "artifact_type": "scova-cf-external-agreement",
        "schema_version": 3,
        "protocol_checksum": protocol.checksum,
        "git_commit": _git_commit(),
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
            "degenerate_difference_in_scova_se": DEGENERATE_DIFFERENCE_IN_SCOVA_SE,
            "scored_on": "cells-that-are-not-degenerate",
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
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a non-authoritative smoke artifact without requiring the full frozen lane.",
    )
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
    # A truncated smoke run checks that the lane executes, not that it agrees
    # informatively -- see `_summary`. It still requires shared_score, which is
    # the arithmetic check and is never degenerate by design.
    partial_agreement_passed = bool(
        evidence["shared_score"]["passed"]
        and all(
            row["status"] in SMOKE_ADMISSIBLE_STATUSES
            for row in evidence["end_to_end"]["implementations"]
        )
    )
    if not evidence["all_numerical_agreement_gates_passed"] and not (
        args.allow_incomplete and partial_agreement_passed
    ):
        raise SystemExit("External numerical agreement did not pass")


if __name__ == "__main__":
    main()
