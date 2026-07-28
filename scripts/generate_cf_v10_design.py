"""Generate the frozen SCOVA-CF v10 observational design and specification.

The v10 campaign validates the regime SCOVA-CF exists for and has never been
validated in: naturally occurring groups, where assignment depends on the
covariates and the propensity must be estimated.

Three things make it a new campaign rather than an amendment to v9:

* Every retained cell declares confounding, so every cell is observational.
  A randomized cell in this design would let randomization-supported evidence
  vouch for an assumption-dependent-causal profile.
* No evidence is reused. v9 inherited frozen external-agreement and
  simultaneous-inference evidence from v5/v6, but those runs are randomized and
  say nothing about estimated assignment. v10 generates all three lanes.
* The design is selected for pairwise coverage over a factor space that now
  includes confounding strength, confounding form, and overlap.

Run with --write to emit benchmarks/specs/cf_reference_v10.json.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from benchmarks.cf_reference_campaign import dependency_lock_checksum, plasmode_source_checksum

# "none" is deliberately absent: a cell without confounding is randomized, and
# this protocol may only contain cells its promoted profile can govern.
FACTORS: dict[str, tuple[Any, ...]] = {
    "allocation": ("balanced", "moderate", "rare"),
    "confounding": ("weak", "moderate", "strong"),
    "confounding_form": ("linear", "nonlinear"),
    "effect": ("null", "partial-null", "constant", "heterogeneous"),
    "learner": ("linear", "adaptive"),
    "n_covariates": (5, 20, 50),
    "n_groups": (2, 3, 5),
    "n_per_group": (20, 30, 50, 80, 150, 250),
    "noise": ("normal", "heteroskedastic", "heavy-tailed"),
    "overlap": ("full", "partial", "poor"),
    "support": ("strong", "weak", "structural-failure"),
    "surface": ("linear", "smooth-nonlinear", "threshold", "interaction", "weak"),
}

RETAINED_CELLS = 48
NAMES = tuple(FACTORS)

# A cell is infeasible when its smallest group is expected to be too small to
# fit at all. The estimator needs n_splits (3) observations per group, so a
# design point whose smallest arm is expected to hold ~3 units refuses on
# roughly half its draws and calibrates thresholds against noise rather than
# against the method.
#
# The binding quantity is n_per_group * n_groups * min(baseline propensity),
# and measuring it shows `rare` allocation is not the worst corner: at
# n_per_group 20 it expects 3.0-3.9 units in the smallest arm, while `weak`
# support expects 1.2-1.8 and overrides allocation entirely. Excluding only the
# rare corner would leave most degenerate cells in place.
#
# The floor is deliberately above the hard n_splits limit so that refusal stays
# possible where the design intends it -- `structural-failure` still collapses
# a group on purpose -- without the calibration lane being mostly refusals.
MINIMUM_EXPECTED_ARM = 10.0

# Plasmode source checksums hash the bundled scikit-learn data, so they are a
# property of the pinned numerical stack rather than of this repository. Every
# campaign tier installs benchmarks/requirements-cf-validation.txt, which pins
# scikit-learn 1.6.1; computing these on whatever the generator happens to have
# installed would freeze a spec that the campaign can never satisfy. These are
# the 1.6.1 values, identical to the ones the v9 spec was frozen with in CI.
PINNED_DATASET_CHECKSUMS = {
    "breast-cancer": "ba4d19cf8137014a7cbcc9f1d625891c176da5a3dbf2af07333081d81f57e90f",
    "diabetes": "28128d0ec207a1c0ac5e23a5fdbad720215a35b5f18741fef26c4ecd254dc278",
}
PINNED_SKLEARN = "1.6.1"


def dataset_checksums() -> dict[str, str]:
    """Pinned plasmode checksums, cross-checked when the stack actually matches."""
    import sklearn

    if sklearn.__version__ == PINNED_SKLEARN:
        computed = {name: plasmode_source_checksum(name) for name in PINNED_DATASET_CHECKSUMS}
        if computed != PINNED_DATASET_CHECKSUMS:
            raise SystemExit(
                "scikit-learn "
                f"{PINNED_SKLEARN} now yields {computed}, not the pinned values. "
                "The frozen datasets changed; do not regenerate the spec silently."
            )
    return dict(PINNED_DATASET_CHECKSUMS)


def expected_smallest_arm(cell: dict[str, Any]) -> float:
    from benchmarks.cf_reference_campaign import _probabilities

    baseline = _probabilities(
        int(cell["n_groups"]), str(cell["allocation"]), str(cell["support"])
    )
    return float(cell["n_per_group"]) * int(cell["n_groups"]) * float(baseline.min())


def is_feasible(cell: dict[str, Any]) -> bool:
    return expected_smallest_arm(cell) >= MINIMUM_EXPECTED_ARM


def _feasibility_subspace() -> list[dict[str, Any]]:
    """Every combination of the four factors feasibility depends on."""
    from itertools import product

    keys = ("allocation", "support", "n_groups", "n_per_group")
    return [
        dict(zip(keys, values, strict=True))
        for values in product(*(FACTORS[key] for key in keys))
    ]


def _all_pairs() -> set[tuple[str, Any, str, Any]]:
    """Pairs a feasible cell can actually realize.

    Coverage is scored against this rather than the full cross product, so an
    unreachable pair like (support=weak, n_per_group=20) is not counted as a
    coverage failure for a design that is right to avoid it.
    """
    constrained = {"allocation", "support", "n_groups", "n_per_group"}
    reachable = [cell for cell in _feasibility_subspace() if is_feasible(cell)]
    pairs = set()
    for left, right in combinations(NAMES, 2):
        for left_value in FACTORS[left]:
            for right_value in FACTORS[right]:
                candidate = {left: left_value, right: right_value}
                relevant = {k: v for k, v in candidate.items() if k in constrained}
                if relevant and not any(
                    all(cell[k] == v for k, v in relevant.items()) for cell in reachable
                ):
                    continue
                pairs.add((left, left_value, right, right_value))
    return pairs


def _covered_by(cell: dict[str, Any]) -> set[tuple[str, Any, str, Any]]:
    return {
        (left, cell[left], right, cell[right]) for left, right in combinations(NAMES, 2)
    }


def _sorted_pairs(pairs: set[tuple[str, Any, str, Any]]) -> list[tuple[str, Any, str, Any]]:
    """Deterministic order independent of set iteration order."""
    return sorted(pairs, key=lambda pair: (NAMES.index(pair[0]), str(pair[1]),
                                           NAMES.index(pair[2]), str(pair[3])))


def _grow_from_anchor(
    anchor: tuple[str, Any, str, Any], remaining: set[tuple[str, Any, str, Any]]
) -> dict[str, Any]:
    """Fix the anchor pair, then fill every other factor greedily.

    Anchoring on a still-uncovered pair guarantees each cell consumes at least
    one pair, which is what makes the construction terminate. Levels are
    considered in declared order and ties break to the earliest, so the design
    is a deterministic function of FACTORS alone -- no seed to record.
    """
    left_name, left_value, right_name, right_value = anchor
    cell: dict[str, Any] = {left_name: left_value, right_name: right_value}
    for name in NAMES:
        if name in cell:
            continue
        best_level, best_gain = FACTORS[name][0], -1
        for level in FACTORS[name]:
            gain = sum(
                1
                for assigned, value in cell.items()
                if (assigned, value, name, level) in remaining
                or (name, level, assigned, value) in remaining
            )
            if gain > best_gain:
                best_level, best_gain = level, gain
        cell[name] = best_level
    ordered = {name: cell[name] for name in NAMES}
    if not is_feasible(ordered):
        # Repair by raising n_per_group to the smallest feasible level. Every
        # (allocation, support, n_groups) triple has one, so this always
        # succeeds; the anchor is preserved unless it fixed n_per_group, and
        # such anchors are excluded from the pair universe above.
        for level in FACTORS["n_per_group"]:
            if is_feasible({**ordered, "n_per_group": level}):
                ordered["n_per_group"] = level
                break
    return ordered


def select_design() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy pairwise-covering construction; deterministic, no randomness."""
    total = len(_all_pairs())
    remaining = _all_pairs()
    every_pair = _sorted_pairs(_all_pairs())
    chosen: list[dict[str, Any]] = []
    filler = 0
    while len(chosen) < RETAINED_CELLS:
        if remaining:
            anchor = _sorted_pairs(remaining)[0]
        else:
            # Coverage is complete before the cell budget is spent. Keep
            # anchoring deterministically through the full pair list so the
            # remaining cells replicate distinct regions rather than repeat one.
            anchor = every_pair[filler % len(every_pair)]
            filler += 1
        cell = _grow_from_anchor(anchor, remaining)
        remaining -= _covered_by(cell)
        if cell not in chosen:
            chosen.append(cell)
    provenance = {
        "method": "v10-observational-greedy-pairwise-coverage-feasible",
        "candidate_order": "declared-factor-level-order",
        "retained_cells": RETAINED_CELLS,
        "tie_break": "earliest-declared-level",
        "pairwise_pairs_total": total,
        "pairwise_pairs_covered": total - len(remaining),
        "minimum_expected_smallest_arm": MINIMUM_EXPECTED_ARM,
    }
    return chosen, provenance


def _observational(cell: dict[str, Any]) -> dict[str, Any]:
    """Confounding keys every non-retained lane must also carry."""
    return {
        "confounding": cell["confounding"],
        "confounding_form": cell["confounding_form"],
        "overlap": cell["overlap"],
    }


def build_spec() -> dict[str, Any]:
    retained, provenance = select_design()
    # Plasmode, external and inference lanes are hand-fixed rather than
    # generated: they answer narrower questions and their cell counts are
    # pinned by the schema. All are observational and well-supported, because
    # agreement and simultaneous-inference claims are only meaningful where the
    # estimator is expected to work at all.
    plasmode_cells = [
        {
            "allocation": allocation,
            "confounding": confounding,
            "confounding_form": form,
            "dataset": dataset,
            "effect": effect,
            "learner": learner,
            "n_groups": groups,
            "n_per_group": per_group,
            "noise": "normal",
            "overlap": "full",
        }
        for dataset, groups, per_group, allocation, effect, learner, confounding, form in (
            ("diabetes", 2, 100, "balanced", "null", "linear", "weak", "linear"),
            ("diabetes", 2, 100, "balanced", "constant", "adaptive", "moderate", "nonlinear"),
            ("diabetes", 2, 150, "moderate", "constant", "adaptive", "strong", "nonlinear"),
            ("diabetes", 3, 100, "balanced", "heterogeneous", "adaptive", "moderate", "linear"),
            ("diabetes", 2, 150, "balanced", "constant", "linear", "moderate", "linear"),
            ("diabetes", 3, 100, "moderate", "null", "adaptive", "strong", "nonlinear"),
            ("breast-cancer", 2, 100, "balanced", "null", "linear", "weak", "nonlinear"),
            ("breast-cancer", 2, 100, "balanced", "constant", "adaptive", "moderate", "nonlinear"),
            ("breast-cancer", 2, 150, "moderate", "constant", "adaptive", "strong", "linear"),
            ("breast-cancer", 3, 100, "balanced", "heterogeneous", "adaptive", "moderate",
             "nonlinear"),
            ("breast-cancer", 2, 150, "balanced", "constant", "linear", "moderate", "nonlinear"),
            ("breast-cancer", 3, 100, "moderate", "null", "adaptive", "strong", "linear"),
        )
    ]
    external_cells = [
        {
            "allocation": "balanced",
            "confounding": confounding,
            "confounding_form": form,
            "effect": effect,
            "learner": learner,
            "n_covariates": 5,
            "n_groups": groups,
            "n_per_group": 250,
            "noise": "normal",
            "overlap": "full",
            "support": "strong",
            "surface": surface,
        }
        for groups, effect, learner, surface, confounding, form in (
            (2, "null", "linear", "linear", "weak", "linear"),
            (2, "constant", "linear", "linear", "moderate", "linear"),
            (2, "constant", "adaptive", "smooth-nonlinear", "moderate", "nonlinear"),
            (2, "constant", "adaptive", "smooth-nonlinear", "strong", "nonlinear"),
            (3, "null", "linear", "linear", "moderate", "linear"),
            (3, "constant", "adaptive", "smooth-nonlinear", "moderate", "nonlinear"),
            (3, "heterogeneous", "adaptive", "interaction", "strong", "nonlinear"),
            (2, "heterogeneous", "adaptive", "threshold", "strong", "linear"),
        )
    ]
    inference_cells = [
        {
            "cell": {
                "allocation": "balanced",
                "confounding": confounding,
                "confounding_form": form,
                "effect": effect,
                "learner": learner,
                "n_covariates": 5,
                "n_groups": groups,
                "n_per_group": per_group,
                "noise": "normal",
                "overlap": "full",
                "support": "strong",
                "surface": surface,
            }
        }
        for groups, per_group, effect, learner, surface, confounding, form in (
            (2, 80, "null", "linear", "linear", "weak", "linear"),
            (2, 150, "null", "adaptive", "smooth-nonlinear", "moderate", "nonlinear"),
            (2, 250, "partial-null", "adaptive", "smooth-nonlinear", "strong", "nonlinear"),
            (3, 80, "null", "linear", "linear", "moderate", "linear"),
            (3, 150, "partial-null", "adaptive", "smooth-nonlinear", "moderate", "nonlinear"),
            (3, 250, "partial-null", "adaptive", "interaction", "strong", "nonlinear"),
        )
    ]
    return {
        "protocol_id": "cf-observational-continuous-aipw-unnormalized-v10",
        "schema_version": 2,
        "frozen": True,
        "reference_profile": {
            "mode": "observational-causal",
            "assignment": "estimated",
            "outcome_type": "continuous",
            "estimator": "aipw-unnormalized",
            "estimand_id": "study-population-standardized-means",
            "independent_unit": "row",
            "minimum_group_count": 50,
            "maximum_group_count": 3,
        },
        "factors": {name: list(values) for name, values in FACTORS.items()},
        "retained_cells": retained,
        "plasmode_cells": plasmode_cells,
        "external_cells": external_cells,
        "inference_cells": inference_cells,
        "learners": ["linear", "adaptive"],
        # A namespace below 1e9; v3-v9 occupy 1.0e9 upward and the ceiling is
        # 2**32-1, so there is no room to extend above v9's inference lane.
        "seed_partitions": {
            "pilot": {"start": 500_000_000, "count": 20},
            "calibration": {"start": 500_100_000, "count": 2000},
            "validation": {"start": 500_400_000, "count": 2000},
            "external": {"start": 500_700_000, "count": 50},
            "inference": {"start": 500_800_000, "count": 2000},
        },
        "calibration_fit_fraction": 0.6,
        "calibration_enrichment_screening": True,
        "calibration_candidate_retention_fraction": 0.85,
        "calibration_screening": {
            "confidence_level": 0.95,
            "type_i_error": 0.05,
            "monte_carlo_standard_error_multiplier": 2,
            "maximum_standardized_bias": 0.15,
            "minimum_se_ratio": 0.8,
            "maximum_se_ratio": 1.25,
            "strong_support_minimum_expected_arm_count": 30.0,
        },
        "metrics": {
            "confidence_level": 0.95,
            "type_i_error": 0.05,
            "monte_carlo_standard_error_multiplier": 2,
            "maximum_standardized_bias": 0.1,
            "minimum_se_ratio": 0.9,
            "maximum_se_ratio": 1.1,
            "strong_support_minimum_expected_arm_count": 30.0,
            "minimum_strong_cell_pass_fraction": 0.8,
            "minimum_strong_replication_pass_fraction": 0.8,
            "minimum_unstable_absolute_enrichment": 0.05,
            "minimum_unstable_risk_ratio": 2.0,
            "unstable_risk_ratio_selection_confidence": 0.95,
            "coverage_family_wise_error": 0.05,
            "shared_score_tolerance": 1e-12,
            "doubleml_shared_tolerance": 1e-10,
            "end_to_end_mean_se_difference": 0.25,
            "end_to_end_max_se_difference": 1.0,
            "end_to_end_signed_se_difference": 0.05,
        },
        # calibrate_cf_support reads exactly two keys: a lower grid for
        # minimum_ess_ratio and ONE shared upper grid for every upper feature.
        # A per-metric mapping parses as valid JSON and passes every gate up to
        # and including 128 calibration shards, then dies on KeyError.
        "threshold_quantiles": {
            "minimum_ess_ratio": [0, 0.01, 0.025, 0.05, 0.1, 0.2],
            "upper_metrics": [0.8, 0.9, 0.95, 0.975, 0.99, 1],
        },
        "design_selection": provenance,
        "dataset_checksums": dataset_checksums(),
        "dependency_lock_checksum": dependency_lock_checksum(),
        "software": {"python": "3.12.13", "scikit-learn": "1.6.1", "numpy": "2.2.6"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/specs/cf_reference_v10.json")
    )
    args = parser.parse_args()
    spec = build_spec()
    provenance = spec["design_selection"]
    covered = provenance["pairwise_pairs_covered"]
    total = provenance["pairwise_pairs_total"]
    print(f"retained cells: {len(spec['retained_cells'])}")
    print(f"pairwise coverage: {covered}/{total} ({covered / total:.1%})")
    for name in FACTORS:
        used = {cell[name] for cell in spec["retained_cells"]}
        print(f"  {name:18} {len(used)}/{len(FACTORS[name])} levels used")
    if args.write:
        args.output.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
