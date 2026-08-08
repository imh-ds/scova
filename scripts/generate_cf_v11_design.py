"""Generate the SCOVA-CF v11 observational design and specification.

v11 exists because the propensity parameterization changed. `src/scova/estimator.py`
is a core numerical path, so fitting the propensity one arm at a time moves every
evidence fingerprint and no v10 lane carries over.

Since the grid has to be rebuilt anyway, it is rebuilt to fix what v10's could not
express. v10 selected purely for pairwise coverage, and profile eligibility is a
CONJUNCTION -- strong support, at most three arms, a smallest arm of at least 50,
and at least 10 units per covariate in it. Pairwise coverage optimizes marginals,
so eligible cells only ever arose by accident: 11 of 60 cells, distributed
9 / 1 / 0 / 1 across (n_groups, learner). Eligibility decides the calibration
denominator, so the empty (k=3, linear) stratum meant no gate in the campaign could
detect a multi-arm defect under a linear learner. The external lane found one
immediately.

The fix is not a different search. It is reserving the region first and spending
what is left on coverage, which is what MANDATORY_CELLS does. The factor space is
UNCHANGED from v10 -- no factor is collapsed, and pairwise coverage stays complete.

Run with --write to emit benchmarks/specs/cf_reference_v11.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.generate_cf_v10_design import (
    build_spec as build_v10_spec,
)
from scripts.generate_cf_v10_design import (
    design_coverage_failures,
    eligible_cells_by_stratum,
    expected_smallest_arm,
    select_design,
)

PROTOCOL_ID = "cf-observational-continuous-aipw-unnormalized-v11"
SELECTION_METHOD = "v11-observational-reserved-eligibility-then-pairwise-coverage"

# Pre-specified before any v11 evidence exists, which is the only time a
# procedure like this can be specified honestly. v10 asserted an arm-density
# bound of 10.0 and then ran a lane that could not inform it: every eligible
# cell sat at or above the bound and five sat exactly on it.
#
# Every choice below is fixed here rather than left to the analyst:
#
# * The unit of observation is the CELL. Replications inside a cell share a
#   design point, so they sharpen that point's pass rate; they do not tell you
#   where the boundary is. Treating them as independent would inflate the
#   effective sample by ~250x and produce an interval that means nothing.
# * The outcome is the calibration screening gate that already decides whether
#   a cell is supported. A second definition of "supported" is how the r4 cycle
#   was lost, when a fix landed in one of two enrichment implementations.
# * The support set deliberately INCLUDES cells below the declared bound, and
#   excludes cells below the absolute arm-count floor. Below that floor the
#   count term binds rather than density, so such a cell would attribute a
#   count failure to density.
# * A common slope with per-stratum intercepts. Per-stratum slopes are not
#   identifiable on this grid -- (k=3, linear) carries three distinct densities,
#   leaving one residual degree of freedom -- while per-stratum intercepts still
#   let k=3 sit at a different boundary from k=2, which is the whole open
#   question.
# * report-only adoption. The estimate goes into the evidence and the report;
#   moving `reference_profile.minimum_arm_units_per_covariate` stays a human
#   decision needing a new freeze. A campaign that can rescope itself to the
#   cells that passed is the v8 trap with extra arithmetic.
# The external end-to-end lane, preregistered.
#
# Handed SCOVA's own folds the comparators reproduce SCOVA's nuisances exactly:
# both fit the declared learner family, and since v11 both fit the propensity
# one arm at a time. Measured on the v11 external cells, the k=3 comparisons --
# which carried 100% of the r9 lane's information -- now agree at exactly
# 0.000e+00. An identity is not corroboration, and `shared_score` already
# certifies the arithmetic at 1e-13.
#
# So the comparators get independent folds, which makes agreement a claim about
# two implementations rather than about one implementation twice. The cost is
# that raw differences now carry fold noise: fold-induced scatter alone was
# measured at a pooled mean |d| of 0.6324 against the old 0.25 tolerance and a
# maximum of 6.70 against 1.0. The same measurement kept the standardized
# offset within +/-1.5 across every stratum, because random fold noise averages
# out across replications and a real implementation difference does not. That
# is the statistic this lane now scores.
#
# The informative fraction is 1.0. Under independent splits there is no
# legitimate route to identity, so a degenerate cell means the independence did
# not take effect -- which is the silent-harness failure this lane has produced
# twice. Requiring every cell to be informative makes that unmissable.
EXTERNAL_AGREEMENT: dict[str, Any] = {
    "comparator_folds": "independent",
    # Added to each cell's declared random_state. The fold construction runs its
    # seed through a SplitMix64 avalanche, so any offset reorders the whole
    # partition rather than perturbing ties.
    "comparator_fold_seed_offset": 811,
    "statistic": "standardized-offset-z",
    "unit_of_observation": "cell-replication",
    "strata": "n_groups-by-learner",
    "family_wise_error": 0.05,
    "minimum_informative_cell_fraction": 1.0,
    "degenerate_difference_in_scova_se": 1e-10,
}

BOUNDARY_ESTIMATION: dict[str, Any] = {
    "target": "minimum_arm_units_per_covariate",
    "unit_of_observation": "cell",
    "outcome": "calibration-screening-cell-gate",
    "predictor": "log10-arm-units-per-covariate",
    "model": "logistic-common-slope-per-stratum-intercept",
    "strata": "n_groups-by-learner-within-claimed-scope",
    # Named rather than copied, so it cannot drift away from the gate that
    # actually defines a passing cell.
    "pass_probability_target": "metrics.minimum_strong_cell_pass_fraction",
    "minimum_distinct_densities_per_stratum": 3,
    "minimum_observations_per_parameter": 5,
    "require_bracketing_per_stratum": True,
    "interval_method": "cell-percentile-bootstrap",
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20260802,
    "adoption": "report-only",
}

# The strata the reference profile claims. maximum_group_count is 3, so k=5 is
# outside the claim and is covered by the pairwise fill rather than reserved.
CLAIMED_STRATA = tuple((groups, learner) for groups in (2, 3) for learner in ("linear", "adaptive"))


# Free factors -- confounding, confounding_form, effect, noise, overlap, surface --
# are deliberately NOT fixed here. `select_design` fills them greedily against the
# uncovered pair set, so a reserved cell still earns its budget back in coverage.
def _mandatory_cells() -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = []
    for groups, learner in CLAIMED_STRATA:
        # Two eligible cells per claimed stratum, differing in allocation AND in
        # arm density. One cell per stratum cannot separate a cell-specific
        # artifact from a property of the regime, which is exactly the position
        # v10's calibration lane was in with its single eligible k=3 cell.
        #
        # The first sits exactly ON the density bound: balanced allocation puts
        # n_per_group in the smallest arm, and 50 units over 5 covariates is
        # 10.0 against a bound of 10.0.
        cells.append(
            {
                "support": "strong",
                "n_groups": groups,
                "learner": learner,
                "allocation": "balanced",
                "n_per_group": 50,
                "n_covariates": 5,
            }
        )
        # The second sits well above it, under a skewed allocation so the
        # stratum is not a single design point measured twice.
        #
        # 150 rather than 250: `moderate` puts ~26% (k=2) or ~18% (k=3) of the
        # sample in the smallest arm, so 150 already yields 78 and 81 units
        # there -- clear of the count floor of 50, and 15.6 and 16.2 per
        # covariate against a bound of 10.0. Going to 250 buys more margin on a
        # term that is already satisfied while costing 67% more rows in the
        # most expensive cells in the reserved set.
        cells.append(
            {
                "support": "strong",
                "n_groups": groups,
                "learner": learner,
                "allocation": "moderate",
                "n_per_group": 150,
                "n_covariates": 5,
            }
        )
        # And one BELOW the bound, ineligible on density alone: 150 units in the
        # smallest arm clears the count floor of 50 outright, but over 20
        # covariates that is 7.5 per covariate.
        #
        # v10 could not inform its own density bound. Every one of its eligible
        # cells sat at or above 10.0 and five sat exactly on it, so the lane the
        # bound was fitted on contained nothing on the other side of it. These
        # cells are what make the bound estimable rather than merely asserted.
        cells.append(
            {
                "support": "strong",
                "n_groups": groups,
                "learner": learner,
                "allocation": "balanced",
                "n_per_group": 150,
                "n_covariates": 20,
            }
        )
    return tuple(cells)


MANDATORY_CELLS = _mandatory_cells()


def build_spec() -> dict[str, Any]:
    """v10's specification with the v11 identity and the rebuilt grid.

    Everything else is inherited verbatim and on purpose. The metrics, threshold
    quantile grids, seed partitions, software pins and the plasmode, external and
    inference lanes are not what v11 changes, and regenerating them would put
    unrelated churn into a checksum that gates evidence reuse.
    """
    spec = dict(build_v10_spec())
    retained, provenance = select_design(MANDATORY_CELLS, method=SELECTION_METHOD)
    spec["protocol_id"] = PROTOCOL_ID
    spec["retained_cells"] = retained
    spec["design_selection"] = provenance
    spec["boundary_estimation"] = dict(BOUNDARY_ESTIMATION)
    spec["external_agreement"] = dict(EXTERNAL_AGREEMENT)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/specs/cf_reference_v11.json")
    )
    args = parser.parse_args()
    spec = build_spec()
    provenance = spec["design_selection"]
    covered = provenance["pairwise_pairs_covered"]
    total = provenance["pairwise_pairs_total"]
    print(f"retained cells: {len(spec['retained_cells'])}")
    print(f"reserved cells: {provenance.get('mandatory_cells', 0)}")
    print(f"pairwise coverage: {covered}/{total} ({covered / total:.1%})")
    print("profile-eligible cells by (n_groups, learner):")
    for (groups, learner), count in sorted(eligible_cells_by_stratum(spec).items()):
        print(f"  k={groups} {learner:9} {count}")
    densities = sorted(
        round(expected_smallest_arm(cell) / cell["n_covariates"], 2)
        for cell in spec["retained_cells"]
        if cell["support"] == "strong" and int(cell["n_groups"]) <= 3
    )
    bound = float(spec["reference_profile"]["minimum_arm_units_per_covariate"])
    below = [value for value in densities if value < bound]
    print(f"strong k<=3 cells below the {bound} density bound: {len(below)} {below[:8]}")
    failures = design_coverage_failures(spec)
    for failure in failures:
        print(f"COVERAGE FAILURE: {failure}")
    if failures:
        raise SystemExit(f"{len(failures)} coverage failure(s); refusing to write")
    if args.write:
        args.output.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
