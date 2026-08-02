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

# The strata the reference profile claims. maximum_group_count is 3, so k=5 is
# outside the claim and is covered by the pairwise fill rather than reserved.
CLAIMED_STRATA = tuple(
    (groups, learner) for groups in (2, 3) for learner in ("linear", "adaptive")
)

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
