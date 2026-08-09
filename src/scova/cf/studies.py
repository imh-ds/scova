"""Frozen-program metadata and deterministic designs for SCOVA-CF studies.

The two program types deliberately do not share an interpretation.  A
qualification artifact may enter the support-profile workflow; a methods
artifact may describe simulation behaviour only.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from itertools import combinations, product
from typing import Any

from .validation import canonical_checksum


class StudyProgram(str, Enum):
    """The permitted interpretation of a simulation artifact."""

    QUALIFICATION = "qualification"
    METHODS = "methods"


# v1 omitted two mandatory enrichment-screen thresholds.  It remains historical
# execution evidence only; v2 is the first dispatchable qualification freeze.
QUALIFICATION_PROTOCOL_ID = "cf-observational-adaptive-qualification-v2"
METHODS_STUDY_ID = "cf-observational-factorial-methods-v1"
ARTIFACT_SCHEMA_VERSION = 1


METHODS_FACTORS: Mapping[str, tuple[Any, ...]] = {
    "n_groups": (2, 3),
    "n_covariates": (3, 5),
    "n_per_group": (50, 150),
    "overlap": ("full", "poor"),
    "confounding_form": ("linear", "nonlinear"),
    "surface": ("linear", "interaction"),
    "noise": ("normal", "heavy-tailed"),
}


def _bits(index: int, width: int) -> tuple[int, ...]:
    return tuple((index >> shift) & 1 for shift in range(width))


def factorial_cells(surface_family: str = "interaction") -> tuple[dict[str, Any], ...]:
    """Return a 64-run regular fractional factorial design.

    The first six factors form a full 2**6 factorial.  The seventh uses the
    product of all six signs, yielding defining relation ``ABCDEFG = I`` and
    resolution VII (therefore no main or two-factor aliasing). Supplemental
    blocks compare linear with one named nonlinear surface at a time.
    """
    if surface_family not in {"interaction", "smooth-nonlinear", "threshold"}:
        raise ValueError("unknown methods surface family")
    names = tuple(METHODS_FACTORS)
    cells: list[dict[str, Any]] = []
    for index in range(64):
        bits = _bits(index, 6)
        seventh = sum(bits) % 2
        levels = (*bits, seventh)
        cell = {
            name: METHODS_FACTORS[name][level] for name, level in zip(names, levels, strict=True)
        }
        cell.update(
            {
                "cell_id": f"{surface_family}-{index:02d}",
                "allocation": "moderate",
                "confounding": "strong",
                "effect": "null",
                "support": "strong",
                "learner": "adaptive",
            }
        )
        # Each block is a linear-versus-named-surface comparison.  The
        # supplemental blocks therefore retain the factorial contrast rather
        # than pooling different nonlinear surfaces into one factor level.
        if surface_family != "interaction" and cell["surface"] == "interaction":
            cell["surface"] = surface_family
        cells.append(cell)
    return tuple(cells)


def methods_design() -> dict[str, Any]:
    """Describe the primary and supplemental factorial blocks without pooling."""
    primary = factorial_cells("interaction")
    supplemental = {family: factorial_cells(family) for family in ("smooth-nonlinear", "threshold")}
    payload = {
        "study_id": METHODS_STUDY_ID,
        "program_type": StudyProgram.METHODS.value,
        "replications_per_cell": 1000,
        "factors": {name: list(levels) for name, levels in METHODS_FACTORS.items()},
        "primary_surface_family": "interaction",
        "primary_cells": list(primary),
        "supplemental_surface_families": {
            name: list(cells) for name, cells in supplemental.items()
        },
        "alias_structure": "ABCDEFG=I (resolution VII; main and two-factor effects unaliased)",
    }
    return {**payload, "design_checksum": canonical_checksum(payload)}


QUALIFICATION_FACTORS: Mapping[str, tuple[Any, ...]] = {
    "allocation": ("balanced", "moderate"),
    "confounding": ("moderate", "strong"),
    "confounding_form": ("linear", "nonlinear"),
    "effect": ("null", "heterogeneous"),
    "learner": ("adaptive",),
    "n_covariates": (3, 5),
    "n_groups": (2, 3),
    "n_per_group": (50, 150),
    "noise": ("normal", "heavy-tailed"),
    "overlap": ("full", "poor"),
    "support": ("strong",),
    "surface": ("linear", "smooth-nonlinear", "threshold", "interaction"),
}


def _qualification_candidates() -> tuple[dict[str, Any], ...]:
    names = tuple(QUALIFICATION_FACTORS)
    return tuple(
        dict(zip(names, levels, strict=True))
        for levels in product(*(QUALIFICATION_FACTORS[name] for name in names))
    )


def _pair_set(cell: Mapping[str, Any]) -> set[tuple[str, Any, str, Any]]:
    return {
        (left, cell[left], right, cell[right])
        for left, right in combinations(tuple(QUALIFICATION_FACTORS), 2)
    }


def qualification_cells() -> tuple[dict[str, Any], ...]:
    """Select 48 frozen cells with mandatory stress cells and pair coverage."""
    candidates = _qualification_candidates()
    # One row for every group-count/surface/effect combination, all at the
    # difficult small-arm, poor-overlap, nonlinear, heavy-tailed corner.
    mandatory = [
        {
            "allocation": "balanced",
            "confounding": "strong",
            "confounding_form": "nonlinear",
            "effect": effect,
            "learner": "adaptive",
            "n_covariates": 5,
            "n_groups": groups,
            "n_per_group": 50,
            "noise": "heavy-tailed",
            "overlap": "poor",
            "support": "strong",
            "surface": surface,
        }
        for groups, surface, effect in product(
            (2, 3), ("smooth-nonlinear", "threshold", "interaction"), ("null", "heterogeneous")
        )
    ]
    selected = list(mandatory)
    uncovered = set().union(*(_pair_set(cell) for cell in candidates))
    for cell in selected:
        uncovered.difference_update(_pair_set(cell))
    remaining = [cell for cell in candidates if cell not in selected]
    while len(selected) < 48:
        best = max(
            remaining,
            key=lambda cell: (len(_pair_set(cell) & uncovered), canonical_checksum(cell)),
        )
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(_pair_set(best))
    return tuple({"cell_id": f"q{index:02d}", **cell} for index, cell in enumerate(selected))


def qualification_design() -> dict[str, Any]:
    """The prospective, adaptive-only qualification design and its fixed rules."""
    cells = qualification_cells()
    payload = {
        "protocol_id": QUALIFICATION_PROTOCOL_ID,
        "program_type": StudyProgram.QUALIFICATION.value,
        "replications_per_cell": 2000,
        "candidate_profile_state": "unpromoted",
        "promotion_rule": "independent-held-out-validation-and-human-approval",
        "maximum_covariate_count": 5,
        "nuisance_strategy": "adaptive",
        "factors": {name: list(levels) for name, levels in QUALIFICATION_FACTORS.items()},
        "mandatory_stress_conditions": {
            "n_per_group": 50,
            "overlap": "poor",
            "confounding_form": "nonlinear",
            "noise": "heavy-tailed",
            "surfaces": ["smooth-nonlinear", "threshold", "interaction"],
        },
        "support_calibration": "existing-frozen-screening-gates",
        "multiplicity": "Sidak-family-wise-coverage-and-type-I-control",
        "boundary_estimation": None,
        "cells": list(cells),
    }
    return {**payload, "design_checksum": canonical_checksum(payload)}


def assert_program_artifact(artifact: Mapping[str, Any], expected: StudyProgram) -> None:
    """Reject an artifact used outside its declared interpretation."""
    if artifact.get("program_type") != expected.value:
        raise ValueError(
            f"Expected a {expected.value} artifact; got {artifact.get('program_type')!r}"
        )
    if int(artifact.get("artifact_schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported SCOVA-CF study artifact schema")
    if not artifact.get("design_checksum") or not artifact.get("dependency_lock_checksum"):
        raise ValueError("Study artifact lacks frozen design or dependency identity")
