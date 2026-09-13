"""Public namespace and distribution metadata contracts."""

from __future__ import annotations

from pathlib import Path

import tomllib

import scova


def test_runtime_version_matches_project_metadata() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert scova.__version__ == project["project"]["version"]


def test_experimental_stages_are_not_top_level_exports() -> None:
    experimental_names = (
        "AnchoredBoundsDeclaration",
        "AnchoredBoundsResult",
        "AnchoredContrastResult",
        "ComparabilityGraphResult",
        "DesignDeclaration",
        "DesignLock",
        "LipschitzAnchorResult",
        "LipschitzContrastResult",
        "OutcomeFreeDesignData",
        "PairwiseDiagnosticInput",
        "PairwiseEdge",
        "SCOVADesign",
        "SCOVADesignResult",
        "SCOVAGraphResult",
        "SubsetDiagnosticInput",
        "SubsetHyperedge",
        "SupportGeometryDeclaration",
    )

    assert all(not hasattr(scova, name) for name in experimental_names)


def test_experimental_namespace_exposes_experimental_stages() -> None:
    from scova.experimental import (
        DesignDeclaration,
        OutcomeFreeDesignData,
        SCOVADesign,
    )

    assert DesignDeclaration is not None
    assert OutcomeFreeDesignData is not None
    assert SCOVADesign is not None
