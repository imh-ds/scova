from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.cf_methods_study import methods_artifact
from benchmarks.cf_qualification_program import qualification_protocol, qualification_spec
from scova.cf import (
    StudyProgram,
    factorial_cells,
    methods_design,
    qualification_cells,
    qualification_design,
)
from scripts.calibrate_cf_support import calibrate
from scripts.render_cf_study_report import render


def test_qualification_design_is_adaptive_frozen_and_contains_every_stress_surface() -> None:
    design = qualification_design()
    cells = qualification_cells()
    assert design["program_type"] == StudyProgram.QUALIFICATION.value
    assert design["replications_per_cell"] == 2000
    assert len(cells) == 48
    assert len({cell["cell_id"] for cell in cells}) == 48
    assert {cell["learner"] for cell in cells} == {"adaptive"}
    assert {cell["n_groups"] for cell in cells} == {2, 3}
    assert max(cell["n_covariates"] for cell in cells) == 5
    stress = [
        cell
        for cell in cells
        if cell["n_per_group"] == 50
        and cell["overlap"] == "poor"
        and cell["confounding_form"] == "nonlinear"
        and cell["noise"] == "heavy-tailed"
    ]
    assert {"smooth-nonlinear", "threshold", "interaction"}.issubset(
        {cell["surface"] for cell in stress}
    )
    assert qualification_spec()["boundary_estimation"] is None


def test_qualification_protocol_has_no_density_boundary_and_is_deterministic() -> None:
    first = qualification_protocol()
    second = qualification_protocol()
    assert first.checksum == second.checksum
    assert first.reference_profile["maximum_covariate_count"] == 5
    assert "minimum_arm_units_per_covariate" not in first.reference_profile
    assert first.calibration.count == 2000
    assert first.metrics["minimum_unstable_risk_ratio"] == 2.0
    assert first.metrics["minimum_unstable_absolute_enrichment"] == 0.05


def test_qualification_protocol_refuses_enrichment_without_its_frozen_thresholds() -> None:
    protocol = qualification_protocol()
    malformed_metrics = dict(protocol.metrics)
    malformed_metrics.pop("minimum_unstable_risk_ratio")
    with pytest.raises(ValueError, match="Calibration enrichment screening is missing metrics"):
        replace(protocol, metrics=malformed_metrics)


def test_methods_design_is_64_run_unaliased_primary_with_separate_supplements() -> None:
    design = methods_design()
    assert len(design["primary_cells"]) == 64
    assert design["replications_per_cell"] == 1000
    assert "resolution VII" in design["alias_structure"]
    assert set(design["supplemental_surface_families"]) == {"smooth-nonlinear", "threshold"}
    for family in ("interaction", "smooth-nonlinear", "threshold"):
        cells = factorial_cells(family)
        assert len(cells) == 64
        assert len({cell["cell_id"] for cell in cells}) == 64
        assert {cell["surface"] for cell in cells} == {"linear", family}


def test_methods_artifact_is_complete_only_when_every_frozen_replication_exists() -> None:
    records = {cell["cell_id"]: [] for cell in factorial_cells("interaction")}
    artifact = methods_artifact(
        surface_family="interaction", cell_records=records, replications_per_cell=1000
    )
    assert artifact["program_type"] == "methods"
    assert not artifact["complete"]
    assert artifact["planned_replications"] == 64_000
    assert "candidate profile" in artifact["interpretation"]


def test_calibration_rejects_a_methods_artifact_before_any_profile_logic() -> None:
    with pytest.raises(ValueError, match="Methods-study evidence"):
        calibrate(qualification_protocol(), {"program_type": "methods", "evidence_checksum": "x"})


def test_methods_report_refuses_to_use_qualification_language() -> None:
    records = {cell["cell_id"]: [] for cell in factorial_cells("interaction")}
    report = render(methods_artifact(surface_family="interaction", cell_records=records))
    assert "do not create a support profile" in report
    assert "Design checksum" in report
