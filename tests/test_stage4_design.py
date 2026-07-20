import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from scova import (
    DesignDeclaration,
    DesignLock,
    OutcomeFreeDesignData,
    SCOVADesign,
    SCOVADesignResult,
    SCOVAGraphResult,
)
from scova.experimental.gates import DiagnosticThresholds
from scova.simulate import generate_data


def thresholds() -> DiagnosticThresholds:
    return DiagnosticThresholds(
        version="stage4-test",
        calibrated=True,
        artifact_sha256="test-artifact",
        min_group_ess_warning=1,
        min_group_ess_refuse=0,
        min_target_ess_ratio_warning=0,
        min_target_ess_ratio_refuse=0,
        max_influence_share_warning=1,
        max_influence_share_refuse=1,
        max_weight_concentration_warning=1,
        max_weight_concentration_refuse=1,
        min_propensity_q01_warning=1e-12,
        min_propensity_q01_refuse=1e-14,
        max_calibration_error_warning=1,
        max_calibration_error_refuse=1,
        max_balance_warning=1_000,
        max_balance_refuse=10_000,
        max_crossfit_instability_warning=1,
        max_crossfit_instability_refuse=1,
    )


def prepared() -> tuple[SCOVADesign, SCOVADesignResult, np.ndarray]:
    simulation = generate_data("observational", n=300, seed=91)
    frame = simulation.data
    data = OutcomeFreeDesignData.from_arrays(
        frame.loc[:, ["x1", "x2", "x3"]].to_numpy(),
        frame["group"].tolist(),
        row_ids=list(range(len(frame))),
    )
    declaration = DesignDeclaration(
        group="group",
        covariates=("x1", "x2", "x3"),
        random_state=91,
        n_splits=2,
        lambdas=(0.0, 1.0),
    )
    engine = SCOVADesign(thresholds=thresholds())
    return engine, engine.prepare_design(data, declaration), frame["outcome"].to_numpy()


def test_design_accepts_an_unfitted_sklearn_outcome_model() -> None:
    model = RandomForestRegressor(n_estimators=2, random_state=1)
    engine = SCOVADesign(outcome_model=model)
    assert engine.outcome_model is model


def test_outer_split_analysis_round_trip_and_reports(tmp_path) -> None:
    engine, design, outcomes = prepared()
    assert design.selected_family
    destination = tmp_path / "design.json"
    design.save(destination)
    loaded = SCOVADesignResult.load(destination, data=design.data)
    ids = loaded.lock.estimation_row_ids
    result = engine.analyze_outcomes(
        loaded,
        [outcomes[row_id] for row_id in ids],
        row_ids=ids,
        n_bootstrap=9,
    )
    assert result.inference is not None
    assert result.reliability == "certified-overlap-only"
    report_file = tmp_path / "analysis.json"
    result.save(report_file)
    assert SCOVAGraphResult.load(report_file).report() == result.report()


def test_design_lock_and_outcome_alignment_refusals(tmp_path) -> None:
    engine, design, outcomes = prepared()
    ids = design.lock.estimation_row_ids
    with pytest.raises(ValueError, match="exactly match"):
        engine.analyze_outcomes(design, outcomes[: len(ids)], row_ids=ids[:-1])
    artifact = tmp_path / "design.json"
    design.save(artifact)
    altered = OutcomeFreeDesignData.from_arrays(
        design.data.covariates + 1,
        design.data.groups,
        row_ids=design.data.row_ids,
    )
    with pytest.raises(ValueError, match="design data"):
        SCOVADesignResult.load(artifact, data=altered)


def test_design_artifact_rejects_declaration_changes(tmp_path) -> None:
    _, design, _ = prepared()
    artifact = tmp_path / "design.json"
    design.save(artifact)
    values = json.loads(artifact.read_text(encoding="utf-8"))
    values["declaration_hash"] = "bad"
    artifact.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="declaration"):
        SCOVADesignResult.load(artifact, data=design.data)


@pytest.mark.parametrize(
    "matrix,groups,row_ids,message",
    [
        ([[0.0]], ["a"], [0], "at least two finite rows"),
        ([[np.nan], [0.0]], ["a", "b"], [0, 1], "finite rows"),
        ([[0.0], [1.0]], ["a", "a"], [0, 1], "two observed groups"),
        ([[0.0], [1.0]], ["a", "b"], [0, 0], "row_ids must be unique"),
    ],
)
def test_outcome_free_design_data_rejections(matrix, groups, row_ids, message) -> None:
    with pytest.raises(ValueError, match=message):
        OutcomeFreeDesignData.from_arrays(matrix, groups, row_ids=row_ids)


def test_outer_split_rejects_insufficient_group_rows() -> None:
    data = OutcomeFreeDesignData.from_arrays([[0.0], [1.0], [2.0], [3.0]], ["a", "a", "b", "b"])
    declaration = DesignDeclaration("group", ("x1",), n_splits=2)
    with pytest.raises(ValueError, match="n_splits"):
        SCOVADesign(thresholds=thresholds()).prepare_design(data, declaration)


def test_design_lock_rejects_tampering_and_verifies_inputs() -> None:
    engine, design, _ = prepared()
    lock = design.lock
    with pytest.raises(ValueError, match="checksum"):
        DesignLock(
            declaration_hash=lock.declaration_hash,
            data_hash=lock.data_hash,
            row_ids=lock.row_ids,
            split_assignments=lock.split_assignments,
            design_metadata=lock.design_metadata,
            lock_hash="tampered",
        )
    changed_declaration = DesignDeclaration(
        "group", ("x1", "x2", "x3"), random_state=92, n_splits=2, lambdas=(0.0, 1.0)
    )
    with pytest.raises(ValueError, match="declaration"):
        lock.verify(changed_declaration, design.data)
    assert engine.thresholds.version == "stage4-test"


def test_design_data_rejects_dataframe_and_non_json_labels() -> None:
    import pandas as pd

    with pytest.raises(TypeError, match="array"):
        OutcomeFreeDesignData.from_arrays(pd.DataFrame({"x": [0.0, 1.0]}), ["a", "b"])
    with pytest.raises(TypeError, match="JSON scalar"):
        OutcomeFreeDesignData.from_arrays([[0.0], [1.0]], [{"a": 1}, "b"])


def test_design_lock_rejects_invalid_partition_and_changed_data() -> None:
    _, design, _ = prepared()
    lock = design.lock
    with pytest.raises(ValueError, match="both design and estimation"):
        DesignLock(
            declaration_hash=lock.declaration_hash,
            data_hash=lock.data_hash,
            row_ids=lock.row_ids,
            split_assignments=("design",) * len(lock.row_ids),
            design_metadata=lock.design_metadata,
            lock_hash=lock.lock_hash,
        )
    reordered = OutcomeFreeDesignData.from_arrays(
        design.data.covariates,
        design.data.groups,
        row_ids=tuple(reversed(design.data.row_ids)),
    )
    with pytest.raises(ValueError, match="outcome-free"):
        lock.verify(design.declaration, reordered)


def test_design_lock_rejects_non_mapping_metadata() -> None:
    _, design, _ = prepared()
    lock = design.lock
    with pytest.raises(TypeError, match="mapping"):
        DesignLock(
            declaration_hash=lock.declaration_hash,
            data_hash=lock.data_hash,
            row_ids=lock.row_ids,
            split_assignments=lock.split_assignments,
            design_metadata=[],  # type: ignore[arg-type]
            lock_hash=lock.lock_hash,
        )
