import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scova import SCOVA, ContrastSpec, NuisancePredictions, SCOVADeclaration, SCOVAResult
from scova.estimator import _assemble_aipw, _validate_probabilities
from scova.experimental import PathDeclaration, SCOVAPathResult, fit_path
from scova.experimental.tilts import (
    finite_difference_gradient,
    geometric_tilt_and_gradient,
    harmonic_overlap,
    validate_active_groups,
)
from scova.simulate import generate_data


def test_declaration_and_contrast_failure_branches() -> None:
    with pytest.raises(ValueError, match="empty"):
        ContrastSpec(" ", (("a", 1), ("b", -1)))
    with pytest.raises(ValueError, match="at least two"):
        ContrastSpec("one", (("a", 0),))
    with pytest.raises(ValueError, match="repeat"):
        ContrastSpec("dup", (("a", 1), ("a", -1)))
    with pytest.raises(ValueError, match="nonzero"):
        ContrastSpec("zero", (("a", 0), ("b", 0)))
    with pytest.raises(TypeError, match="JSON scalar"):
        ContrastSpec("object", ((("tuple",), 1), ("b", -1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        SCOVADeclaration("", "a", ("x",))
    with pytest.raises(ValueError, match="Interpretation"):
        SCOVADeclaration("y", "a", ("x",), interpretation="invalid")  # type: ignore[arg-type]
    duplicate = ContrastSpec("same", (("a", 1), ("b", -1)))
    with pytest.raises(ValueError, match="unique"):
        SCOVADeclaration("y", "a", ("x",), contrasts=(duplicate, duplicate))


def test_estimator_validation_failure_branches() -> None:
    with pytest.raises(ValueError, match="shape"):
        _validate_probabilities(np.ones((2, 2)), 3, 2)
    with pytest.raises(ValueError, match="finite"):
        _validate_probabilities(np.array([[np.nan, np.nan]]), 1, 2)
    with pytest.raises(ValueError, match="sum to one"):
        _validate_probabilities(np.array([[0.2, 0.2]]), 1, 2)
    with pytest.raises(ValueError, match="shape"):
        _assemble_aipw(np.ones(2), np.array([0, 1]), np.full((2, 2), 0.5), np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        _assemble_aipw(
            np.ones(2),
            np.array([0, 1]),
            np.full((2, 2), 0.5),
            np.full((2, 2), np.nan),
        )
    declaration = SCOVADeclaration("y", "a", ("x",), n_splits=2)
    with pytest.raises(TypeError, match="DataFrame"):
        SCOVA().fit([], declaration)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing declared"):
        SCOVA().fit(pd.DataFrame({"x": [1, 2]}), declaration)
    nonnumeric = pd.DataFrame({"x": ["x", "y", "z", "w"], "a": [0, 0, 1, 1], "y": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="numeric"):
        SCOVA().fit(nonnumeric, declaration)
    one_group = pd.DataFrame({"x": range(4), "a": [0] * 4, "y": range(4)})
    with pytest.raises(ValueError, match="at least two"):
        SCOVA().fit(one_group, declaration)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"lambdas": (0.0, np.nan, 1.0)}, "finite"),
        ({"target": "bad"}, "unknown path"),
        ({"target": "subset", "active_groups": ("g0",)}, "at least two"),
        ({"target": "kway", "active_groups": ("g0", "g1")}, "infer active"),
        ({"target": "pairwise", "active_groups": ("g0", "g0")}, "duplicates"),
        ({"contrast_names": ("x", "x")}, "duplicates"),
        ({"confidence_level": 1.0}, "strictly between"),
    ],
)
def test_path_declaration_additional_failures(kwargs, message) -> None:
    base = SCOVADeclaration("y", "a", ("x",))
    with pytest.raises(ValueError, match=message):
        PathDeclaration(base, **kwargs)


def test_tilt_failure_branches() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        validate_active_groups(3, (0, 0))
    with pytest.raises(ValueError, match="outside"):
        validate_active_groups(2, (0, 2))
    with pytest.raises(ValueError, match="two-dimensional"):
        harmonic_overlap(np.array([0.5, 0.5]), (0, 1))
    with pytest.raises(ValueError, match="finite"):
        harmonic_overlap(np.array([[np.nan, 0.5]]), (0, 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        geometric_tilt_and_gradient(np.array([[0.5, 0.5]]), np.ones((1, 1)), (0, 1))
    with pytest.raises(ValueError, match="simplex boundary"):
        finite_difference_gradient(np.array([1e-8, 1 - 1e-8]), np.array([0.5]), (0, 1))


def test_path_fit_and_inference_failure_branches(tmp_path) -> None:
    simulation = generate_data("randomized", n=240, seed=5)
    base = SCOVADeclaration("outcome", "group", ("x1", "x2", "x3"), n_splits=3)
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    unknown_group = PathDeclaration(
        base, lambdas=(0.0, 1.0), target="pairwise", active_groups=("g0", "missing")
    )
    with pytest.raises(ValueError, match="unknown active"):
        fit_path(simulation.data, unknown_group, nuisance_predictions=nuisance)
    unknown_contrast = PathDeclaration(base, lambdas=(0.0, 1.0), contrast_names=("missing",))
    with pytest.raises(ValueError, match="unknown contrast"):
        fit_path(simulation.data, unknown_contrast, nuisance_predictions=nuisance)
    declaration = PathDeclaration(base, lambdas=(0.0, 0.5, 1.0))
    result = fit_path(simulation.data, declaration, nuisance_predictions=nuisance)
    with pytest.raises(ValueError, match="at least one"):
        result.infer(())
    name = next(iter(result.contrasts))
    with pytest.raises(ValueError, match="duplicate"):
        result.infer((name, name))
    with pytest.raises(ValueError, match="unknown"):
        result.infer(("missing",))
    with pytest.raises(ValueError, match="confidence"):
        result.infer((name,), confidence_level=0)
    with pytest.raises(ValueError, match="positive"):
        result.infer((name,), n_bootstrap=0)
    bad = result.contrasts[name]
    result.contrasts[name] = replace(bad, standard_errors=np.zeros_like(bad.standard_errors))
    with pytest.raises(ValueError, match="positive standard error"):
        result.infer((name,))
    result.contrasts[name] = bad
    assert result.infer((name,), n_bootstrap=9).status.value == "warning"
    destination = tmp_path / "path.scova"
    result.save(destination)
    with np.load(destination, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["schema_version"] = 99
    arrays["metadata"] = np.array(json.dumps(metadata))
    corrupt = tmp_path / "corrupt.scova"
    with corrupt.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError, match="unsupported"):
        SCOVAPathResult.load(corrupt)


def test_core_result_failure_branches(tmp_path) -> None:
    simulation = generate_data("randomized", n=180, seed=6)
    declaration = SCOVADeclaration("outcome", "group", ("x1", "x2", "x3"), n_splits=3)
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    result = SCOVA().fit(simulation.data, declaration, nuisance_predictions=nuisance)
    with pytest.raises(ValueError, match="confidence_level"):
        result.group_confidence_intervals(0)
    with pytest.raises(ValueError, match="unknown groups"):
        result.contrast({"missing": 1, "g0": -1})
    with pytest.raises(ValueError, match="finite"):
        result.contrast([np.nan, 0, 0])
    with pytest.raises(ValueError, match="nonzero"):
        result.contrast([0, 0, 0])
    with pytest.raises(ValueError, match="confidence_level"):
        result.contrast([1, -1, 0], confidence_level=1)
    result.diagnostics["warnings"] = "bad"
    with pytest.raises(ValueError, match="must be a list"):
        result.infer(n_bootstrap=9)
    path = tmp_path / "result.scova"
    result.diagnostics.pop("warnings")
    result.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["schema_version"] = 99
    arrays["metadata"] = np.array(json.dumps(metadata))
    corrupt = tmp_path / "bad-result.scova"
    with corrupt.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with pytest.raises(ValueError, match="Unsupported"):
        SCOVAResult.load(corrupt)
