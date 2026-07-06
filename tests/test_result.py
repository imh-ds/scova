import numpy as np
import pytest

from scova import SCOVA, NuisancePredictions, SCOVADeclaration, SCOVAResult
from scova.simulate import generate_data


def fitted_result() -> SCOVAResult:
    simulation = generate_data("randomized", n=240, seed=4)
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    declaration = SCOVADeclaration("outcome", "group", ("x1", "x2", "x3"), n_splits=3)
    return SCOVA().fit(simulation.data, declaration, nuisance_predictions=nuisance)


def test_custom_contrast_and_sign_invariance() -> None:
    result = fitted_result()
    contrast = result.contrast({"g0": 0.5, "g1": -0.5}, name="half")
    reverse = result.contrast({"g0": -0.5, "g1": 0.5})
    assert reverse.estimate == pytest.approx(-contrast.estimate)
    assert reverse.standard_error == pytest.approx(contrast.standard_error)
    assert reverse.p_value == pytest.approx(contrast.p_value)
    with pytest.raises(ValueError, match="sum to zero"):
        result.contrast([1, 1, 0])


def test_non_pickle_round_trip(tmp_path) -> None:
    result = fitted_result()
    result.contrast([1, -0.5, -0.5], name="one-versus-rest")
    path = tmp_path / "result.scova"
    result.save(path)
    loaded = SCOVAResult.load(path)
    assert loaded.group_labels == result.group_labels
    assert loaded.declaration_hash == result.declaration_hash
    assert loaded.diagnostics == result.diagnostics
    np.testing.assert_allclose(loaded.group_means, result.group_means)
    np.testing.assert_allclose(
        loaded.contrasts["one-versus-rest"].influence_values,
        result.contrasts["one-versus-rest"].influence_values,
    )
