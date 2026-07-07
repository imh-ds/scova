import numpy as np
import pytest

from scova import SCOVA, NuisancePredictions, SCOVADeclaration
from scova.experimental import PathDeclaration, SCOVAPathResult, fit_path
from scova.simulate import generate_data


def setup_path(*, target="kway", active_groups=(), contrast_names=()):
    simulation = generate_data("observational", n=500, seed=29)
    base = SCOVADeclaration(
        "outcome", "group", ("x1", "x2", "x3"), n_splits=4, random_state=19
    )
    declaration = PathDeclaration(
        base,
        lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
        target=target,
        active_groups=active_groups,
        contrast_names=contrast_names,
    )
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    return simulation, base, declaration, nuisance


def test_lambda_zero_exactly_reduces_to_fixed_target() -> None:
    simulation, base, declaration, nuisance = setup_path()
    fixed = SCOVA().fit(simulation.data, base, nuisance_predictions=nuisance)
    path = fit_path(simulation.data, declaration, nuisance_predictions=nuisance)
    np.testing.assert_allclose(path.group_means[0], fixed.group_means, atol=1e-14)
    np.testing.assert_allclose(path.influence_values[:, 0], fixed.influence_values, atol=1e-13)
    assert path.diagnostics["inference_scope"] == "declared-finite-grid"
    assert path.drift.standardized_mean_shifts.shape == (5, 3)
    np.testing.assert_allclose(path.drift.standardized_mean_shifts[0], 0, atol=1e-14)


def test_correction_ablation_and_study_path() -> None:
    simulation, _, declaration, nuisance = setup_path()
    path = fit_path(simulation.data, declaration, nuisance_predictions=nuisance)
    np.testing.assert_allclose(path.group_means[0], path.naive_group_means[0])
    assert np.max(np.abs(path.group_means[-1] - path.naive_group_means[-1])) > 1e-8
    study = PathDeclaration(declaration.base, lambdas=(0.0, 0.5, 1.0), target="study")
    study_result = fit_path(simulation.data, study, nuisance_predictions=nuisance)
    np.testing.assert_allclose(
        study_result.group_means,
        np.repeat(study_result.group_means[[0]], len(study_result.lambdas), axis=0),
    )
    np.testing.assert_allclose(study_result.naive_group_means, study_result.group_means)


def test_pairwise_target_keeps_only_compatible_contrast() -> None:
    simulation, _, declaration, nuisance = setup_path(
        target="pairwise", active_groups=("g0", "g2")
    )
    path = fit_path(simulation.data, declaration, nuisance_predictions=nuisance)
    assert tuple(path.contrasts) == ("g0 - g2",)
    assert path.active_groups == ("g0", "g2")
    explicit_bad = PathDeclaration(
        declaration.base,
        lambdas=declaration.lambdas,
        target="pairwise",
        active_groups=("g0", "g2"),
        contrast_names=("g0 - g1",),
    )
    with pytest.raises(ValueError, match="outside"):
        fit_path(simulation.data, explicit_bad, nuisance_predictions=nuisance)


def test_path_inference_determinism_certificates_and_persistence(tmp_path) -> None:
    simulation, _, declaration, nuisance = setup_path()
    path = fit_path(simulation.data, declaration, nuisance_predictions=nuisance)
    first = path.infer(n_bootstrap=99, random_state=7, batch_size=13)
    second = path.infer(n_bootstrap=99, random_state=7, batch_size=99)
    np.testing.assert_allclose(first.lower_bands, second.lower_bands)
    np.testing.assert_allclose(first.upper_bands, second.upper_bands)
    assert first.critical_value == pytest.approx(second.critical_value)
    assert len(first.sign_certificates) == len(first.family)
    assert all(
        item.simultaneous_upper_bound >= item.estimated_maximum_drift
        for item in first.stability_certificates
    )
    destination = tmp_path / "path.scova"
    path.save(destination)
    loaded = SCOVAPathResult.load(destination)
    np.testing.assert_allclose(loaded.group_means, path.group_means)
    replay = loaded.infer(n_bootstrap=99, random_state=7, batch_size=17)
    np.testing.assert_allclose(replay.lower_bands, first.lower_bands)


def test_path_is_row_order_invariant_with_oracles() -> None:
    simulation, _, declaration, nuisance = setup_path()
    original = fit_path(simulation.data, declaration, nuisance_predictions=nuisance)
    permutation = np.random.default_rng(91).permutation(len(simulation.data))
    shuffled = fit_path(
        simulation.data.iloc[permutation].reset_index(drop=True),
        declaration,
        nuisance_predictions=NuisancePredictions(
            simulation.propensity[permutation],
            simulation.outcome_regression[permutation],
            simulation.group_labels,
        ),
    )
    np.testing.assert_allclose(shuffled.group_means, original.group_means)
    np.testing.assert_allclose(
        shuffled.drift.standardized_mean_shifts,
        original.drift.standardized_mean_shifts,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"lambdas": (0.1, 1.0)}, "endpoints"),
        ({"lambdas": (0.0, 0.5, 0.5, 1.0)}, "strictly increasing"),
        ({"target": "pairwise", "active_groups": ("g0",)}, "exactly two"),
    ],
)
def test_path_declaration_failures(kwargs, message) -> None:
    base = SCOVADeclaration("outcome", "group", ("x1",))
    with pytest.raises(ValueError, match=message):
        PathDeclaration(base, **kwargs)
