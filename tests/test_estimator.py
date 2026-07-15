import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from scova import SCOVA, NuisancePredictions, SCOVADeclaration, Verdict
from scova.simulate import generate_data


def declaration(seed: int = 11) -> SCOVADeclaration:
    return SCOVADeclaration(
        outcome="outcome",
        group="group",
        covariates=("x1", "x2", "x3"),
        n_splits=4,
        random_state=seed,
    )


def test_oracle_aipw_matches_hand_calculation() -> None:
    simulation = generate_data("observational", n=500, seed=3)
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    result = SCOVA().fit(simulation.data, declaration(), nuisance_predictions=nuisance)
    codes = (
        simulation.data["group"]
        .map({label: code for code, label in enumerate(result.group_labels)})
        .to_numpy()
    )
    y = simulation.data["outcome"].to_numpy()
    observed = np.eye(len(result.group_labels))[codes]
    signal = simulation.outcome_regression + observed / simulation.propensity * (
        y[:, None] - simulation.outcome_regression
    )
    np.testing.assert_allclose(result.group_means, signal.mean(axis=0))
    np.testing.assert_allclose(result.influence_values.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(result.covariance, result.covariance.T)
    assert np.linalg.eigvalsh(result.covariance).min() >= -1e-12
    assert result.group_standard_errors.shape == result.group_means.shape
    assert result.group_confidence_intervals().shape == (len(result.group_labels), 2)
    assert result.verdict is Verdict.DESCRIPTIVE_ONLY


def test_cross_fit_is_deterministic_and_complete() -> None:
    simulation = generate_data("randomized", n=360, seed=2)
    first = SCOVA().fit(simulation.data, declaration(seed=19))
    second = SCOVA().fit(simulation.data, declaration(seed=19))
    np.testing.assert_array_equal(first.fold_assignments, second.fold_assignments)
    np.testing.assert_allclose(first.group_means, second.group_means)
    assert set(first.fold_assignments) == {0, 1, 2, 3}
    assert np.all(np.isfinite(first.propensity_predictions))
    assert np.all(np.isfinite(first.outcome_predictions))


def test_default_adaptive_nuisance_selection_is_recorded() -> None:
    simulation = generate_data("randomized", n=360, seed=20)
    result = SCOVA().fit(simulation.data, declaration(seed=23))
    metadata = result.nuisance_metadata
    assert metadata["nuisance_strategy"] == "adaptive"
    assert metadata["propensity_model"] == "adaptive"
    assert metadata["outcome_model"] == "adaptive"
    selection = metadata["selection"]
    assert selection["criterion"] == {
        "propensity": "log_loss",
        "outcome": "mean_squared_error",
    }
    assert len(selection["propensity"]) == 4
    assert set(selection["outcome"]) == set(result.group_labels)


def test_linear_and_custom_nuisance_strategies_remain_available() -> None:
    simulation = generate_data("randomized", n=240, seed=21)
    linear = SCOVA(nuisance_strategy="linear").fit(simulation.data, declaration(seed=24))
    assert linear.nuisance_metadata["propensity_model"] == "LogisticRegression"
    assert linear.nuisance_metadata["outcome_model"] == "Ridge"
    custom = SCOVA(
        propensity_model=LogisticRegression(max_iter=2000),
        outcome_model=Ridge(alpha=2.0),
        nuisance_strategy="custom",
    ).fit(simulation.data, declaration(seed=24))
    assert custom.nuisance_metadata["nuisance_strategy"] == "custom"
    with pytest.raises(ValueError, match="supplied together"):
        SCOVA(propensity_model=LogisticRegression(max_iter=2000))
    with pytest.raises(ValueError, match="nuisance_strategy"):
        SCOVA(nuisance_strategy="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires both"):
        SCOVA(nuisance_strategy="custom")
    with pytest.raises(ValueError, match="require nuisance_strategy='custom'"):
        SCOVA(
            propensity_model=LogisticRegression(max_iter=2000),
            outcome_model=Ridge(),
            nuisance_strategy="linear",
        )


def test_row_order_and_group_relabeling_invariance_with_oracles() -> None:
    simulation = generate_data("observational", n=400, seed=12)
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    original = SCOVA().fit(simulation.data, declaration(), nuisance_predictions=nuisance)
    permutation = np.random.default_rng(5).permutation(len(simulation.data))
    shuffled_nuisance = NuisancePredictions(
        simulation.propensity[permutation],
        simulation.outcome_regression[permutation],
        simulation.group_labels,
    )
    shuffled = SCOVA().fit(
        simulation.data.iloc[permutation].reset_index(drop=True),
        declaration(),
        nuisance_predictions=shuffled_nuisance,
    )
    np.testing.assert_allclose(original.group_means, shuffled.group_means)

    relabeled_data = simulation.data.copy()
    mapping = {"g0": 20, "g1": 5, "g2": 10}
    relabeled_data["group"] = relabeled_data["group"].map(mapping)
    reorder = [1, 2, 0]  # canonical labels are (5, 10, 20)
    relabeled_nuisance = NuisancePredictions(
        simulation.propensity[:, reorder],
        simulation.outcome_regression[:, reorder],
        (5, 10, 20),
    )
    relabeled = SCOVA().fit(relabeled_data, declaration(), nuisance_predictions=relabeled_nuisance)
    np.testing.assert_allclose(relabeled.group_means, original.group_means[reorder])


def test_invalid_data_and_probabilities_are_rejected() -> None:
    simulation = generate_data("randomized", n=120, seed=8)
    bad = simulation.data.copy()
    bad.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        SCOVA().fit(bad, declaration())

    probability = simulation.propensity.copy()
    probability[0, 0] = 0
    nuisance = NuisancePredictions(
        probability, simulation.outcome_regression, simulation.group_labels
    )
    with pytest.raises(ValueError, match="strictly positive"):
        SCOVA().fit(simulation.data, declaration(), nuisance_predictions=nuisance)


def test_small_group_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "x1": range(7),
            "x2": range(7),
            "x3": range(7),
            "group": ["a"] * 6 + ["b"],
            "outcome": range(7),
        }
    )
    with pytest.raises(ValueError, match="n_splits"):
        SCOVA().fit(data, declaration())
