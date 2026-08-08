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
        "propensity": "fixed-flexible",
        "outcome": "mean_squared_error",
    }
    assert len(selection["propensity"]) == 4
    assert set(selection["outcome"]) == set(result.group_labels)


def test_adaptive_propensity_is_always_the_flexible_learner() -> None:
    """The propensity is fixed, not scored.

    Selecting it by predictive fit chose the misspecified linear learner in
    ~all folds under nonlinear confounding, biasing the estimate while leaving
    the intervals narrow.  Every fold must report the flexible learner, and no
    log-loss score may be recorded, whatever the data look like.
    """
    for seed in (20, 31, 42):
        simulation = generate_data("randomized", n=360, seed=seed)
        result = SCOVA().fit(simulation.data, declaration(seed=seed + 1))
        folds = result.nuisance_metadata["selection"]["propensity"]
        assert folds, "adaptive cross-fitting must record a propensity choice per fold"
        assert {entry["selected"] for entry in folds} == {"HistGradientBoostingClassifier"}
        assert not any("scores" in entry for entry in folds)


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


def _propensity_rmse(fitted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(fitted - truth))))


def test_two_arm_propensity_is_bit_identical_under_either_parameterization() -> None:
    """The acceptance condition for the parameterization change.

    At two arms one-vs-rest and 2-class multinomial are the same estimator, so
    switching must not perturb a single bit -- not "agree to 1e-14", which is
    what refitting as two separate binary problems would actually give, since a
    solver run on `y` and on `1 - y` need not return exactly negated
    coefficients. Two-arm output is therefore left on the multiclass fit, and
    this test is what holds that decision in place.
    """
    simulation = generate_data("observational", n=400, n_groups=2, seed=5)
    multiclass = SCOVA().fit(simulation.data, declaration())
    one_vs_rest = SCOVA(propensity_parameterization="one-vs-rest").fit(
        simulation.data, declaration()
    )

    assert (
        multiclass.propensity_predictions.tobytes() == one_vs_rest.propensity_predictions.tobytes()
    )
    assert multiclass.group_means.tobytes() == one_vs_rest.group_means.tobytes()
    assert one_vs_rest.nuisance_metadata["propensity_parameterization"] == "multiclass"


def test_one_vs_rest_propensity_is_closer_to_truth_at_three_arms() -> None:
    """The reason the change exists, measured against the true per-unit propensity.

    AIPW's arm-`a` equation contains only `e_hat_a`, so the simplex constraint
    is a property of the true propensity rather than a requirement of the
    estimator. Buying it forces finite-sample estimation error to be shared
    across arms, and on confounded data with a flexible learner that costs
    accuracy.
    """
    simulation = generate_data("observational", n=600, n_groups=3, seed=7)
    multiclass = SCOVA().fit(simulation.data, declaration())
    one_vs_rest = SCOVA(propensity_parameterization="one-vs-rest").fit(
        simulation.data, declaration()
    )

    assert _propensity_rmse(
        one_vs_rest.propensity_predictions, simulation.propensity
    ) < _propensity_rmse(multiclass.propensity_predictions, simulation.propensity)
    assert one_vs_rest.nuisance_metadata["propensity_parameterization"] == "one-vs-rest"
    # Unnormalized by construction, and that is the point -- normalizing would
    # reintroduce exactly the coupling the change removes.
    assert not np.allclose(one_vs_rest.propensity_predictions.sum(axis=1), 1.0)


def test_default_parameterization_keeps_the_propensity_on_the_simplex() -> None:
    """`bounded_pairwise_anchor` scores `one_hot - propensity`.

    That is the multinomial score residual, and it is only a residual on the
    simplex. `design.py` feeds it a genuinely fitted propensity, so the default
    must stay multiclass until that score is derived again for unnormalized
    columns.
    """
    simulation = generate_data("observational", n=400, n_groups=3, seed=9)
    fitted = SCOVA().fit(simulation.data, declaration())

    assert SCOVA().propensity_parameterization == "multiclass"
    np.testing.assert_allclose(fitted.propensity_predictions.sum(axis=1), 1.0)
    assert fitted.nuisance_metadata["propensity_parameterization"] == "multiclass"


def test_one_vs_rest_uses_the_declared_learner_family() -> None:
    """Each arm model must be cloned from the spec the cell declares.

    An earlier throwaway harness hardcoded the flexible classifier regardless of
    the declared learner, so a linear cell compared LogisticRegression
    multiclass against boosted one-vs-rest and reported a spurious result. That
    is a comparison of learners wearing the costume of a comparison of
    parameterizations. Reconstructing the columns from the result's own folds
    proves the family and the fold structure both carried through.
    """
    simulation = generate_data("observational", n=400, n_groups=3, seed=13)
    fitted = SCOVA(
        propensity_model=LogisticRegression(max_iter=2000),
        outcome_model=Ridge(alpha=1.0),
        nuisance_strategy="custom",
        propensity_parameterization="one-vs-rest",
    ).fit(simulation.data, declaration())

    x = simulation.data.loc[:, ("x1", "x2", "x3")].to_numpy(dtype=float)
    codes = (
        simulation.data["group"]
        .map({label: code for code, label in enumerate(fitted.group_labels)})
        .to_numpy()
    )
    expected = np.empty_like(fitted.propensity_predictions)
    for fold in np.unique(fitted.fold_assignments):
        test = fitted.fold_assignments == fold
        train = ~test
        for code in range(len(fitted.group_labels)):
            arm = LogisticRegression(max_iter=2000)
            arm.fit(x[train], (codes[train] == code).astype(int))
            positive = int(np.flatnonzero(np.asarray(arm.classes_, dtype=int) == 1)[0])
            expected[test, code] = arm.predict_proba(x[test])[:, positive]

    np.testing.assert_allclose(fitted.propensity_predictions, expected, rtol=0, atol=1e-12)


def test_unknown_parameterization_is_rejected() -> None:
    with pytest.raises(ValueError, match="propensity_parameterization"):
        SCOVA(propensity_parameterization="ovr")  # type: ignore[arg-type]
