import numpy as np

from scova.simulate import eif_perturbation_check, generate_data


def test_simulation_is_deterministic_and_oracles_are_valid() -> None:
    first = generate_data("weak_overlap", n=200, seed=9)
    second = generate_data("weak_overlap", n=200, seed=9)
    assert first.data.equals(second.data)
    np.testing.assert_allclose(first.propensity, second.propensity)
    np.testing.assert_allclose(first.propensity.sum(axis=1), 1)
    assert np.all(first.propensity > 0)
    np.testing.assert_allclose(first.true_group_means, first.outcome_regression.mean(axis=0))


def test_eif_perturbation_identity() -> None:
    signal = np.array([-1.0, 0.5, 2.0, 4.0])
    score = np.array([0.2, -0.8, 1.2, -0.3])
    finite_difference, inner_product = eif_perturbation_check(signal, score)
    assert finite_difference == pytest.approx(inner_product, rel=1e-8, abs=1e-9)


import pytest  # noqa: E402
