import numpy as np
import pytest

from scova.experimental.tilts import (
    finite_difference_gradient,
    geometric_tilt_and_gradient,
    harmonic_overlap,
)


def test_binary_and_multigroup_harmonic_overlap() -> None:
    binary = np.array([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_allclose(harmonic_overlap(binary, (0, 1)), binary[:, 0] * binary[:, 1])
    multi = np.array([[0.2, 0.3, 0.5]])
    expected = 1 / np.sum(1 / multi, axis=1)
    np.testing.assert_allclose(harmonic_overlap(multi, (0, 1, 2)), expected)
    np.testing.assert_allclose(harmonic_overlap(multi, (0, 2)), 1 / (1 / 0.2 + 1 / 0.5))


def test_geometric_endpoints_and_analytic_gradient() -> None:
    propensity = np.array([[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]])
    lambdas = np.array([0.0, 0.4, 1.0])
    tilt, gradient = geometric_tilt_and_gradient(propensity, lambdas, (0, 1, 2))
    np.testing.assert_allclose(tilt[:, 0], 1)
    np.testing.assert_allclose(gradient[:, 0], 0)
    overlap = harmonic_overlap(propensity, (0, 1, 2))
    np.testing.assert_allclose(tilt[:, -1], 3 * overlap)
    for row in propensity:
        numerical = finite_difference_gradient(row, lambdas, (0, 1, 2))
        _, analytic = geometric_tilt_and_gradient(row[None, :], lambdas, (0, 1, 2))
        np.testing.assert_allclose(analytic[0], numerical, rtol=2e-5, atol=2e-7)


def test_simplex_gradient_constant_invariance() -> None:
    propensity = np.array([[0.2, 0.3, 0.5]])
    lambdas = np.array([0.3, 1.0])
    _, gradient = geometric_tilt_and_gradient(propensity, lambdas, (0, 1, 2))
    residual = np.array([[0.8, -0.3, -0.5]])
    original = np.einsum("nlk,nk->nl", gradient, residual)
    shifted = np.einsum("nlk,nk->nl", gradient + 7.2, residual)
    np.testing.assert_allclose(original, shifted)


def test_study_tilt_and_failures() -> None:
    propensity = np.array([[0.4, 0.6]])
    tilt, gradient = geometric_tilt_and_gradient(
        propensity, np.array([0.0, 1.0]), (0, 1), target="study"
    )
    np.testing.assert_allclose(tilt, 1)
    np.testing.assert_allclose(gradient, 0)
    with pytest.raises(ValueError, match="strictly positive"):
        harmonic_overlap(np.array([[0.0, 1.0]]), (0, 1))
    with pytest.raises(ValueError, match="at least two"):
        harmonic_overlap(propensity, (0,))


def test_optional_jax_gradient_agrees() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    row = jnp.array([0.2, 0.3, 0.5])
    lam = 0.7

    def tilt_function(probability):
        overlap = 1 / jnp.sum(1 / probability)
        return (3 * overlap) ** lam

    automatic = np.asarray(jax.grad(tilt_function)(row))
    _, analytic = geometric_tilt_and_gradient(np.asarray(row)[None, :], np.array([lam]), (0, 1, 2))
    np.testing.assert_allclose(automatic, analytic[0, 0], rtol=2e-5, atol=2e-6)
