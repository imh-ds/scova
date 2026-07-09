import numpy as np
import pytest

from scova.experimental.tilts import geometric_tilt_and_gradient


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_randomized_jax_gradient_matrix(dtype: str) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    np_dtype = np.float64 if dtype == "float64" else np.float32
    rtol, atol = (1e-5, 1e-8) if dtype == "float64" else (5e-4, 1e-6)
    rng = np.random.default_rng(918_273 if dtype == "float64" else 192_837)
    cases = 256
    for case in range(cases):
        n_groups = (2, 4, 8)[case % 3]
        concentration = np.full(n_groups, 0.7)
        row = rng.dirichlet(concentration).astype(np_dtype)
        if case % 17 == 0:
            row[0] = np_dtype(1e-8)
            row[1:] *= np_dtype((1 - row[0]) / row[1:].sum())
        subset_size = 2 + case % (n_groups - 1)
        active = tuple(sorted(rng.choice(n_groups, subset_size, replace=False).tolist()))
        lam = np_dtype((case % 21) / 20)

        def function(probability, active_codes=active, lambda_value=lam):
            selected = probability[jnp.array(active_codes)]
            overlap = 1 / jnp.sum(1 / selected)
            return (len(active_codes) * overlap) ** lambda_value

        automatic = np.asarray(jax.grad(function)(jnp.asarray(row)))
        _, analytic = geometric_tilt_and_gradient(row[None, :], np.array([lam]), active)
        np.testing.assert_allclose(
            automatic,
            analytic[0, 0],
            rtol=rtol,
            atol=atol,
            err_msg=f"seeded JAX case {case}, dtype={dtype}, K={n_groups}, active={active}",
        )
