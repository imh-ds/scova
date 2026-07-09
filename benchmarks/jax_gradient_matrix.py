"""Large randomized analytic-versus-JAX gradient audit."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scova.experimental.tilts import geometric_tilt_and_gradient


def run(cases: int, seed: int, output: Path) -> None:
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(seed)
    failures = []
    maximum_scaled_error = 0.0
    for case in range(cases):
        dtype = np.float64 if case % 2 == 0 else np.float32
        rtol, atol = (1e-5, 1e-8) if dtype is np.float64 else (5e-4, 1e-6)
        n_groups = (2, 4, 8)[case % 3]
        row = rng.dirichlet(np.full(n_groups, 0.7)).astype(dtype)
        if case % 23 == 0:
            row[0] = dtype(1e-8)
            row[1:] *= dtype((1 - row[0]) / row[1:].sum())
        active_count = 2 + case % (n_groups - 1)
        active = tuple(sorted(rng.choice(n_groups, active_count, replace=False).tolist()))
        lam = dtype(rng.uniform())

        def function(probability, active_codes=active, lambda_value=lam):
            selected = probability[jnp.array(active_codes)]
            overlap = 1 / jnp.sum(1 / selected)
            return (len(active_codes) * overlap) ** lambda_value

        automatic = np.asarray(jax.grad(function)(jnp.asarray(row)))
        _, analytic = geometric_tilt_and_gradient(row[None, :], np.array([lam]), active)
        difference = np.abs(automatic - analytic[0, 0])
        tolerance = atol + rtol * np.abs(analytic[0, 0])
        scaled_error = float(np.max(difference / tolerance))
        maximum_scaled_error = max(maximum_scaled_error, scaled_error)
        if scaled_error > 1 or not np.all(np.isfinite(automatic)):
            failures.append(
                {
                    "case": case,
                    "dtype": np.dtype(dtype).name,
                    "n_groups": n_groups,
                    "active": active,
                    "lambda": float(lam),
                    "propensity": row.tolist(),
                    "automatic": automatic.tolist(),
                    "analytic": analytic[0, 0].tolist(),
                    "scaled_error": scaled_error,
                }
            )
    payload = {
        "schema_version": 1,
        "cases": cases,
        "seed": seed,
        "jax_version": jax.__version__,
        "maximum_scaled_error": maximum_scaled_error,
        "failures": failures,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    payload["sha256"] = sha256(encoded.encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} of {cases} JAX gradient cases failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=8_675_309)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cases, args.seed, args.output)


if __name__ == "__main__":
    main()
