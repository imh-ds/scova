"""Scheduled finite-grid coverage and correction-ablation benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from scova import SCOVADeclaration
from scova.experimental import PathDeclaration, fit_path
from scova.experimental.tilts import geometric_tilt_and_gradient
from scova.simulate import generate_data


def _naive_uniform_band(
    estimates: np.ndarray,
    influence: np.ndarray,
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(influence)
    errors = np.sqrt(np.sum(np.square(influence), axis=0) / (n * (n - 1)))
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(n_bootstrap, n))
    multipliers -= multipliers.mean(axis=1, keepdims=True)
    statistics = (multipliers @ influence) / n / errors
    maxima = np.max(np.abs(statistics), axis=1)
    critical = np.quantile(maxima, confidence_level, method="higher")
    return estimates - critical * errors, estimates + critical * errors


def run_benchmark(repetitions: int, n_bootstrap: int, output: Path) -> None:
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for overlap in ("strong", "moderate"):
        scenario = "randomized" if overlap == "strong" else "observational"
        for n_groups in (2, 4, 8):
            corrected_coverage = 0
            naive_coverage = 0
            pointwise_path_coverage = 0
            corrected_error = 0.0
            naive_error = 0.0
            for repetition in range(repetitions):
                simulation = generate_data(
                    scenario,
                    n=750,
                    n_groups=n_groups,
                    seed=300_000 + repetition,
                )
                base = SCOVADeclaration(
                    "outcome",
                    "group",
                    ("x1", "x2", "x3"),
                    n_splits=5,
                    random_state=repetition,
                )
                declaration = PathDeclaration(base)
                result = fit_path(simulation.data, declaration)
                contrast_name = f"g0 - g{n_groups - 1}"
                contrast = result.contrasts[contrast_name]
                inference = result.infer(
                    (contrast_name,), n_bootstrap=n_bootstrap, random_state=repetition
                )
                tilt, _ = geometric_tilt_and_gradient(
                    simulation.propensity,
                    result.lambdas,
                    tuple(range(n_groups)),
                )
                truth_means = (
                    np.einsum("nl,nk->lk", tilt, simulation.outcome_regression)
                    / tilt.sum(axis=0)[:, None]
                )
                truth = truth_means[:, 0] - truth_means[:, -1]
                lower = inference.lower_bands[0]
                upper = inference.upper_bands[0]
                corrected_coverage += bool(np.all((lower <= truth) & (truth <= upper)))
                naive_estimate = result.naive_group_means[:, 0] - result.naive_group_means[:, -1]
                naive_influence = (
                    result.naive_influence_values[:, :, 0]
                    - result.naive_influence_values[:, :, -1]
                )
                naive_lower, naive_upper = _naive_uniform_band(
                    naive_estimate,
                    naive_influence,
                    confidence_level=0.95,
                    n_bootstrap=n_bootstrap,
                    seed=repetition,
                )
                naive_coverage += bool(
                    np.all((naive_lower <= truth) & (truth <= naive_upper))
                )
                critical = norm.ppf(0.975)
                pointwise_path_coverage += bool(
                    np.all(
                        (contrast.estimates - critical * contrast.standard_errors <= truth)
                        & (truth <= contrast.estimates + critical * contrast.standard_errors)
                    )
                )
                corrected_error += float(np.mean(np.square(contrast.estimates - truth)))
                naive_error += float(np.mean(np.square(naive_estimate - truth)))
            records.append(
                {
                    "overlap": overlap,
                    "n_groups": n_groups,
                    "repetitions": repetitions,
                    "n_bootstrap": n_bootstrap,
                    "corrected_uniform_coverage": corrected_coverage / repetitions,
                    "naive_uniform_coverage": naive_coverage / repetitions,
                    "all_pointwise_intervals_cover": pointwise_path_coverage / repetitions,
                    "corrected_path_rmse": np.sqrt(corrected_error / repetitions),
                    "naive_path_rmse": np.sqrt(naive_error / repetitions),
                }
            )
    payload = {
        "schema_version": 1,
        "benchmark": "stage3_path",
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=1999)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_artifacts/stage3_path.json")
    )
    arguments = parser.parse_args()
    run_benchmark(arguments.repetitions, arguments.bootstrap, arguments.output)


if __name__ == "__main__":
    main()
