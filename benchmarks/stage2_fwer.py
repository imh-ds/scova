"""Scheduled Stage-2 max-t FWER and Wald-size benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scova import SCOVA, NuisancePredictions, SCOVADeclaration
from scova.simulate import generate_data


def run_benchmark(repetitions: int, n_bootstrap: int, output: Path) -> None:
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for overlap in ("strong", "moderate"):
        scenario = "randomized" if overlap == "strong" else "observational"
        for n_groups in (2, 4, 8):
            max_t_rejections = 0
            wald_rejections = 0
            simultaneous_coverage = 0
            any_contrast_power = 0
            for repetition in range(repetitions):
                simulation = generate_data(
                    scenario, n=500, n_groups=n_groups, seed=100_000 + repetition
                )
                common_mean = simulation.outcome_regression.mean(axis=1, keepdims=True)
                null_regression = np.repeat(common_mean, n_groups, axis=1)
                rng = np.random.default_rng(200_000 + repetition)
                codes = simulation.data["group"].str[1:].astype(int).to_numpy()
                data = simulation.data.copy()
                data["outcome"] = null_regression[np.arange(len(data)), codes] + rng.normal(
                    size=len(data)
                )
                nuisance = NuisancePredictions(
                    simulation.propensity, null_regression, simulation.group_labels
                )
                declaration = SCOVADeclaration(
                    "outcome",
                    "group",
                    ("x1", "x2", "x3"),
                    n_splits=5,
                    random_state=repetition,
                )
                result = SCOVA().fit(data, declaration, nuisance_predictions=nuisance)
                inference = result.infer(n_bootstrap=n_bootstrap)
                max_t_rejections += inference.global_test.max_t_p_value <= 0.05
                wald_rejections += inference.global_test.wald_p_value <= 0.05

                alternative_nuisance = NuisancePredictions(
                    simulation.propensity,
                    simulation.outcome_regression,
                    simulation.group_labels,
                )
                alternative = (
                    SCOVA()
                    .fit(
                        simulation.data,
                        declaration,
                        nuisance_predictions=alternative_nuisance,
                    )
                    .infer(n_bootstrap=n_bootstrap)
                )
                all_covered = True
                for contrast in alternative.contrasts:
                    left, right = (int(value[1:]) for value in contrast.name.split(" - "))
                    truth = float(
                        simulation.true_group_means[left] - simulation.true_group_means[right]
                    )
                    lower, upper = contrast.simultaneous_confidence_interval
                    all_covered &= lower <= truth <= upper
                simultaneous_coverage += all_covered
                any_contrast_power += alternative.global_test.max_t_p_value <= 0.05
            records.append(
                {
                    "overlap": overlap,
                    "n_groups": n_groups,
                    "repetitions": repetitions,
                    "n_bootstrap": n_bootstrap,
                    "max_t_fwer": max_t_rejections / repetitions,
                    "wald_size": wald_rejections / repetitions,
                    "simultaneous_coverage": simultaneous_coverage / repetitions,
                    "any_contrast_power": any_contrast_power / repetitions,
                }
            )
    payload = {
        "schema_version": 1,
        "benchmark": "stage2_fwer",
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1999)
    parser.add_argument("--output", type=Path, default=Path("benchmark_artifacts/stage2_fwer.json"))
    arguments = parser.parse_args()
    run_benchmark(arguments.repetitions, arguments.bootstrap, arguments.output)


if __name__ == "__main__":
    main()
