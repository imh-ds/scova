"""Show where SCOVA-CF helps in a nonlinear between-subjects design."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scova import ContrastSpec
from scova.cf import (
    SCOVACF,
    AnalysisMode,
    KnownAssignment,
    SCOVACFDeclaration,
    SCOVACFResult,
)


def simulate_nonlinear_trial(*, n: int = 1200, seed: int = 2) -> tuple[
    pd.DataFrame, np.ndarray
]:
    """Simulate mutually exclusive groups with nonlinear heterogeneous response surfaces."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    base = 1.4 * np.sin(1.3 * x[:, 0]) + 0.8 * (np.square(x[:, 1]) - 1) + 0.3 * x[:, 2]
    conditional_means = np.column_stack(
        (
            base,
            base + 0.8 + 1.0 * np.tanh(x[:, 0]),
            base - 0.35 + 0.9 * (np.square(x[:, 1]) - 1),
        )
    )
    group_codes = rng.integers(0, 3, size=n)
    outcome = conditional_means[np.arange(n), group_codes] + rng.normal(scale=1.0, size=n)
    labels = np.array(["g0", "g1", "g2"])
    data = pd.DataFrame(x, columns=["x1", "x2", "x3"])
    data["group"] = labels[group_codes]
    data["outcome"] = outcome
    return data, conditional_means.mean(axis=0)


def _declaration() -> SCOVACFDeclaration:
    return SCOVACFDeclaration(
        outcome="outcome",
        group="group",
        covariates=("x1", "x2", "x3"),
        mode=AnalysisMode.RANDOMIZED,
        scientific_question=(
            "What would the eligible population mean be if everyone were assigned to each group?"
        ),
        eligibility="All simulated study units",
        target_population="Eligible simulated study-unit population",
        group_definitions=(
            ("g0", "reference intervention"),
            ("g1", "intervention with an x1-heterogeneous response"),
            ("g2", "intervention with an x2-nonlinear response"),
        ),
        outcome_time="end of follow-up",
        outcome_units="points",
        covariate_rationales=(
            ("x1", "baseline nonlinear prognostic and effect-modifying factor"),
            ("x2", "baseline quadratic prognostic and effect-modifying factor"),
            ("x3", "baseline prognostic factor"),
        ),
        assignment=KnownAssignment(
            probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))
        ),
        contrasts=(
            ContrastSpec("g1 - g0", (("g1", 1.0), ("g0", -1.0))),
            ContrastSpec("g2 - g0", (("g2", 1.0), ("g0", -1.0))),
        ),
        n_splits=5,
        random_state=47,
    )


def run_demo(*, n: int = 1200, seed: int = 2) -> dict[str, Any]:
    """Run the comparison and return its machine-checkable summary."""
    data, target_means = simulate_nonlinear_trial(n=n, seed=seed)
    analysis = SCOVACF().analyze(data, _declaration())
    if not isinstance(analysis, SCOVACFResult):
        raise RuntimeError(f"SCOVA-CF refused the demonstration: {analysis.to_dict()}")
    raw_means = np.asarray(analysis.benchmarks["unadjusted"]["means"], dtype=float)
    linear_values = analysis.benchmarks["lin_interacted"]
    if linear_values["status"] != "complete":
        raise RuntimeError(f"Linear benchmark unavailable: {linear_values}")
    linear_means = np.asarray(linear_values["means"], dtype=float)
    cf_means = analysis.group_means
    errors = {
        "raw_anova_style": float(np.sqrt(np.mean(np.square(raw_means - target_means)))),
        "linear_ancova": float(np.sqrt(np.mean(np.square(linear_means - target_means)))),
        "scova_cf": float(np.sqrt(np.mean(np.square(cf_means - target_means)))),
    }
    return {
        "group_labels": list(analysis.group_labels),
        "true_counterfactual_population_means": target_means.tolist(),
        "raw_anova_style_means": raw_means.tolist(),
        "linear_ancova_means": linear_means.tolist(),
        "scova_cf_means": cf_means.tolist(),
        "rmse_against_truth": errors,
        "support_status": analysis.status.to_dict(),
        "claim_class": analysis.claim_class.value,
        "estimand_id": analysis.estimand_id,
        "known_assignment_used": bool(
            np.allclose(analysis.propensity_predictions, np.full((n, 3), 1 / 3))
        ),
        "core_benefits": [
            (
                "Every group is evaluated over the same eligible population rather than its "
                "sampled rows."
            ),
            (
                "Flexible cross-fitted outcome models can learn the sine, quadratic, and "
                "heterogeneous terms."
            ),
            (
                "Known randomization probabilities support residual correction without "
                "estimating assignment."
            ),
            (
                "Claim class, support status, estimand, contrasts, and limitations remain "
                "attached to output."
            ),
        ],
        "interpretation_caveat": (
            "This fixed simulation illustrates the intended nonlinear regime; SCOVA-CF is not "
            "guaranteed to outperform ANOVA or a well-specified ANCOVA in every sample."
        ),
    }


def main() -> None:
    summary = run_demo()
    print("True population counterfactual means:", summary["true_counterfactual_population_means"])
    print("Raw ANOVA-style means:            ", summary["raw_anova_style_means"])
    print("Linear ANCOVA means:               ", summary["linear_ancova_means"])
    print("SCOVA-CF means:                    ", summary["scova_cf_means"])
    print("RMSE against known truth:          ", summary["rmse_against_truth"])
    print("\nCore benefit flags:")
    for benefit in summary["core_benefits"]:
        print(" -", benefit)
    print("\nCaveat:", summary["interpretation_caveat"])
    print("\nGovernance:")
    print(" - claim class:", summary["claim_class"])
    print(" - support:", summary["support_status"])
    print(" - known assignment used:", summary["known_assignment_used"])


if __name__ == "__main__":
    main()
