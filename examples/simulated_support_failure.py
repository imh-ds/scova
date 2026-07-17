"""Show why a graph-conditional SCOVA refusal can be safer than ANCOVA.

This fictional example deliberately violates overlap: the two tracks occupy
disjoint baseline-preparation ranges.  Their potential outcome rule is the
same nonlinear curve, so the simulated direct track effect is zero at every
preparation value.  A linear ANCOVA nevertheless reports a precise, nonzero
coefficient because it must extrapolate its linear trend across the gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, t

from scova import (
    AnchoredBoundsDeclaration,
    DesignDeclaration,
    OutcomeFreeDesignData,
    SCOVADesign,
    SupportGeometryDeclaration,
)
from scova.experimental.gates import DiagnosticThresholds


def make_data(n_per_group: int = 400, seed: int = 17) -> pd.DataFrame:
    """Create two disjoint covariate populations with no direct group effect."""
    rng = np.random.default_rng(seed)
    standard = rng.uniform(-2.0, -0.4, n_per_group)
    accelerated = rng.uniform(0.4, 1.2, n_per_group)
    preparation = np.concatenate((standard, accelerated))
    track = np.repeat(("standard", "accelerated"), n_per_group)
    # The same potential-outcome curve applies to both tracks.  No group term
    # appears here: the true conditional contrast is exactly zero everywhere.
    score = np.clip(65 + 10 * preparation**2 + rng.normal(0, 3, len(preparation)), 30, 100)
    return pd.DataFrame(
        {"track": track, "baseline_preparation": preparation, "achievement_score": score}
    )


def ancova(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit the intentionally misspecified conventional linear ANCOVA."""
    accelerated = (frame["track"].to_numpy() == "accelerated").astype(float)
    design = np.column_stack(
        (np.ones(len(frame)), accelerated, frame["baseline_preparation"].to_numpy())
    )
    outcome = frame["achievement_score"].to_numpy()
    coefficient, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficient
    degrees_of_freedom = len(outcome) - design.shape[1]
    covariance = np.linalg.inv(design.T @ design) * (residual @ residual / degrees_of_freedom)
    standard_error = float(np.sqrt(covariance[1, 1]))
    critical = float(t.ppf(0.975, degrees_of_freedom))
    return {
        "contrast": "accelerated - standard",
        "estimate": float(coefficient[1]),
        "confidence_interval": [
            float(coefficient[1] - critical * standard_error),
            float(coefficient[1] + critical * standard_error),
        ],
        "model": "achievement_score ~ track + baseline_preparation (linear)",
    }


def thresholds() -> DiagnosticThresholds:
    """Illustrative, versioned gates that expose the deliberate support failure."""
    return DiagnosticThresholds(
        version="simulated-support-failure-v1",
        calibrated=True,
        artifact_sha256="example",
        min_group_ess_warning=20,
        min_group_ess_refuse=10,
        min_target_ess_ratio_warning=0.1,
        min_target_ess_ratio_refuse=0.05,
        max_influence_share_warning=0.9,
        max_influence_share_refuse=0.99,
        max_weight_concentration_warning=0.5,
        max_weight_concentration_refuse=0.9,
        min_propensity_q01_warning=0.01,
        min_propensity_q01_refuse=0.001,
        max_calibration_error_warning=1,
        max_calibration_error_refuse=2,
        max_balance_warning=2,
        max_balance_refuse=5,
        max_crossfit_instability_warning=1,
        max_crossfit_instability_refuse=2,
    )


def run(seed: int = 17) -> dict[str, Any]:
    frame = make_data(seed=seed)
    group_means = {
        group: float(frame.loc[frame["track"] == group, "achievement_score"].mean())
        for group in ("standard", "accelerated")
    }
    anova = f_oneway(
        frame.loc[frame["track"] == "standard", "achievement_score"],
        frame.loc[frame["track"] == "accelerated", "achievement_score"],
    )
    data = OutcomeFreeDesignData.from_arrays(
        frame[["baseline_preparation"]].to_numpy(),
        frame["track"].tolist(),
        row_ids=range(len(frame)),
    )
    declaration = DesignDeclaration(
        group="track",
        covariates=("baseline_preparation",),
        n_splits=3,
        random_state=seed,
        lambdas=(0.0, 1.0),
        anchored_bounds=AnchoredBoundsDeclaration(
            30,
            100,
            support_geometry=SupportGeometryDeclaration(gamma_grid=(0.0, 0.5, 1.0, 2.0)),
        ),
    )
    engine = SCOVADesign(thresholds=thresholds())
    locked = engine.prepare_design(data, declaration)
    row_ids = locked.lock.estimation_row_ids
    outcome = frame["achievement_score"].to_numpy()[list(row_ids)]
    b1 = engine.analyze_anchored_bounds(locked, outcome, row_ids=row_ids)
    b2 = engine.analyze_lipschitz_anchors(locked, outcome, row_ids=row_ids)
    return {
        "constructs": {
            "group": "fictional program track",
            "covariate": "baseline preparation",
            "outcome": "bounded achievement score (30 to 100)",
            "support_design": "standard: [-2.0, -0.4]; accelerated: [0.4, 1.2]",
            "simulation_truth": (
                "The direct accelerated-minus-standard contrast is zero at every preparation value."
            ),
        },
        "anova": {
            "raw_group_means": group_means,
            "f_statistic": float(anova.statistic),
            "p_value": float(anova.pvalue),
        },
        "linear_ancova": ancova(frame),
        "scova_graph": {
            "supported_pairwise_edges": [list(edge) for edge in locked.graph.supported_edges],
            "refused_pairs": [
                {"groups": list(edge.groups), "reasons": list(edge.refusal_reasons)}
                for edge in locked.graph.refused_edges
            ],
        },
        "scova_b1_anchor": b1.report(),
        "scova_b2_experimental": b2.report(),
        "interpretation": (
            "The ANCOVA coefficient is an extrapolation from disjoint covariate ranges. "
            "The Stage-4 graph refuses the pair, so B1 and B2 correctly produce no anchor claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("release/artifacts/simulated-support-failure.json")
    )
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
