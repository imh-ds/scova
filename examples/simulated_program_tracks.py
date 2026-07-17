"""Compare ANOVA, ANCOVA, and graph-conditional SCOVA on fictional program tracks.

The groups are fictional education-program tracks.  Baseline preparation has
modest track-specific shifts but common support.  It is a teaching example,
not a causal-data-generating recommendation.
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
    SCOVA,
    AnchoredBoundsDeclaration,
    ContrastSpec,
    DesignDeclaration,
    OutcomeFreeDesignData,
    SCOVADeclaration,
    SCOVADesign,
    SupportGeometryDeclaration,
)
from scova.experimental.gates import DiagnosticThresholds

GROUPS = ("standard", "bridge", "accelerated")
PAIRWISE = (("standard", "bridge"), ("standard", "accelerated"), ("bridge", "accelerated"))


def make_program_track_data(n_per_group: int = 240, seed: int = 71) -> pd.DataFrame:
    """Create bounded achievement scores with shared, shifted covariate support."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    preparation_locations = (-0.25, 0.0, 0.25)
    direct_track_effect = (0.0, 2.0, 4.0)
    for group, location, effect in zip(
        GROUPS, preparation_locations, direct_track_effect, strict=True
    ):
        preparation = rng.normal(location, 0.8, size=n_per_group)
        attendance = np.clip(0.55 * preparation + rng.normal(0, 0.65, n_per_group), -2.5, 2.5)
        score = 70 + 7 * preparation + 4 * attendance + effect + rng.normal(0, 3.5, n_per_group)
        parts.append(
            pd.DataFrame(
                {
                    "track": group,
                    "baseline_preparation": preparation,
                    "attendance_context": attendance,
                    "achievement_score": np.clip(score, 30, 100),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _ancova(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit conventional linear ANCOVA with standard track as the reference."""
    group = frame["track"].to_numpy()
    design = np.column_stack(
        (
            np.ones(len(frame)),
            group == "bridge",
            group == "accelerated",
            frame["baseline_preparation"].to_numpy(),
            frame["attendance_context"].to_numpy(),
        )
    ).astype(float)
    outcome = frame["achievement_score"].to_numpy()
    coefficient, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficient
    degrees_of_freedom = len(outcome) - design.shape[1]
    covariance = np.linalg.inv(design.T @ design) * (residual @ residual / degrees_of_freedom)
    critical = float(t.ppf(0.975, degrees_of_freedom))
    contrasts = {
        "standard - bridge": np.array([0, -1, 0, 0, 0], dtype=float),
        "standard - accelerated": np.array([0, 0, -1, 0, 0], dtype=float),
        "bridge - accelerated": np.array([0, 1, -1, 0, 0], dtype=float),
    }
    return {
        name: {
            "estimate": float(weight @ coefficient),
            "confidence_interval": [
                float(weight @ coefficient - critical * np.sqrt(weight @ covariance @ weight)),
                float(weight @ coefficient + critical * np.sqrt(weight @ covariance @ weight)),
            ],
        }
        for name, weight in contrasts.items()
    }


def _illustrative_thresholds() -> DiagnosticThresholds:
    """Calibrated illustrative gates; production analyses use released thresholds."""
    return DiagnosticThresholds(
        version="simulated-program-tracks-v1", calibrated=True, artifact_sha256="example",
        min_group_ess_warning=20, min_group_ess_refuse=10, min_target_ess_ratio_warning=0.1,
        min_target_ess_ratio_refuse=0.05, max_influence_share_warning=0.9,
        max_influence_share_refuse=0.99, max_weight_concentration_warning=0.5,
        max_weight_concentration_refuse=0.9, min_propensity_q01_warning=0.001,
        min_propensity_q01_refuse=0.0001, max_calibration_error_warning=1,
        max_calibration_error_refuse=2, max_balance_warning=2, max_balance_refuse=5,
        max_crossfit_instability_warning=1, max_crossfit_instability_refuse=2,
    )


def run(seed: int = 71) -> dict[str, Any]:
    frame = make_program_track_data(seed=seed)
    raw = {
        group: float(frame.loc[frame["track"] == group, "achievement_score"].mean())
        for group in GROUPS
    }
    anova = f_oneway(*(frame.loc[frame["track"] == group, "achievement_score"] for group in GROUPS))
    scova_declaration = SCOVADeclaration(
        outcome="achievement_score",
        group="track",
        covariates=("baseline_preparation", "attendance_context"),
        n_splits=3,
        random_state=seed,
        contrasts=tuple(
            ContrastSpec(f"{left} - {right}", ((left, 1.0), (right, -1.0)))
            for left, right in PAIRWISE
        ),
    )
    fixed = SCOVA().fit(frame, scova_declaration)
    fixed_contrasts = {
        name: {
            "estimate": contrast.estimate,
            "confidence_interval": list(contrast.confidence_interval),
        }
        for name, contrast in fixed.contrasts.items()
    }
    design_data = OutcomeFreeDesignData.from_arrays(
        frame.loc[:, ["baseline_preparation", "attendance_context"]].to_numpy(),
        frame["track"].tolist(),
        row_ids=range(len(frame)),
    )
    locked_declaration = DesignDeclaration(
        group="track",
        covariates=("baseline_preparation", "attendance_context"),
        n_splits=3,
        random_state=seed,
        lambdas=(0.0, 1.0),
        candidate_subsets=(GROUPS,),
        anchored_bounds=AnchoredBoundsDeclaration(
            30,
            100,
            support_geometry=SupportGeometryDeclaration(gamma_grid=(0.0, 0.5, 1.0, 2.0)),
        ),
    )
    engine = SCOVADesign(thresholds=_illustrative_thresholds())
    locked = engine.prepare_design(design_data, locked_declaration)
    row_ids = locked.lock.estimation_row_ids
    outcomes = frame["achievement_score"].to_numpy()[list(row_ids)]
    anchored = engine.analyze_anchored_bounds(locked, outcomes, row_ids=row_ids)
    lipschitz = engine.analyze_lipschitz_anchors(locked, outcomes, row_ids=row_ids)
    return {
        "constructs": {
            "group": "fictional program track",
            "outcome": "bounded achievement score (30 to 100)",
            "covariates": ["baseline preparation", "attendance context"],
            "support_design": "all three tracks share support, with modest baseline shifts",
        },
        "anova": {
            "raw_group_means": raw,
            "f_statistic": float(anova.statistic),
            "p_value": float(anova.pvalue),
        },
        "ancova": _ancova(frame),
        "scova_fixed_target": fixed_contrasts,
        "scova_graph": {
            "supported_pairwise_edges": [list(pair) for pair in locked.graph.supported_edges],
            "supported_kway_hyperedges": [
                list(edge.groups) for edge in locked.graph.supported_maximal_hyperedges
            ],
            "refused_pairs": [
                {"groups": list(edge.groups), "reasons": list(edge.refusal_reasons)}
                for edge in locked.graph.refused_edges
            ],
        },
        "scova_b1_anchor": anchored.report(),
        "scova_b2_experimental": lipschitz.report(),
        "interpretation": (
            "ANOVA reports raw group differences; linear ANCOVA returns all requested adjusted "
            "contrasts; SCOVA displays the outcome-blind support graph first and only anchors "
            "graph-supported pairs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--output", type=Path, default=Path("release/artifacts/simulated-program-tracks.json")
    )
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
