"""Run an experimental, graph-conditional Stage 5B B2 anchor analysis."""

from __future__ import annotations

from scova import (
    AnchoredBoundsDeclaration,
    DesignDeclaration,
    OutcomeFreeDesignData,
    SCOVADesign,
    SupportGeometryDeclaration,
)
from scova.experimental.gates import DiagnosticThresholds
from scova.simulate import generate_data


def simulation_thresholds() -> DiagnosticThresholds:
    """Permissive, calibrated gates solely for this reproducible example."""
    return DiagnosticThresholds(
        version="stage5b-example", calibrated=True, artifact_sha256="example",
        min_group_ess_warning=1, min_group_ess_refuse=0, min_target_ess_ratio_warning=0,
        min_target_ess_ratio_refuse=0, max_influence_share_warning=1,
        max_influence_share_refuse=1, max_weight_concentration_warning=1,
        max_weight_concentration_refuse=1, min_propensity_q01_warning=1e-12,
        min_propensity_q01_refuse=1e-14, max_calibration_error_warning=1,
        max_calibration_error_refuse=1, max_balance_warning=1_000,
        max_balance_refuse=10_000, max_crossfit_instability_warning=1,
        max_crossfit_instability_refuse=1,
    )


def main() -> None:
    simulation = generate_data("observational", n=600, seed=51)
    frame = simulation.data
    declaration = DesignDeclaration(
        group="group",
        covariates=("x1", "x2", "x3"),
        n_splits=2,
        random_state=51,
        lambdas=(0.0, 1.0),
        anchored_bounds=AnchoredBoundsDeclaration(
            -100,
            100,
            support_geometry=SupportGeometryDeclaration(gamma_grid=(0.0, 0.5, 1.0, 2.0)),
        ),
    )
    data = OutcomeFreeDesignData.from_arrays(
        frame.loc[:, ["x1", "x2", "x3"]].to_numpy(),
        frame["group"].tolist(),
        row_ids=range(len(frame)),
    )
    engine = SCOVADesign(thresholds=simulation_thresholds())
    locked = engine.prepare_design(data, declaration)
    row_ids = locked.lock.estimation_row_ids
    result = engine.analyze_lipschitz_anchors(
        locked, frame["outcome"].to_numpy()[list(row_ids)], row_ids=row_ids
    )
    print(result.verdict)  # Always experimental for B2 in this release.
    print(result.transport_diagnostics)
    for contrast in result.contrasts:
        print(contrast.name, contrast.gamma_grid, contrast.confidence_intervals)

    # A finite-range violation returns a typed refusal rather than widening the lock.
    refused = engine.analyze_lipschitz_anchors(
        locked, [1_000.0] * len(row_ids), row_ids=row_ids
    )
    print(refused.verdict, refused.refused)


if __name__ == "__main__":
    main()
