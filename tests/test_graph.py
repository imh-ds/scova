import json
from hashlib import sha256

import pytest

from scova import (
    DesignDeclaration,
    PairwiseDiagnosticInput,
    SubsetDiagnosticInput,
    build_comparability_graph,
    build_pairwise_comparability_graph,
)
from scova.experimental.gates import DiagnosticThresholds


def locked_thresholds() -> DiagnosticThresholds:
    artifact = {
        "version": "stage3-locked-test-v1",
        "calibrated": True,
        "pass_profile": {
            "min_group_ess": 50,
            "min_target_ess_ratio": 0.25,
            "max_influence_share": 0.4,
            "max_weight_concentration": 0.1,
            "min_propensity_q01": 0.005,
            "max_calibration_error": 0.1,
            "max_balance": 0.25,
            "max_crossfit_instability": 0.3,
        },
        "warning_floor_profile": {
            "min_group_ess": 25,
            "min_target_ess_ratio": 0.15,
            "max_influence_share": 0.6,
            "max_weight_concentration": 0.15,
            "min_propensity_q01": 0.0025,
            "max_calibration_error": 0.15,
            "max_balance": 0.35,
            "max_crossfit_instability": 0.4,
        },
    }
    artifact["sha256"] = sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()
    return DiagnosticThresholds.from_calibration_artifact(artifact)


def diagnostic_input(*, weak_all: bool = False) -> PairwiseDiagnosticInput:
    ess = [100.0, 100.0, 100.0]
    if weak_all:
        ess = [20.0, 20.0, 20.0]
    return PairwiseDiagnosticInput(
        group_labels=("a", "b", "c"),
        diagnostics={
            "path_gate_grid": {
                "schema_version": 1,
                "lambdas": [0.0, 0.5, 1.0],
                "target_ess_ratio": [0.8, 0.8, 0.8],
                "group_effective_sample_size": [[value, value, value] for value in ess],
                "target_weight_concentration": [0.02, 0.02, 0.02],
                "maximum_weighted_covariate_imbalance": [
                    [0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05],
                ],
                "normalization_finite": [True, True, True],
            },
            "propensity_quantiles": {
                "a": {"q01": 0.1},
                "b": {"q01": 0.1},
                "c": {"q01": 0.1},
            },
            "propensity_calibration": {"worst_class_expected_calibration_error": 0.01},
            "crossfit_instability": 0.01,
        },
    )


def test_pairwise_graph_retains_grid_evidence_and_refusals() -> None:
    declaration = DesignDeclaration("group", ("x1",), lambdas=(0.0, 0.5, 1.0))
    graph = build_pairwise_comparability_graph(
        declaration,
        ("c", "a", "b"),
        {
            ("a", "b"): diagnostic_input(),
            ("a", "c"): diagnostic_input(weak_all=True),
        },
        thresholds=locked_thresholds(),
    )
    assert graph.group_labels == ("a", "b", "c")
    assert graph.supported_edges == (("a", "b"),)
    assert graph.edge_for("b", "a").supported_lambdas == (0.0, 0.5, 1.0)
    refused = graph.edge_for("a", "c")
    assert not refused.supported
    assert "no declared lambda" in refused.refusal_reasons[0]
    assert "not supplied" in graph.edge_for("b", "c").refusal_reasons[0]


def test_pairwise_graph_rejects_incompatible_diagnostics() -> None:
    declaration = DesignDeclaration("group", ("x1",), lambdas=(0.0, 0.5, 1.0))
    bad = diagnostic_input()
    bad.diagnostics["path_gate_grid"]["lambdas"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="lambda grid"):
        build_pairwise_comparability_graph(
            declaration,
            ("a", "b"),
            {("a", "b"): bad},
            thresholds=locked_thresholds(),
        )


def test_hyperedges_are_jointly_supported_and_not_implied_by_a_clique() -> None:
    declaration = DesignDeclaration(
        "group", ("x1",), lambdas=(0.0, 0.5, 1.0), candidate_subsets=(("a", "b", "c"),)
    )
    pairs = {
        ("a", "b"): diagnostic_input(),
        ("a", "c"): diagnostic_input(),
        ("b", "c"): diagnostic_input(),
    }
    refused = diagnostic_input(weak_all=True)
    graph = build_comparability_graph(
        declaration,
        ("c", "a", "b"),
        pairs,
        subset_diagnostics={
            ("a", "b", "c"): SubsetDiagnosticInput(refused.group_labels, refused.diagnostics)
        },
        thresholds=locked_thresholds(),
    )
    assert graph.maximal_pairwise_cliques == (("a", "b", "c"),)
    assert not graph.supported_maximal_hyperedges
    assert not graph.hyperedge_for(("c", "b", "a")).supported
