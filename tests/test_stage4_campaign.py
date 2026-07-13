import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import stage4_campaign
from benchmarks.summarize_stage4 import summarize
from scripts import check_stage4_release
from scripts.verify_stage4_calibration import verify as verify_calibration
from scripts.verify_stage4_smoke import verify


def specification() -> dict:
    return {
        "protocol": "stage4-test",
        "frozen": True,
        "target_path": {"lambdas": [0.0, 1.0]},
        "seed_namespaces": {"smoke": 1, "validation": 2},
        "directional_pass_criteria": {
            "conditional_fwer_upper_bound_max": 1.0,
            "simultaneous_coverage_min": 0.0,
            "simultaneous_coverage_max": 1.0,
            "minimum_strong_complete_graph_rate": 0.0,
            "maximum_false_global_claim_rate_pairwise_only": 1.0,
            "minimum_supported_pairwise_edge_rate_pairwise_only": 0.0,
            "minimum_rare_group_refusal_rate": 0.0,
            "post_lock_confirmatory_mutation_rate": 0.0,
            "minimum_accepted_repetitions_per_cell": 1,
            "confidence_level": 0.95,
        },
        "tiers": {
            "directional_validation": {
                "cells": 5,
                "repetitions": 1,
                "bootstrap": 1,
                "scenarios": [
                    "global_null",
                    "sparse_pairwise_signal",
                    "strong_complete_graph",
                    "rare_group",
                    "pairwise_without_kway",
                ],
                "n": [30],
                "n_groups": [3],
                "p": [3],
                "seed_namespace": "validation",
            }
        },
    }


def test_v2_release_specification_is_frozen() -> None:
    path = Path("benchmarks/specs/stage4_graph_release_v2.json")
    release_specification = json.loads(path.read_text(encoding="utf-8"))
    assert release_specification["protocol"] == "stage4-graph-firewall-v2"
    assert release_specification["status"] == "frozen"
    assert release_specification["frozen"] is True


def test_v3_release_specification_is_frozen_with_fresh_seeds() -> None:
    release = stage4_campaign.load_specification(
        Path("benchmarks/specs/stage4_graph_release_v3.json")
    )
    assert release["protocol"] == "stage4-graph-firewall-v3"
    assert release["frozen"] is True
    assert release["metric_contract"] == "stage4-v3"
    assert release["seed_namespaces"]["validation"] != 222000000
    assert all(
        cell.expected_outcome == "inferential"
        for cell in stage4_campaign.frozen_cells("directional_validation", release)
        if cell.scenario == "strong_complete_graph"
    )


def test_v4_catalog_has_no_nonrare_low_sample_eight_group_cells() -> None:
    release = stage4_campaign.load_specification(
        Path("benchmarks/specs/stage4_graph_release_v4.json")
    )
    cells = stage4_campaign.frozen_cells("directional_validation", release)
    assert release["protocol"] == "stage4-graph-firewall-v4"
    assert release["seed_namespaces"]["validation"] == 422000000
    assert release["catalog_definition_sha256"]
    assert all(
        cell.scenario == "rare_group" or cell.expected_outcome == "inferential"
        for cell in cells
    )
    assert all(
        not (cell.scenario != "rare_group" and cell.n == 500 and cell.n_groups == 8)
        for cell in cells
    )


def test_catalog_sharding_and_seed_truth_are_deterministic() -> None:
    cells = stage4_campaign.frozen_cells("directional_validation", specification())
    assert [cell.id for cell in cells] == [
        f"directional_validation-{index:03d}" for index in range(5)
    ]
    assert set(stage4_campaign.shard_items(cells, 1, 0, 2)).isdisjoint(
        stage4_campaign.shard_items(cells, 1, 1, 2)
    )
    pairwise = stage4_campaign.generate_stage4_data(cells[-1], 10)
    assert pairwise.true_hyperedges == ()
    assert pairwise.true_pairs
    rare = stage4_campaign.generate_stage4_data(cells[3], 10)
    assert all("g2" not in pair for pair in rare.true_pairs)
    assert rare.groups.count("g2") == 2


def test_pairwise_only_dgp_has_all_pairs_but_no_joint_hyperedge() -> None:
    cell = stage4_campaign.Stage4Cell("pairwise", "pairwise_without_kway", n=300, n_groups=3, p=5)
    data = stage4_campaign.generate_stage4_data(cell, seed=7)
    assert data.true_pairs == (("g0", "g1"), ("g0", "g2"), ("g1", "g2"))
    assert data.true_hyperedges == ()
    support_by_group = {
        group: set(data.covariates[np.array(data.groups) == group, 0])
        for group in ("g0", "g1", "g2")
    }
    assert all(
        support_by_group[first].intersection(support_by_group[second])
        for first, second in data.true_pairs
    )
    assert not set.intersection(*support_by_group.values())


def test_strong_complete_graph_dgp_is_balanced_and_complete() -> None:
    cell = stage4_campaign.Stage4Cell("strong", "strong_complete_graph", 400, 4, 5)
    data = stage4_campaign.generate_stage4_data(cell, seed=7)
    assert {group: data.groups.count(group) for group in set(data.groups)} == {
        "g0": 100,
        "g1": 100,
        "g2": 100,
        "g3": 100,
    }
    assert data.true_pairs == tuple(itertools.combinations(("g0", "g1", "g2", "g3"), 2))


def _shard_payload(spec: dict, records: list[dict]) -> dict:
    payload = {
        "schema_version": 3 if spec.get("metric_contract") in {"stage4-v3", "stage4-v4"} else 2,
        "protocol": "stage4-test",
        "metric_contract": spec.get("metric_contract", "legacy"),
        "tier": "directional_validation",
        "specification_sha256": "spec",
        "catalog_sha256": "catalog",
        "threshold_artifact_sha256": "threshold",
        "shard_count": 2,
        "records": records,
    }
    payload["sha256"] = stage4_campaign._digest(payload)
    return payload


def test_aggregation_rejects_missing_duplicate_and_bad_checksum(tmp_path: Path) -> None:
    spec = specification()
    cells = stage4_campaign.frozen_cells("directional_validation", spec)
    records = []
    for index, cell in enumerate(cells):
        records.append(
            {
                "cell_index": index,
                "repetition": 0,
                "cell": {"id": cell.id, "scenario": cell.scenario},
                "status": "completed",
                "accepted": True,
                "simultaneous_coverage": True,
                "any_rejection": False,
                "selected_edges": [["g0", "g1"]],
                "selected_hyperedges": [],
                "post_lock_mutation_rejected": True,
            }
        )
    path = tmp_path / "good.json"
    path.write_text(json.dumps(_shard_payload(spec, records)), encoding="utf-8")
    result = summarize([path], spec, "directional_validation")
    assert result["status"] == "pass"
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(_shard_payload(spec, records[:-1])), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        summarize([missing], spec, "directional_validation")
    with pytest.raises(ValueError, match="duplicate"):
        summarize([path, path], spec, "directional_validation")
    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["sha256"] = "bad"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        summarize([path], spec, "directional_validation")


def test_v3_summary_excludes_declared_refusals_from_inferential_metrics(tmp_path: Path) -> None:
    spec = specification()
    spec["metric_contract"] = "stage4-v3"
    spec["catalogs"] = {
        "directional_validation": [
            {
                "id": f"directional_validation-{index:03d}",
                "scenario": scenario,
                "n": 30,
                "n_groups": 3,
                "p": 3,
                "expected_outcome": "refusal" if scenario == "rare_group" else "inferential",
            }
            for index, scenario in enumerate(spec["tiers"]["directional_validation"]["scenarios"])
        ]
    }
    cells = stage4_campaign.frozen_cells("directional_validation", spec)
    records = []
    for index, cell in enumerate(cells):
        refusal = cell.expected_outcome == "refusal"
        records.append(
            {
                "cell_index": index,
                "repetition": 0,
                "cell": {
                    "id": cell.id,
                    "scenario": cell.scenario,
                    "expected_outcome": cell.expected_outcome,
                },
                "status": "completed",
                "accepted": not refusal,
                "expected_refusal": "insufficient_per_split_group_count" if refusal else None,
                "simultaneous_coverage": not refusal,
                "any_rejection": False,
                "selected_edges": [["g0", "g1"]],
                "selected_hyperedges": [],
                "complete_graph_recovered": cell.scenario == "strong_complete_graph",
                "post_lock_mutation_rejected": None if refusal else True,
            }
        )
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(_shard_payload(spec, records)), encoding="utf-8")
    result = summarize([path], spec, "directional_validation")
    assert result["metrics"]["minimum_rare_group_refusal_rate"]["denominator"] == 1
    assert result["metrics"]["conditional_fwer_upper_bound_max"]["denominator"] == 1
    assert result["criteria"]["strong_complete_graph:directional_validation-002"]


def test_v4_summary_reports_each_cell_acceptance_gate(tmp_path: Path) -> None:
    spec = specification()
    spec["metric_contract"] = "stage4-v4"
    spec["directional_pass_criteria"]["minimum_accepted_repetitions_per_cell"] = 2
    spec["tiers"]["directional_validation"]["repetitions"] = 2
    spec["tiers"]["directional_validation"]["cells"] = 1
    spec["tiers"]["directional_validation"]["scenarios"] = ["global_null"]
    spec["catalogs"] = {
        "directional_validation": [
            {
                "id": "directional_validation-000",
                "scenario": "global_null",
                "n": 30,
                "n_groups": 2,
                "p": 3,
                "expected_outcome": "inferential",
            }
        ]
    }
    cells = stage4_campaign.frozen_cells("directional_validation", spec)
    record = {
        "cell_index": 0,
        "repetition": 0,
        "cell": {**stage4_campaign.asdict(cells[0])},
        "status": "completed",
        "accepted": True,
        "simultaneous_coverage": True,
        "any_rejection": False,
        "selected_edges": [["g0", "g1"]],
        "selected_hyperedges": [],
        "complete_graph_recovered": True,
        "post_lock_mutation_rejected": True,
    }
    refused = {**record, "repetition": 1, "accepted": False, "simultaneous_coverage": None}
    payload = _shard_payload(spec, [record, refused])
    path = tmp_path / "v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = summarize([path], spec, "directional_validation")
    name = "accepted_repetitions:directional_validation-000"
    assert result["criteria"][name] is False
    assert result["metrics"][name]["numerator"] == 1


def test_v4_release_checker_uses_per_cell_criteria(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passing V4 artifact must not be checked against legacy aggregate keys."""
    root = tmp_path
    spec = {
        "protocol": "stage4-graph-firewall-v4",
        "metric_contract": "stage4-v4",
        "directional_pass_criteria": {
            "minimum_strong_complete_graph_rate": 0.8,
        },
    }
    spec_path = root / "stage4-v4.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    artifact_dir = root / "release" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "stage3-directional-thresholds.json").write_text(
        "{}", encoding="utf-8"
    )
    evidence = {
        "protocol": spec["protocol"],
        "threshold_artifact_sha256": "threshold",
        "status": "pass",
        "criteria": {
            "strong_complete_graph:directional_validation-002": True,
            "robustness:strong_complete_graph:directional_robustness-002": True,
        },
        "metrics": {},
    }
    (artifact_dir / "stage4-evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )

    class CalibratedThresholds:
        calibrated = True
        artifact_sha256 = "threshold"

    monkeypatch.setattr(
        check_stage4_release.DiagnosticThresholds,
        "from_calibration_artifact",
        lambda _: CalibratedThresholds(),
    )
    assert check_stage4_release.blocking_reasons(root, spec_path) == []


def test_campaign_requires_stage3_thresholds_and_rejects_mixed_threshold_shards(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="threshold artifact"):
        stage4_campaign.run_campaign(
            tier="calibration",
            thresholds_path=None,
            shard_index=0,
            shard_count=64,
            output=tmp_path / "never-written.json",
        )
    spec = specification()
    cells = stage4_campaign.frozen_cells("directional_validation", spec)
    records = [
        {
            "cell_index": index,
            "repetition": 0,
            "cell": {"id": cell.id, "scenario": cell.scenario},
            "status": "completed",
            "accepted": True,
            "simultaneous_coverage": True,
            "any_rejection": False,
            "selected_edges": [],
            "selected_hyperedges": [],
            "post_lock_mutation_rejected": True,
        }
        for index, cell in enumerate(cells)
    ]
    first = _shard_payload(spec, records[:2])
    second = _shard_payload(spec, records[2:])
    second["threshold_artifact_sha256"] = "other"
    second["sha256"] = stage4_campaign._digest(
        {key: value for key, value in second.items() if key != "sha256"}
    )
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        summarize([first_path, second_path], spec, "directional_validation")


def test_smoke_verifier_requires_complete_accepted_lock_safe_records(tmp_path: Path) -> None:
    spec = {
        **specification(),
        "catalogs": {
            "engineering_smoke": [
                {
                    "id": "engineering_smoke-000",
                    "scenario": "strong_complete_graph",
                    "n": 20,
                    "n_groups": 3,
                    "p": 3,
                    "expected_outcome": "inferential",
                },
                {
                    "id": "engineering_smoke-001",
                    "scenario": "pairwise_without_kway",
                    "n": 20,
                    "n_groups": 3,
                    "p": 3,
                    "expected_outcome": "inferential",
                },
                {
                    "id": "engineering_smoke-002",
                    "scenario": "rare_group",
                    "n": 20,
                    "n_groups": 3,
                    "p": 3,
                    "expected_outcome": "refusal",
                },
                {
                    "id": "engineering_smoke-003",
                    "scenario": "global_null",
                    "n": 20,
                    "n_groups": 3,
                    "p": 3,
                    "expected_outcome": "inferential",
                },
            ]
        },
        "tiers": {
            "engineering_smoke": {
                "cells": 4,
                "repetitions": 5,
                "bootstrap": 1,
                "scenarios": [
                    "strong_complete_graph",
                    "pairwise_without_kway",
                    "rare_group",
                    "global_null",
                ],
                "n": [20],
                "n_groups": [3],
                "p": [3],
                "seed_namespace": "smoke",
            }
        },
    }
    cells = stage4_campaign.frozen_cells("engineering_smoke", spec)
    paths = []
    for shard in range(4):
        records = []
        for cell_index, repetition in stage4_campaign.shard_items(cells, 5, shard, 4):
            rare = cells[cell_index].expected_outcome == "refusal"
            records.append(
                {
                    "cell_index": cell_index,
                    "repetition": repetition,
                    "status": "completed",
                    "accepted": not rare,
                    "expected_refusal": "insufficient_per_split_group_count" if rare else None,
                    "post_lock_mutation_rejected": None if rare else True,
                }
            )
        payload = {
            "tier": "engineering_smoke",
            "shard_count": 4,
            "records": records,
        }
        payload["sha256"] = stage4_campaign._digest(payload)
        path = tmp_path / f"smoke-{shard}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    assert verify(paths, spec) == {"records": 20, "cells": 4}
    invalid = json.loads(paths[0].read_text(encoding="utf-8"))
    invalid["records"][0]["accepted"] = False
    invalid["sha256"] = stage4_campaign._digest(
        {key: value for key, value in invalid.items() if key != "sha256"}
    )
    paths[0].write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out inference"):
        verify(paths, spec)


def test_rare_group_count_refusal_is_completed_campaign_evidence() -> None:
    cell = stage4_campaign.Stage4Cell("rare", "rare_group", 10, 3, 2)
    error = ValueError("each design-split group requires at least n_splits observations")
    assert stage4_campaign._is_expected_rare_group_refusal(cell, error)
    record = stage4_campaign._expected_rare_group_refusal(error)
    assert record["status"] == "completed"
    assert record["expected_refusal"] == "insufficient_per_split_group_count"


def test_calibration_verifier_rejects_failed_records(tmp_path: Path) -> None:
    spec = specification()
    spec["tiers"]["calibration"] = {
        **spec["tiers"]["directional_validation"],
        "cells": 5,
        "repetitions": 1,
        "seed_namespace": "validation",
    }
    cells = stage4_campaign.frozen_cells("calibration", spec)
    records = [
        {
            "cell_index": index,
            "repetition": 0,
            "status": "completed",
            "expected_refusal": "insufficient_per_split_group_count"
            if cell.scenario == "rare_group"
            else None,
        }
        for index, cell in enumerate(cells)
    ]
    payload = {
        "schema_version": 2,
        "protocol": "stage4-test",
        "tier": "calibration",
        "specification_sha256": "spec",
        "catalog_sha256": "catalog",
        "threshold_artifact_sha256": "threshold",
        "shard_count": 2,
        "records": records,
    }
    payload["sha256"] = stage4_campaign._digest(payload)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_calibration([path], spec, shard_count=2)["records"] == 5
    failed = json.loads(path.read_text(encoding="utf-8"))
    failed["records"][0]["status"] = "failed"
    failed["sha256"] = stage4_campaign._digest(
        {key: value for key, value in failed.items() if key != "sha256"}
    )
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="did not complete"):
        verify_calibration([path], spec, shard_count=2)
