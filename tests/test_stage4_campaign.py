import json
from pathlib import Path

import pytest

from benchmarks import stage4_campaign
from benchmarks.summarize_stage4 import summarize
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


def _shard_payload(spec: dict, records: list[dict]) -> dict:
    payload = {
        "schema_version": 2,
        "protocol": "stage4-test",
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
            rare = cells[cell_index].scenario == "rare_group"
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
