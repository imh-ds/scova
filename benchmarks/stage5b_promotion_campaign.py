"""Scheduled factorial evidence for the Stage 5B B2 promotion audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from benchmarks.stage5b_campaign import _perturbation_harness, _run, _thresholds
from scova.anchor import bounded_pairwise_anchor, lipschitz_pairwise_anchor


def _coverage(records: list[dict[str, Any]], field: str) -> float:
    values = [value for record in records for value in record[field]]
    return float(np.mean(values))


def _boundary_block_is_recorded() -> bool:
    propensity = np.full((6, 2), 0.5)
    bounded = bounded_pairwise_anchor(
        groups=("a", "b"), group_codes=np.array([0, 1, 0, 1, 0, 1]),
        outcomes=np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6]), propensity=propensity,
        outcome_predictions=np.full((6, 2), 0.5), active_codes=(0, 1), outcome_lower=0,
        outcome_upper=1, confidence_level=0.95,
    )
    result = lipschitz_pairwise_anchor(
        bounded=bounded, propensity=propensity, active_codes=(0, 1), gamma_grid=np.array([0.0]),
        smooth_distances=np.zeros((6, 2)), reference_predictions=np.ones((6, 2)),
        outcome_lower=0, outcome_upper=1, confidence_level=0.95,
    )
    return result.inference_status == "blocked-boundary"


def _transport_refusal_is_recorded() -> bool:
    settings = replace(
        _thresholds(), min_group_ess_warning=20_000, min_group_ess_refuse=10_000
    )
    result, _ = _run(5_600_000, 300, threshold_settings=settings)
    return result["verdict"] == "refused"


def _profile_records(repetitions: int) -> dict[str, list[dict[str, Any]]]:
    profiles: dict[str, Any | None] = {
        "default": None,
        "random_forest": RandomForestRegressor(
            n_estimators=20, min_samples_leaf=5, random_state=5_700_000, n_jobs=1
        ),
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for offset, (name, model) in enumerate(profiles.items()):
        records[name] = [
            _run(5_700_000 + offset * repetitions + index, 300, outcome_model=model)[0]
            for index in range(repetitions)
        ]
    return records


def run(specification: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    repetitions = int(specification["repetitions_per_arm"])
    records = _profile_records(repetitions)
    exact = {name: _coverage(values, "covered") for name, values in records.items()}
    conservative = {
        name: _coverage(values, "conservative_covered") for name, values in records.items()
    }
    midpoint_means = {
        name: float(np.mean([value for item in values for value in item["endpoint_midpoints"]]))
        for name, values in records.items()
    }
    pairwise, _ = _run(5_800_000, 300, pairwise_without_kway=True)
    violation = [_run(5_900_000 + index, 300, violation=True)[0] for index in range(32)]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    minimum = float(specification["minimum_coverage"])
    criteria = {
        "audit_manifest_valid": (
            audit.get("protocol") == specification["protocol"]
            and audit.get("public_verdict") == "experimental"
            and audit.get("artifact_schema_version") == 2
        ),
        "exact_gamma_endpoint_coverage": all(value >= minimum for value in exact.values()),
        "conservative_gamma_endpoint_coverage": all(
            value >= minimum for value in conservative.values()
        ),
        "learner_sensitivity_recorded": len(midpoint_means) == 2,
        "pairwise_without_kway_retained": pairwise["supported_pairs"] > 0,
        "boundary_refusal_recorded": _boundary_block_is_recorded(),
        "transport_refusal_recorded": _transport_refusal_is_recorded(),
        "violation_never_promoted": all(item["all_experimental"] for item in violation),
        "public_verdict_remains_experimental": audit.get("public_verdict") == "experimental",
    }
    payload = {
        "schema_version": 1,
        "protocol": specification["protocol"],
        "status": "pass" if all(criteria.values()) else "fail",
        "audit_manifest_sha256": sha256(audit_path.read_bytes()).hexdigest(),
        "criteria": criteria,
        "metrics": {
            "repetitions_per_arm": repetitions,
            "exact_gamma_endpoint_coverage": exact,
            "conservative_gamma_endpoint_coverage": conservative,
            "learner_midpoint_means": midpoint_means,
            "learner_midpoint_spread": float(
                max(midpoint_means.values()) - min(midpoint_means.values())
            ),
            "violation_runs": len(violation),
            "pairwise_supported_edges": pairwise["supported_pairs"],
            "pairwise_supported_hyperedges": pairwise["supported_hyperedges"],
            "perturbation_harness": _perturbation_harness(),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = sha256(encoded).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmarks/specs/stage5b_promotion_audit.json")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("release/stage5b_promotion_audit.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("release/artifacts/stage5b-promotion-evidence.json")
    )
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.repetitions is not None:
        specification["repetitions_per_arm"] = args.repetitions
    payload = run(specification, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
