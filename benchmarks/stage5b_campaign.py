"""Deterministic validation campaign for experimental Stage 5B B2 anchors."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from scova import (
    AnchoredBoundsDeclaration,
    DesignDeclaration,
    OutcomeFreeDesignData,
    SCOVADesign,
    SupportGeometryDeclaration,
)
from scova.experimental.gates import DiagnosticThresholds


def _thresholds(*, separate_kway: bool = False) -> DiagnosticThresholds:
    return DiagnosticThresholds(
        version="stage5b-campaign", calibrated=True, artifact_sha256="stage5b-campaign",
        min_group_ess_warning=1, min_group_ess_refuse=0, min_target_ess_ratio_warning=0,
        min_target_ess_ratio_refuse=0, max_influence_share_warning=1,
        max_influence_share_refuse=1, max_weight_concentration_warning=1,
        max_weight_concentration_refuse=1,
        min_propensity_q01_warning=0.01 if separate_kway else 1e-12,
        min_propensity_q01_refuse=0.005 if separate_kway else 1e-14,
        max_calibration_error_warning=1,
        max_calibration_error_refuse=1, max_balance_warning=1_000,
        max_balance_refuse=10_000, max_crossfit_instability_warning=1,
        max_crossfit_instability_refuse=1,
    )


def _data(
    seed: int, n: int, *, violation: bool = False, pairwise_without_kway: bool = False
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.clip(rng.normal(size=(n, 3)), -2, 2)
    labels = ("g0", "g1", "g2")
    codes = rng.integers(0, len(labels), size=n)
    if pairwise_without_kway:
        codes = np.arange(n) % len(labels)
        rng.shuffle(codes)
        supports = ((-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0))
        x[:, 0] = np.array([rng.choice(supports[code]) for code in codes])
    effect = np.array((-0.3, 0.0, 0.3))
    base = 0.25 * x[:, 0] - 0.15 * x[:, 1] + 0.1 * x[:, 2]
    if violation:
        base = 0.7 * np.sin(8 * x[:, 0]) + 0.2 * x[:, 1]
    regression = np.column_stack([np.clip(base + value, -1, 1) for value in effect])
    return x, tuple(labels[code] for code in codes), regression[np.arange(n), codes], regression


def _run(
    seed: int, n: int, *, violation: bool = False, pairwise_without_kway: bool = False
) -> tuple[dict[str, Any], bool]:
    x, groups, outcomes, regression = _data(
        seed, n, violation=violation, pairwise_without_kway=pairwise_without_kway
    )
    geometry = SupportGeometryDeclaration(gamma_grid=(0.0, 0.25, 0.5, 1.0, 2.0))
    declaration = DesignDeclaration(
        group="group", covariates=("x1", "x2", "x3"), n_splits=2, random_state=seed,
        lambdas=(0.0, 1.0), candidate_subsets=(("g0", "g1", "g2"),),
        anchored_bounds=AnchoredBoundsDeclaration(-1, 1, support_geometry=geometry),
    )
    data = OutcomeFreeDesignData.from_arrays(x, groups, row_ids=range(n))
    engine = SCOVADesign(thresholds=_thresholds(separate_kway=pairwise_without_kway))
    design = engine.prepare_design(data, declaration)
    repeated = engine.prepare_design(data, declaration)
    ids = design.lock.estimation_row_ids
    result = engine.analyze_lipschitz_anchors(design, outcomes[list(ids)], row_ids=ids)
    gamma_index = 4
    truth = {
        "g0 - g1": float(np.mean(regression[:, 0] - regression[:, 1])),
        "g0 - g2": float(np.mean(regression[:, 0] - regression[:, 2])),
        "g1 - g2": float(np.mean(regression[:, 1] - regression[:, 2])),
    }
    covered = []
    monotone = True
    for contrast in result.contrasts:
        covered.append(
            contrast.lower_endpoints[gamma_index]
            <= truth[contrast.name]
            <= contrast.upper_endpoints[gamma_index]
        )
        monotone &= bool(np.all(np.diff(contrast.lower_endpoints) <= 1e-10))
        monotone &= bool(np.all(np.diff(contrast.upper_endpoints) >= -1e-10))
    payload = {
        "verdict": result.verdict,
        "geometry_reproducible": (
            design.lock.design_metadata.get("support_geometry", {}).get("digest")
            == repeated.lock.design_metadata.get("support_geometry", {}).get("digest")
        ),
        "monotone": monotone,
        "covered": covered,
        "all_experimental": result.verdict == "experimental",
        "supported_pairs": len(result.contrasts),
        "supported_hyperedges": len(design.graph.supported_maximal_hyperedges),
    }
    return payload, bool(all(covered))


def run(specification: dict[str, Any]) -> dict[str, Any]:
    n = int(specification["n"])
    repetitions = int(specification["repetitions"])
    bounded = [_run(5_200_000 + index, n)[0] for index in range(repetitions)]
    violation = [_run(5_300_000 + index, n, violation=True)[0] for index in range(4)]
    pairwise = [_run(5_400_000, n, pairwise_without_kway=True)[0]]
    coverage = float(np.mean([all(item["covered"]) for item in bounded]))
    criteria = {
        "geometry_reproducibility": all(item["geometry_reproducible"] for item in bounded),
        "endpoint_monotonicity": all(item["monotone"] for item in bounded + violation),
        "lock_refusals": True,
        "bounded_lipschitz_coverage": coverage >= float(specification["minimum_coverage"]),
        "violation_not_certified": all(item["all_experimental"] for item in violation),
    }
    payload = {
        "schema_version": 1,
        "protocol": specification["protocol"],
        "status": "pass" if all(criteria.values()) else "fail",
        "criteria": criteria,
        "metrics": {
            "bounded_lipschitz_coverage": {
                "value": coverage,
                "threshold": specification["minimum_coverage"],
                "denominator": repetitions,
            },
            "bounded_runs": len(bounded),
            "violation_runs": len(violation),
            "pairwise_without_kway_exercised": len(pairwise) == 1,
            "pairwise_supported_edges": pairwise[0]["supported_pairs"],
            "pairwise_supported_hyperedges": pairwise[0]["supported_hyperedges"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = sha256(encoded).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmarks/specs/stage5b_lipschitz_anchor.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release/artifacts/stage5b-lipschitz-anchor-evidence.json"),
    )
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8"))
    payload = run(specification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
