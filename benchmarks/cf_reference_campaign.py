"""Preregistered randomized-continuous SCOVA-CF reference campaign."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scova import ContrastSpec
from scova.cf import (
    SCOVACF,
    AnalysisMode,
    CFValidationProtocol,
    KnownAssignment,
    SCOVACFDeclaration,
    SCOVACFRefusal,
    canonical_checksum,
)


@dataclass(frozen=True, slots=True)
class CampaignData:
    data: pd.DataFrame
    probabilities: tuple[float, ...]
    group_labels: tuple[str, ...]
    true_group_means: np.ndarray


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def simulate_reference_cell(cell: Mapping[str, Any], *, seed: int) -> CampaignData:
    """Generate one randomized cell with finite-population counterfactual truth."""
    rng = np.random.default_rng(seed)
    k = int(cell["n_groups"])
    n = int(cell["n_per_group"]) * k
    labels = tuple(f"g{code}" for code in range(k))
    x = rng.normal(size=(n, 5))
    allocation = str(cell["allocation"])
    probabilities = np.ones(k, dtype=float)
    if allocation == "unbalanced":
        probabilities = np.geomspace(1.0, 0.35, k)
    if cell["support"] == "weak":
        probabilities = np.geomspace(1.0, 0.08, k)
    probabilities /= probabilities.sum()

    surface = str(cell["surface"])
    if surface == "linear":
        baseline = 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]
    elif surface == "smooth-nonlinear":
        baseline = np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2 - 0.3 * x[:, 2]
    elif surface == "threshold":
        baseline = 0.9 * (x[:, 0] > 0) - 0.5 * (x[:, 1] > 0.5) + 0.2 * x[:, 2]
    elif surface == "interaction":
        baseline = 0.7 * x[:, 0] * x[:, 1] + 0.4 * np.sin(x[:, 2])
    else:
        raise ValueError(f"Unknown outcome surface: {surface}")

    effect_kind = str(cell["effect"])
    group_effects = np.zeros(k) if effect_kind == "null" else np.linspace(0.0, 0.8, k)
    conditional_means = np.empty((n, k), dtype=float)
    for code in range(k):
        heterogeneity = 0.0
        if effect_kind == "heterogeneous":
            heterogeneity = 0.35 * code * np.tanh(x[:, 0])
        conditional_means[:, code] = baseline + group_effects[code] + heterogeneity

    group_codes = rng.choice(k, size=n, p=probabilities)
    if cell["support"] == "structural-failure":
        group_codes[group_codes == k - 1] = 0
    noise_kind = str(cell["noise"])
    if noise_kind == "normal":
        errors = rng.normal(size=n)
    elif noise_kind == "heteroskedastic":
        errors = rng.normal(scale=0.6 + 0.5 * np.abs(x[:, 0]), size=n)
    elif noise_kind == "heavy-tailed":
        errors = rng.standard_t(df=4, size=n) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unknown noise distribution: {noise_kind}")
    outcome = conditional_means[np.arange(n), group_codes] + errors
    data = pd.DataFrame(x, columns=[f"x{index}" for index in range(1, 6)])
    data["group"] = [labels[code] for code in group_codes]
    data["outcome"] = outcome
    return CampaignData(
        data=data,
        probabilities=tuple(float(value) for value in probabilities),
        group_labels=labels,
        true_group_means=conditional_means.mean(axis=0),
    )


def _declaration(
    generated: CampaignData,
    cell: Mapping[str, Any],
    *,
    include_stability: bool,
) -> SCOVACFDeclaration:
    covariates = tuple(f"x{index}" for index in range(1, 6))
    return SCOVACFDeclaration(
        outcome="outcome",
        group="group",
        covariates=covariates,
        mode=AnalysisMode.RANDOMIZED,
        scientific_question="Reference randomized population-counterfactual means",
        eligibility="All generated independent units",
        target_population="Generated finite study population",
        group_definitions=tuple(
            (label, f"Randomized condition {label}") for label in generated.group_labels
        ),
        outcome_time="simulated follow-up",
        outcome_units="simulated points",
        covariate_rationales=tuple((name, "Baseline prognostic factor") for name in covariates),
        assignment=KnownAssignment(
            probabilities=tuple(zip(generated.group_labels, generated.probabilities, strict=True))
        ),
        outcome_nuisance_strategy=str(cell["learner"]),  # type: ignore[arg-type]
        n_splits=3,
        random_state=17,
        stability_seeds=(101, 211, 307, 401, 503) if include_stability else (),
        contrasts=(
            ContrastSpec(
                "g1 - g0",
                ((generated.group_labels[1], 1.0), (generated.group_labels[0], -1.0)),
            ),
        ),
    )


def _support_features(result: Any) -> dict[str, float]:
    groups = result.diagnostics["support"]["groups"].values()
    influence = result.diagnostics["influence_concentration"].values()
    return {
        "minimum_ess_ratio": min(group["effective_sample_size_ratio"] for group in groups),
        "maximum_normalized_weight": max(group["maximum_normalized_weight"] for group in groups),
        "maximum_top_one_percent_weight_share": max(
            group["top_one_percent_weight_share"] for group in groups
        ),
        "maximum_absolute_weighted_balance_difference": max(
            group["maximum_absolute_weighted_balance_difference"] for group in groups
        ),
        "maximum_influence_top_one_percent_share": max(
            group["top_one_percent_variance_share"] for group in influence
        ),
        "maximum_seed_standardized_departure": (
            0.0
            if result.seed_stability is None
            else result.seed_stability.maximum_standardized_departure
        ),
    }


def run_campaign(
    protocol: CFValidationProtocol,
    *,
    lane: str,
    replications: int | None = None,
    max_cells: int | None = None,
    include_stability: bool = True,
) -> dict[str, Any]:
    """Run a campaign lane; overrides create smoke evidence that cannot promote."""
    partition = getattr(protocol, lane)
    count = partition.count if replications is None else replications
    if count < 1 or count > partition.count:
        raise ValueError("replications must lie between one and the frozen lane count")
    cells = protocol.retained_cells[:max_cells]
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        cell_records: list[dict[str, Any]] = []
        for offset in range(count):
            seed = partition.start + cell_index * partition.count + offset
            generated = simulate_reference_cell(cell, seed=seed)
            result = SCOVACF().analyze(
                generated.data,
                _declaration(generated, cell, include_stability=include_stability),
            )
            if isinstance(result, SCOVACFRefusal):
                record = {
                    "seed": seed,
                    "refused": True,
                    "status_code": result.status.code,
                }
            else:
                contrast = result.contrasts["g1 - g0"]
                truth = float(generated.true_group_means[1] - generated.true_group_means[0])
                lower, upper = contrast.confidence_interval
                record = {
                    "seed": seed,
                    "refused": False,
                    "status_code": result.status.code,
                    "estimate": contrast.estimate,
                    "standard_error": contrast.standard_error,
                    "truth": truth,
                    "covered": bool(lower <= truth <= upper),
                    "rejected": bool(contrast.p_value < 0.05),
                    "null": bool(abs(truth) <= 1e-12),
                    **_support_features(result),
                }
            cell_records.append(record)
            records.append({"cell_index": cell_index, **record})
        usable = [record for record in cell_records if not record["refused"]]
        estimates = np.array([record["estimate"] for record in usable], dtype=float)
        truths = np.array([record["truth"] for record in usable], dtype=float)
        standard_errors = np.array(
            [record["standard_error"] for record in usable], dtype=float
        )
        empirical_sd = (
            float(np.std(estimates - truths, ddof=1)) if len(usable) > 1 else None
        )
        summaries.append(
            {
                "cell_index": cell_index,
                "cell": dict(cell),
                "replications": count,
                "refusal_rate": 1.0 - len(usable) / count,
                "coverage": (
                    None if not usable else float(np.mean([r["covered"] for r in usable]))
                ),
                "type_i_error": (
                    None
                    if not any(r["null"] for r in usable)
                    else float(np.mean([r["rejected"] for r in usable if r["null"]]))
                ),
                "bias": None if not usable else float(np.mean(estimates - truths)),
                "empirical_standard_deviation": empirical_sd,
                "mean_standard_error": (
                    None if not usable else float(np.mean(standard_errors))
                ),
                "standard_error_ratio": (
                    None
                    if not usable or empirical_sd is None or empirical_sd == 0
                    else float(np.mean(standard_errors) / empirical_sd)
                ),
            }
        )
    complete = (
        count == partition.count
        and len(cells) == len(protocol.retained_cells)
        and include_stability
    )
    evidence = {
        "artifact_type": "scova-cf-reference-campaign",
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "lane": lane,
        "complete_frozen_lane": complete,
        "replications_per_cell": count,
        "cell_count": len(cells),
        "software_versions": {
            package: _installed_version(package)
            for package in ("scova", "numpy", "pandas", "scikit-learn", "doubleml", "econml")
        },
        "summaries": summaries,
        "records": records,
        "promotion_decision": "blocked/no-calibrated-support-profile",
    }
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lane", choices=("pilot", "calibration", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()
    evidence = run_campaign(
        CFValidationProtocol.load(args.spec),
        lane=args.lane,
        replications=args.replications,
        max_cells=args.max_cells,
        include_stability=not args.skip_stability,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
