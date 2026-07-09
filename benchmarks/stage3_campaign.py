"""Shardable, preregistered Stage-3 stabilization campaign."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

import scova
from scova import SCOVADeclaration
from scova.experimental import (
    DiagnosticThresholds,
    InferenceRefusedError,
    PathDeclaration,
    StabilizationData,
    StabilizationSpec,
    fit_path,
    generate_stabilization_data,
    make_learner_profile,
)


def _ungated_calibration_thresholds() -> DiagnosticThresholds:
    """Permit statistical calculations while retaining numerical refusals."""
    return DiagnosticThresholds(
        version="stage3-calibration-ungated-v1",
        min_group_ess_warning=0.0,
        min_group_ess_refuse=0.0,
        min_target_ess_ratio_warning=0.0,
        min_target_ess_ratio_refuse=0.0,
        max_influence_share_warning=1.0,
        max_influence_share_refuse=1.0,
        max_weight_concentration_warning=1.0,
        max_weight_concentration_refuse=1.0,
        min_propensity_q01_warning=1e-15,
        min_propensity_q01_refuse=1e-15,
        max_calibration_error_warning=1.0,
        max_calibration_error_refuse=1.0,
        max_balance_warning=1e9,
        max_balance_refuse=1e9,
        max_crossfit_instability_warning=1e9,
        max_crossfit_instability_refuse=1e9,
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def primary_cells(specification: dict) -> list[StabilizationSpec]:
    factors = specification["primary_factors"]
    names = tuple(factors)
    cells = []
    for values in itertools.product(*(factors[name] for name in names)):
        values_by_name = dict(zip(names, values, strict=True))
        cells.append(
            StabilizationSpec(
                **values_by_name,
                error="normal",
                nuisance="correct_both",
            )
        )
    if len(cells) != 288:
        raise RuntimeError(f"primary specification must contain 288 cells, found {len(cells)}")
    return cells


def robustness_cells() -> list[StabilizationSpec]:
    nuisance_regimes = (
        "oracle",
        "gps_correct_outcome_wrong",
        "outcome_correct_gps_wrong",
        "both_wrong",
        "flexible",
        "deliberately_inadequate",
    )
    bases = list(
        itertools.product(
            (2, 4, 8),
            ("moderate", "weak"),
            ("linear", "nonlinear"),
        )
    )
    cells = [
        StabilizationSpec(k, 1000, 20, overlap, outcome, "balanced", "normal", nuisance)
        for k, overlap, outcome in bases
        for nuisance in nuisance_regimes
    ]
    for index, (k, overlap, outcome) in enumerate(bases):
        # Each stress cell changes one factor from the balanced/normal flexible baseline.
        cells.append(StabilizationSpec(k, 1000, 20, overlap, outcome, "rare", "normal", "flexible"))
        cells.append(
            StabilizationSpec(
                k,
                1000,
                20,
                overlap,
                outcome,
                "balanced",
                "heteroskedastic" if index % 2 == 0 else "heavy_tailed",
                "flexible",
            )
        )
    if len(cells) != 96:
        raise RuntimeError(f"robustness specification must contain 96 cells, found {len(cells)}")
    if len(set(cells)) != len(cells):
        raise RuntimeError("robustness specification contains duplicate cells")
    return cells


def select_cells(all_cells: list[StabilizationSpec], requested: int) -> list[StabilizationSpec]:
    if requested >= len(all_cells):
        return all_cells
    indices = np.linspace(0, len(all_cells) - 1, requested, dtype=int)
    return [all_cells[index] for index in indices]


def _null_data(data: StabilizationData, seed: int) -> StabilizationData:
    labels = data.group_labels
    codes = data.data["group"].map({label: code for code, label in enumerate(labels)}).to_numpy()
    observed_mean = data.outcome_regression[np.arange(len(codes)), codes]
    residual = data.data["outcome"].to_numpy() - observed_mean
    common = data.outcome_regression.mean(axis=1)
    frame = data.data.copy()
    frame["outcome"] = common + residual
    regression = np.repeat(common[:, None], len(labels), axis=1)
    del seed
    return StabilizationData(
        frame,
        data.propensity,
        regression,
        data.pseudo_propensity,
        labels,
    )


def _uniform_band(
    estimates: np.ndarray,
    influence: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(influence)
    errors = np.sqrt(np.sum(np.square(influence), axis=0) / (n * (n - 1)))
    rng = np.random.default_rng(seed)
    multipliers = rng.normal(size=(n_bootstrap, n))
    multipliers -= multipliers.mean(axis=1, keepdims=True)
    maxima = np.max(np.abs((multipliers @ influence) / n / errors), axis=1)
    critical = np.quantile(maxima, 0.95, method="higher")
    return estimates - critical * errors, estimates + critical * errors


def _fit_one(
    data: StabilizationData,
    spec: StabilizationSpec,
    seed: int,
    bootstrap: int,
    thresholds: DiagnosticThresholds,
) -> dict:
    covariates = tuple(column for column in data.data if column.startswith("x"))
    base = SCOVADeclaration("outcome", "group", covariates, n_splits=5, random_state=seed)
    declaration = PathDeclaration(base, thresholds=thresholds)
    nuisance = data.nuisance_predictions(spec.nuisance)
    profile_name = "linear" if spec.outcome == "linear" else "nonlinear"
    if spec.nuisance in ("flexible", "correct_both"):
        profile_name = "ensemble" if spec.nuisance == "flexible" else profile_name
    elif spec.nuisance == "deliberately_inadequate":
        profile_name = "deliberately_inadequate"
    profile = make_learner_profile(profile_name, random_state=seed)
    result = fit_path(
        data.data,
        declaration,
        estimator=profile.estimator(),
        nuisance_predictions=nuisance,
    )
    contrast_name = f"g0 - g{spec.n_groups - 1}"
    contrast = result.contrasts[contrast_name]
    truth = data.target_path(result.lambdas, contrast.weights)
    pseudo_truth = data.target_path(result.lambdas, contrast.weights, pseudo=True)
    record = {
        "gate_status": result.gate_decision.status.value,
        "gate_metrics": {metric.name: metric.value for metric in result.gate_decision.metrics},
        "scientific_target_rmse": float(np.sqrt(np.mean(np.square(contrast.estimates - truth)))),
        "scientific_target_mean_error": float(np.mean(contrast.estimates - truth)),
        "mean_standard_error": float(np.mean(contrast.standard_errors)),
        "pseudo_target_rmse": float(np.sqrt(np.mean(np.square(contrast.estimates - pseudo_truth)))),
        "scientific_pseudo_target_max_drift": float(np.max(np.abs(truth - pseudo_truth))),
        "refused": False,
        "uniform_coverage": None,
        "false_sign_certificate": None,
        "stability_covered": None,
        "naive_uniform_coverage": None,
        "naive_target_rmse": None,
    }
    naive_estimate = result.naive_group_means[:, 0] - result.naive_group_means[:, spec.n_groups - 1]
    naive_influence = (
        result.naive_influence_values[:, :, 0]
        - result.naive_influence_values[:, :, spec.n_groups - 1]
    )
    naive_lower, naive_upper = _uniform_band(naive_estimate, naive_influence, bootstrap, seed)
    record["naive_uniform_coverage"] = bool(np.all((naive_lower <= truth) & (truth <= naive_upper)))
    record["naive_target_rmse"] = float(np.sqrt(np.mean(np.square(naive_estimate - truth))))
    try:
        inference = result.infer((contrast_name,), n_bootstrap=bootstrap, random_state=seed)
    except InferenceRefusedError:
        record["refused"] = True
        return record
    lower, upper = inference.lower_bands[0], inference.upper_bands[0]
    record["uniform_coverage"] = bool(np.all((lower <= truth) & (truth <= upper)))
    sign = inference.sign_certificates[0]
    falsely_positive = any(
        truth[np.where(result.lambdas == value)[0][0]] <= 0 for value in sign.positive_lambdas
    )
    falsely_negative = any(
        truth[np.where(result.lambdas == value)[0][0]] >= 0 for value in sign.negative_lambdas
    )
    record["false_sign_certificate"] = bool(falsely_positive or falsely_negative)
    oracle_stability = float(np.max(np.abs(truth[:-1] - truth[-1])))
    record["stability_covered"] = bool(
        inference.stability_certificates[0].simultaneous_upper_bound >= oracle_stability
    )
    return record


def _failed_fit(error: Exception) -> dict:
    """Record an execution failure as a refusal instead of losing the campaign shard."""
    return {
        "gate_status": "refuse",
        "gate_metrics": {},
        "refused": True,
        "execution_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "uniform_coverage": None,
        "false_sign_certificate": None,
        "stability_covered": None,
        "scientific_target_rmse": None,
        "scientific_target_mean_error": None,
        "mean_standard_error": None,
        "pseudo_target_rmse": None,
        "scientific_pseudo_target_max_drift": None,
        "naive_uniform_coverage": None,
        "naive_target_rmse": None,
    }


def run_campaign(
    *,
    tier: str,
    specification_path: Path,
    output: Path,
    shard_index: int,
    shard_count: int,
    repetitions_override: int | None,
    bootstrap_override: int | None,
    seed_set: str,
    threshold_path: Path | None,
) -> None:
    specification_bytes = specification_path.read_bytes()
    specification = json.loads(specification_bytes)
    tier_spec = specification["tiers"][tier]
    if not specification.get("frozen"):
        raise ValueError("campaign specification must be frozen before execution")
    source_cells = (
        robustness_cells()
        if tier_spec.get("source") == "robustness"
        else primary_cells(specification)
    )
    if "cell_indices" in tier_spec:
        selected_indices = list(tier_spec["cell_indices"])
        if len(selected_indices) != tier_spec["cells"] or len(set(selected_indices)) != len(
            selected_indices
        ):
            raise ValueError("explicit cell manifest has the wrong size or duplicates")
        if any(index < 0 or index >= len(source_cells) for index in selected_indices):
            raise ValueError("explicit cell manifest contains an invalid source index")
    else:
        selected = select_cells(source_cells, tier_spec["cells"])
        selected_indices = [source_cells.index(cell) for cell in selected]
    selected_pairs = [
        (source_index, source_cells[source_index]) for source_index in selected_indices
    ]
    selected_pairs = [
        pair for index, pair in enumerate(selected_pairs) if index % shard_count == shard_index
    ]
    if threshold_path is not None:
        threshold_values = json.loads(threshold_path.read_text(encoding="utf-8"))
        thresholds = DiagnosticThresholds.from_calibration_artifact(threshold_values)
    elif seed_set == "calibration":
        thresholds = _ungated_calibration_thresholds()
    else:
        thresholds = DiagnosticThresholds()
    if (
        tier
        in (
            "directional_validation",
            "directional_robustness",
            "local_validation_pilot",
            "local_robustness_pilot",
        )
        and not thresholds.calibrated
    ):
        raise ValueError("directional validation requires a locked calibrated threshold artifact")
    repetitions = repetitions_override or tier_spec["repetitions"]
    bootstrap = bootstrap_override or tier_spec["bootstrap"]
    seed_namespace = specification[f"{seed_set}_seed_namespace"]
    started = time.perf_counter()
    records = []
    for local_cell, (source_cell_index, cell) in enumerate(selected_pairs):
        cell_id = shard_index + local_cell * shard_count
        print(
            f"stage3 campaign tier={tier} shard={shard_index}/{shard_count} "
            f"cell={cell_id} spec={asdict(cell)}",
            flush=True,
        )
        for repetition in range(repetitions):
            seed = seed_namespace + cell_id * 100_000 + repetition
            try:
                data = generate_stabilization_data(cell, seed=seed)
                alternative = _fit_one(data, cell, seed, bootstrap, thresholds)
            except Exception as error:  # campaign artifact must retain the failing cell
                data = None
                alternative = _failed_fit(error)
            if data is None:
                null_record = _failed_fit(
                    RuntimeError("null fit skipped because alternative data generation failed")
                )
            else:
                try:
                    null_record = _fit_one(
                        _null_data(data, seed), cell, seed + 1, bootstrap, thresholds
                    )
                except Exception as error:  # campaign artifact must retain the failing cell
                    null_record = _failed_fit(error)
            records.append(
                {
                    "cell_id": cell_id,
                    "source_cell_index": source_cell_index,
                    "repetition": repetition,
                    "seed": seed,
                    "spec": asdict(cell),
                    "alternative": alternative,
                    "null": null_record,
                }
            )
            if (repetition + 1) % max(1, min(100, repetitions)) == 0:
                print(
                    f"stage3 campaign cell={cell_id} completed {repetition + 1}/{repetitions}",
                    flush=True,
                )
    payload = {
        "schema_version": 1,
        "tier": tier,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "repetitions": repetitions,
        "bootstrap": bootstrap,
        "seed_set": seed_set,
        "seed_namespace": seed_namespace,
        "validation_level": specification["validation_level"],
        "threshold_version": thresholds.version,
        "threshold_artifact_sha256": thresholds.artifact_sha256,
        "cell_manifest": [
            {"source_cell_index": index, "spec": asdict(source_cells[index])}
            for index in selected_indices
        ],
        "specification_sha256": sha256(specification_bytes).hexdigest(),
        "git_commit": _git_commit(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scova": scova.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.write_text(text, encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        sha256(text.encode()).hexdigest() + "\n", encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tier",
        choices=(
            "pull_request",
            "calibration",
            "directional_validation",
            "directional_robustness",
            "local_validation_pilot",
            "local_robustness_pilot",
            "nightly",
            "publication_release",
            "publication_robustness",
        ),
        required=True,
    )
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/specs/stage3_release.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument(
        "--seed-set",
        choices=("calibration", "validation", "publication", "pilot"),
        default="validation",
    )
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must lie in [0, shard count)")
    run_campaign(
        tier=args.tier,
        specification_path=args.spec,
        output=args.output,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        repetitions_override=args.repetitions,
        bootstrap_override=args.bootstrap,
        seed_set=args.seed_set,
        threshold_path=args.thresholds,
    )


if __name__ == "__main__":
    main()
