"""Shardable frozen SCOVA-CF randomized-reference validation campaign."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from scova import ContrastSpec
from scova.cf import (
    SCOVACF,
    AnalysisMode,
    CFSupportProfile,
    CFValidationProtocol,
    KnownAssignment,
    SCOVACFDeclaration,
    SCOVACFRefusal,
    canonical_checksum,
)

STABILITY_SEEDS = (101, 211, 307, 401, 503)
DEPENDENCY_LOCK = Path(__file__).with_name("requirements-cf-validation.txt")


@dataclass(frozen=True, slots=True)
class CampaignData:
    data: pd.DataFrame
    probabilities: tuple[float, ...]
    group_labels: tuple[str, ...]
    true_group_means: np.ndarray
    source_metadata: Mapping[str, Any]


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


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


def dependency_lock_checksum() -> str:
    return sha256(DEPENDENCY_LOCK.read_bytes()).hexdigest()


def write_deterministic_gzip(path: Path, text: str, *, compresslevel: int = 6) -> None:
    """Write UTF-8 gzip bytes whose checksum is independent of wall-clock time."""
    with path.open("wb") as raw, gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=compresslevel,
        fileobj=raw,
        mtime=0,
    ) as archive, io.TextIOWrapper(archive, encoding="utf-8") as stream:
        stream.write(text)


def _probabilities(k: int, allocation: str, support: str) -> np.ndarray:
    if allocation == "balanced":
        values = np.ones(k)
    elif allocation == "moderate":
        values = np.geomspace(1.0, 0.35, k)
    elif allocation == "rare":
        values = np.geomspace(1.0, 0.08, k)
    else:
        raise ValueError(f"Unknown allocation: {allocation}")
    if support == "weak":
        values = np.geomspace(1.0, 0.03, k)
    return values / values.sum()


def _conditional_means(x: np.ndarray, cell: Mapping[str, Any]) -> np.ndarray:
    k = int(cell["n_groups"])
    surface = str(cell["surface"])
    if surface == "linear":
        baseline = 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]
    elif surface == "smooth-nonlinear":
        baseline = np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2 - 0.3 * x[:, 2]
    elif surface == "threshold":
        baseline = 0.9 * (x[:, 0] > 0) - 0.5 * (x[:, 1] > 0.5) + 0.2 * x[:, 2]
    elif surface == "interaction":
        baseline = 0.7 * x[:, 0] * x[:, 1] + 0.4 * np.sin(x[:, 2])
    elif surface == "weak":
        baseline = 0.08 * x[:, 0] - 0.05 * x[:, 1]
    else:
        raise ValueError(f"Unknown outcome surface: {surface}")
    effect = str(cell["effect"])
    if effect == "null":
        group_effects = np.zeros(k)
    elif effect == "partial-null" and k > 2:
        group_effects = np.linspace(0.0, 0.8, k)
        group_effects[1] = 0.0
    else:
        group_effects = np.linspace(0.0, 0.8, k)
    means = np.empty((len(x), k))
    for code in range(k):
        heterogeneity = (
            0.35 * code * np.tanh(x[:, 0]) if effect == "heterogeneous" else 0.0
        )
        means[:, code] = baseline + group_effects[code] + heterogeneity
    return means


def _errors(rng: np.random.Generator, x: np.ndarray, noise: str) -> np.ndarray:
    if noise == "normal":
        return rng.normal(size=len(x))
    if noise == "heteroskedastic":
        return rng.normal(scale=0.6 + 0.5 * np.abs(x[:, 0]), size=len(x))
    if noise == "heavy-tailed":
        return rng.standard_t(df=4, size=len(x)) / np.sqrt(2.0)
    raise ValueError(f"Unknown noise distribution: {noise}")


def simulate_reference_cell(cell: Mapping[str, Any], *, seed: int) -> CampaignData:
    """Generate a Gaussian-covariate simulation cell with finite-population truth."""
    rng = np.random.default_rng(seed)
    k = int(cell["n_groups"])
    p = int(cell.get("n_covariates", 5))
    n = int(cell["n_per_group"]) * k
    labels = tuple(f"g{code}" for code in range(k))
    innovations = rng.normal(size=(n, p))
    x = innovations.copy()
    for column in range(1, p):
        x[:, column] = 0.35 * x[:, column - 1] + np.sqrt(1 - 0.35**2) * x[:, column]
    probabilities = _probabilities(k, str(cell["allocation"]), str(cell["support"]))
    means = _conditional_means(x, cell)
    codes = rng.choice(k, size=n, p=probabilities)
    if cell["support"] == "structural-failure":
        codes[codes == k - 1] = 0
    outcome = means[np.arange(n), codes] + _errors(rng, x, str(cell["noise"]))
    data = pd.DataFrame(x, columns=[f"x{index}" for index in range(1, p + 1)])
    data["group"] = [labels[code] for code in codes]
    data["outcome"] = outcome
    return CampaignData(
        data=data,
        probabilities=tuple(float(value) for value in probabilities),
        group_labels=labels,
        true_group_means=means.mean(axis=0),
        source_metadata={"kind": "simulation"},
    )


def _dataset_payload(name: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    bunch = load_diabetes() if name == "diabetes" else load_breast_cancer()
    x = np.asarray(bunch.data, dtype=float)
    target = np.asarray(bunch.target, dtype=float)
    names = tuple(str(value) for value in bunch.feature_names)
    return x, target, names, str(bunch.DESCR)


def plasmode_source_checksum(name: str) -> str:
    x, target, names, description = _dataset_payload(name)
    digest = sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(target).tobytes())
    digest.update(json.dumps(names).encode("utf-8"))
    digest.update(description.encode("utf-8"))
    return digest.hexdigest()


def simulate_plasmode_cell(cell: Mapping[str, Any], *, seed: int) -> CampaignData:
    """Resample unique real rows and inject a known continuous counterfactual model."""
    name = str(cell["dataset"])
    x_source, target_source, names, _ = _dataset_payload(name)
    rng = np.random.default_rng(seed)
    k = int(cell["n_groups"])
    n = int(cell["n_per_group"]) * k
    if n > len(x_source):
        raise ValueError("Plasmode samples cannot exceed the unique source rows")
    indices = rng.choice(len(x_source), size=n, replace=False)
    x = x_source[indices].copy()
    target_sd = float(target_source.std(ddof=1))
    baseline = (target_source[indices] - target_source.mean()) / target_sd
    standardized = (x_source - x_source.mean(axis=0)) / np.where(
        x_source.std(axis=0, ddof=1) > 0, x_source.std(axis=0, ddof=1), 1.0
    )
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    loading = vh[0].copy()
    largest = int(np.argmax(np.abs(loading)))
    loading *= np.sign(loading[largest]) or 1.0
    score = standardized[indices] @ loading
    score = (score - score.mean()) / (score.std(ddof=1) or 1.0)
    effect = str(cell["effect"])
    group_effects = np.zeros(k) if effect == "null" else np.linspace(0.0, 0.8, k)
    means = np.empty((n, k))
    for code in range(k):
        heterogeneous = 0.25 * code * score if effect == "heterogeneous" else 0.0
        means[:, code] = baseline + group_effects[code] + heterogeneous
    probabilities = _probabilities(k, str(cell["allocation"]), "strong")
    codes = rng.choice(k, size=n, p=probabilities)
    outcome = means[np.arange(n), codes] + _errors(rng, x, str(cell["noise"]))
    columns = tuple(f"x{index}" for index in range(1, x.shape[1] + 1))
    data = pd.DataFrame(x, columns=columns)
    data["group"] = [f"g{code}" for code in codes]
    data["outcome"] = outcome
    return CampaignData(
        data=data,
        probabilities=tuple(float(value) for value in probabilities),
        group_labels=tuple(f"g{code}" for code in range(k)),
        true_group_means=means.mean(axis=0),
        source_metadata={
            "kind": "plasmode",
            "dataset": name,
            "source_checksum": plasmode_source_checksum(name),
            "source_row_indices": [int(value) for value in indices],
            "source_feature_names": list(names),
            "sampling": "without-replacement",
        },
    )


def _declaration(
    generated: CampaignData,
    cell: Mapping[str, Any],
    *,
    include_stability: bool,
) -> SCOVACFDeclaration:
    covariates = tuple(column for column in generated.data if column.startswith("x"))
    contrasts = tuple(
        ContrastSpec(
            f"{label} - {generated.group_labels[0]}",
            ((label, 1.0), (generated.group_labels[0], -1.0)),
        )
        for label in generated.group_labels[1:]
    )
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
        stability_seeds=STABILITY_SEEDS if include_stability else (),
        contrasts=contrasts,
    )


def _contrast_summary(
    means: Sequence[float], covariance: Sequence[Sequence[float]], truth: np.ndarray
) -> list[dict[str, Any]]:
    values = np.asarray(means, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    summaries = []
    for code in range(1, len(values)):
        weights = np.zeros(len(values))
        weights[code] = 1.0
        weights[0] = -1.0
        estimate = float(weights @ values)
        standard_error = float(np.sqrt(max(float(weights @ cov @ weights), 0.0)))
        target = float(truth[code] - truth[0])
        summaries.append(
            {
                "group_code": code,
                "estimate": estimate,
                "standard_error": standard_error,
                "truth": target,
                "covered": bool(abs(estimate - target) <= 1.959963984540054 * standard_error),
                "rejected": bool(
                    standard_error > 0
                    and abs(estimate / standard_error) > 1.959963984540054
                ),
                "null": bool(abs(target) <= 1e-12),
            }
        )
    return summaries


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


def fit_campaign_record(
    generated: CampaignData,
    cell: Mapping[str, Any],
    *,
    include_stability: bool,
    simultaneous_bootstrap: int = 0,
    seed: int,
) -> dict[str, Any]:
    result = SCOVACF().analyze(
        generated.data,
        _declaration(generated, cell, include_stability=include_stability),
    )
    if isinstance(result, SCOVACFRefusal):
        return {
            "refused": True,
            "status_code": result.status.code,
            "source_metadata": generated.source_metadata,
        }
    truth = generated.true_group_means
    record: dict[str, Any] = {
        "refused": False,
        "status_code": result.status.code,
        "group_means": result.group_means.tolist(),
        "group_standard_errors": result.group_standard_errors.tolist(),
        "true_group_means": truth.tolist(),
        "contrasts": _contrast_summary(result.group_means, result.covariance, truth),
        "omnibus": result.omnibus.to_dict(),
        "support_features": _support_features(result),
        "source_metadata": generated.source_metadata,
        "benchmarks": {},
    }
    for name, benchmark in result.benchmarks.items():
        if "means" in benchmark and "covariance" in benchmark:
            record["benchmarks"][name] = {
                "status": benchmark.get("status", "complete"),
                "contrasts": _contrast_summary(benchmark["means"], benchmark["covariance"], truth),
            }
        else:
            record["benchmarks"][name] = {"status": benchmark.get("status", "limited")}
    if simultaneous_bootstrap:
        inference = result.infer(
            n_bootstrap=simultaneous_bootstrap,
            random_state=seed + 7919,
        )
        core = inference.core
        truths = np.array([item["truth"] for item in record["contrasts"]])
        intervals = np.asarray(
            [value.simultaneous_confidence_interval for value in core.contrasts]
        )
        record["simultaneous"] = {
            "family": list(core.family),
            "covered_family": bool(
                np.all((intervals[:, 0] <= truths) & (truths <= intervals[:, 1]))
            ),
            "any_null_rejected": bool(
                any(
                    item["null"] and adjusted < 0.05
                    for item, adjusted in zip(
                        record["contrasts"],
                        [value.adjusted_p_value for value in core.contrasts],
                        strict=True,
                    )
                )
            ),
        }
    return record


def _all_cells(protocol: CFValidationProtocol) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        *(("simulation", cell) for cell in protocol.retained_cells),
        *(("plasmode", cell) for cell in protocol.plasmode_cells),
    ]


def _generate(kind: str, cell: Mapping[str, Any], seed: int) -> CampaignData:
    return (
        simulate_reference_cell(cell, seed=seed)
        if kind == "simulation"
        else simulate_plasmode_cell(cell, seed=seed)
    )


def run_campaign(
    protocol: CFValidationProtocol,
    *,
    lane: str,
    replications: int | None = None,
    max_cells: int | None = None,
    include_stability: bool = True,
) -> dict[str, Any]:
    """In-memory smoke interface retained for unit tests and local pilots."""
    partition = getattr(protocol, lane)
    count = partition.count if replications is None else replications
    cells = _all_cells(protocol)[:max_cells]
    records = []
    for cell_index, (kind, cell) in enumerate(cells):
        for repetition in range(count):
            seed = partition.start + cell_index * partition.count + repetition
            records.append(
                {
                    "cell_index": cell_index,
                    "cell_kind": kind,
                    "repetition": repetition,
                    "seed": seed,
                    **fit_campaign_record(
                        _generate(kind, cell, seed),
                        cell,
                        include_stability=include_stability,
                        seed=seed,
                    ),
                }
            )
    complete = bool(
        count == partition.count
        and len(cells) == len(_all_cells(protocol))
        and include_stability
    )
    evidence = {
        "artifact_type": "scova-cf-reference-campaign",
        "schema_version": 2,
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "lane": lane,
        "complete_frozen_lane": complete,
        "replications_per_cell": count,
        "cell_count": len(cells),
        "records": records,
        "promotion_decision": "blocked/no-calibrated-support-profile",
    }
    evidence["evidence_checksum"] = canonical_checksum(evidence)
    return evidence


def run_shard(
    protocol: CFValidationProtocol,
    *,
    lane: str,
    output: Path,
    shard_index: int,
    shard_count: int,
    resume: bool,
    replications_override: int | None,
    max_cells: int | None,
    include_stability: bool,
    candidate_profile: CFSupportProfile | None = None,
) -> None:
    partition = getattr(protocol, lane)
    if lane == "validation":
        candidate_source = protocol.candidate_source
        sourced_candidate = bool(
            candidate_profile is not None
            and candidate_source
            and candidate_profile.protocol_checksum == candidate_source.get("protocol_checksum")
            and candidate_profile.checksum == candidate_source.get("profile_checksum")
        )
        if (
            candidate_profile is None
            or candidate_profile.state != "candidate"
            or (
                candidate_profile.protocol_checksum != protocol.checksum
                and not sourced_candidate
            )
        ):
            raise ValueError("Held-out shards require the frozen candidate profile")
    elif candidate_profile is not None:
        raise ValueError("A candidate profile may be supplied only to held-out validation")
    repetitions = partition.count if replications_override is None else replications_override
    if repetitions < 1 or repetitions > partition.count:
        raise ValueError("replications must lie within the frozen lane")
    cells = _all_cells(protocol)[:max_cells]
    partial = output.with_suffix(output.suffix + ".partial.ndjson")
    completed: set[tuple[int, int]] = set()
    if resume and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            cell_index = int(value["cell_index"])
            repetition = int(value["repetition"])
            if not 0 <= cell_index < len(cells) or not 0 <= repetition < repetitions:
                raise ValueError("Checkpoint contains a record outside this shard configuration")
            global_index = cell_index * repetitions + repetition
            expected_seed = partition.start + cell_index * partition.count + repetition
            if (
                global_index % shard_count != shard_index
                or int(value["seed"]) != expected_seed
                or value["cell"] != dict(cells[cell_index][1])
            ):
                raise ValueError("Checkpoint does not belong to this protocol shard")
            completed.add((cell_index, repetition))
    elif partial.exists():
        partial.unlink()
    started = time.perf_counter()
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8") as stream:
        written = 0
        for cell_index, (kind, cell) in enumerate(cells):
            for repetition in range(repetitions):
                global_index = cell_index * repetitions + repetition
                key = (cell_index, repetition)
                if global_index % shard_count != shard_index or key in completed:
                    continue
                seed = partition.start + cell_index * partition.count + repetition
                try:
                    generated = _generate(kind, cell, seed)
                    fitted = fit_campaign_record(
                        generated,
                        cell,
                        include_stability=include_stability,
                        seed=seed,
                    )
                except Exception as error:  # campaign evidence retains execution failures
                    fitted = {
                        "refused": True,
                        "status_code": "execution-error",
                        "execution_error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                record = {
                    "cell_index": cell_index,
                    "cell_kind": kind,
                    "cell": dict(cell),
                    "repetition": repetition,
                    "seed": seed,
                    **fitted,
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                written += 1
                if written % 25 == 0:
                    stream.flush()
    text = partial.read_text(encoding="utf-8")
    write_deterministic_gzip(output, text)
    output.with_suffix(output.suffix + ".sha256").write_text(
        sha256(output.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    metadata = {
        "artifact_type": "scova-cf-reference-shard",
        "schema_version": 2,
        "protocol_id": protocol.protocol_id,
        "protocol_checksum": protocol.checksum,
        "lane": lane,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "replications_per_cell": repetitions,
        "cell_count": len(cells),
        "include_stability": include_stability,
        "complete_frozen_lane_configuration": (
            repetitions == partition.count
            and len(cells) == len(_all_cells(protocol))
            and include_stability
        ),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                package: _installed_version(package)
                for package in ("scova", "numpy", "pandas", "scipy", "scikit-learn")
            },
        },
        "plasmode_source_checksums": {
            name: plasmode_source_checksum(name) for name in ("diabetes", "breast-cancer")
        },
        "dependency_lock_checksum": dependency_lock_checksum(),
        "candidate_profile_checksum": (
            None if candidate_profile is None else candidate_profile.checksum
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "record_count": len(text.splitlines()),
        "records_sha256": sha256(output.read_bytes()).hexdigest(),
    }
    metadata["metadata_checksum"] = canonical_checksum(metadata)
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lane", choices=("pilot", "calibration", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-profile", type=Path)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must lie in [0, shard count)")
    run_shard(
        CFValidationProtocol.load(args.spec),
        lane=args.lane,
        output=args.output,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        resume=args.resume,
        replications_override=args.replications,
        max_cells=args.max_cells,
        include_stability=not args.skip_stability,
        candidate_profile=(
            None
            if args.candidate_profile is None
            else CFSupportProfile.from_dict(
                json.loads(args.candidate_profile.read_text(encoding="utf-8"))
            )
        ),
    )


if __name__ == "__main__":
    main()
