"""Deterministic, artifact-gated Stage 4 graph-firewall campaign runner.

The runner is deliberately independent of GitHub Actions: a shard is a complete,
checksummed record of fixed ``(cell, repetition)`` work items.  This makes a
failed worker observable and rerunnable without changing its seed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy

import scova
from scova import DesignDeclaration, OutcomeFreeDesignData, SCOVADesign
from scova.experimental.gates import DiagnosticThresholds

Scenario = Literal[
    "global_null",
    "sparse_pairwise_signal",
    "strong_complete_graph",
    "rare_group",
    "pairwise_without_kway",
]

SCHEMA_VERSION = 3


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage4Cell:
    """An ordered frozen DGP configuration; its index is its canonical identity."""

    id: str
    scenario: Scenario
    n: int
    n_groups: int
    p: int
    expected_outcome: Literal["inferential", "refusal"] = "inferential"


@dataclass(frozen=True, slots=True)
class Stage4Data:
    covariates: np.ndarray
    groups: tuple[str, ...]
    outcomes: np.ndarray
    row_ids: tuple[int, ...]
    group_effects: dict[str, float]
    true_pairs: tuple[tuple[str, str], ...]
    true_hyperedges: tuple[tuple[str, ...], ...]


def frozen_cells(tier: str, specification: dict[str, Any]) -> tuple[Stage4Cell, ...]:
    """Build the source-controlled ordered catalog declared by the frozen spec."""
    tier_spec = specification["tiers"][tier]
    explicit = specification.get("catalogs", {}).get(tier)
    if explicit is not None:
        cells = tuple(Stage4Cell(**dict(item)) for item in explicit)
        if len(cells) != int(tier_spec["cells"]) or len({cell.id for cell in cells}) != len(cells):
            raise ValueError("explicit Stage 4 catalog has an invalid size or duplicate IDs")
        return cells
    count = int(tier_spec["cells"])
    scenarios: tuple[Scenario, ...] = tuple(tier_spec["scenarios"])
    if not scenarios or count < 1:
        raise ValueError("frozen Stage 4 catalog requires cells and scenarios")
    sizes = tuple(int(value) for value in tier_spec.get("n", (500,)))
    groups = tuple(int(value) for value in tier_spec.get("n_groups", (2, 4)))
    dimensions = tuple(int(value) for value in tier_spec.get("p", (5,)))
    source = tuple(itertools.product(scenarios, sizes, groups, dimensions))
    cells = tuple(
        Stage4Cell(
            id=f"{tier}-{index:03d}",
            scenario=values[0],
            n=values[1],
            n_groups=values[2],
            p=values[3],
        )
        for index, values in enumerate(source[index % len(source)] for index in range(count))
    )
    if len({cell.id for cell in cells}) != count:
        raise RuntimeError("Stage 4 catalog IDs must be unique")
    return cells


def load_specification(path: Path) -> dict[str, Any]:
    """Load the frozen protocol together with its explicit source-controlled catalog."""
    specification = json.loads(path.read_text(encoding="utf-8"))
    catalog_name = specification.get("cell_catalog")
    if catalog_name:
        catalog_path = path.with_name(str(catalog_name))
        catalog_bytes = catalog_path.read_bytes()
        catalog = json.loads(catalog_bytes)
        if catalog.get("base_catalog"):
            base_path = path.with_name(str(catalog["base_catalog"]))
            base_bytes = base_path.read_bytes()
            base = json.loads(base_bytes)
            catalog = {
                **base,
                "overrides": catalog.get("overrides", {}),
                "transforms": catalog.get("transforms", []),
                "schema_version": catalog.get("schema_version"),
            }
            catalog_bytes += base_bytes
        if catalog.get("schema_version") != 1 or not isinstance(catalog.get("catalogs"), dict):
            raise ValueError("Stage 4 cell catalog has an unsupported schema")
        overrides = catalog.get("overrides", {})
        catalogs: dict[str, list[dict[str, Any]]] = {}
        for tier, items in catalog["catalogs"].items():
            rendered: list[dict[str, Any]] = []
            for source_item in items:
                item = {**source_item, **overrides.get(source_item["scenario"], {})}
                for transform in catalog.get("transforms", []):
                    when = transform["when"]
                    matches = all(
                        item.get(key) == value
                        for key, value in when.items()
                        if key != "scenario_not"
                    ) and (
                        "scenario_not" not in when or item["scenario"] != when["scenario_not"]
                    )
                    if matches:
                        item = {**item, **transform["set"]}
                rendered.append(item)
            catalogs[tier] = rendered
        specification["catalogs"] = catalogs
        specification["catalog_definition_sha256"] = sha256(catalog_bytes).hexdigest()
    return specification


def work_items(cells: tuple[Stage4Cell, ...], repetitions: int) -> tuple[tuple[int, int], ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return tuple(
        (cell_index, repetition)
        for cell_index in range(len(cells))
        for repetition in range(repetitions)
    )


def shard_items(
    cells: tuple[Stage4Cell, ...], repetitions: int, shard_index: int, shard_count: int
) -> tuple[tuple[int, int], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")
    return tuple(
        item
        for index, item in enumerate(work_items(cells, repetitions))
        if index % shard_count == shard_index
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def generate_stage4_data(cell: Stage4Cell, seed: int) -> Stage4Data:
    """Generate covariates, observed groups/outcomes, and outcome-free support truth."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(cell.n, cell.p))
    labels = tuple(f"g{index}" for index in range(cell.n_groups))
    if cell.scenario == "strong_complete_graph":
        # A deliberately strong, complete-overlap target: group assignment is
        # balanced and independent of covariates, so every pair is supported.
        codes = np.arange(cell.n) % cell.n_groups
        rng.shuffle(codes)
        outcomes = 0.6 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(size=cell.n)
        return Stage4Data(
            covariates=x,
            groups=tuple(labels[code] for code in codes),
            outcomes=outcomes,
            row_ids=tuple(range(cell.n)),
            group_effects=dict.fromkeys(labels, 0.0),
            true_pairs=tuple(itertools.combinations(labels, 2)),
            true_hyperedges=(tuple(labels),) if len(labels) >= 3 else (),
        )
    if cell.scenario == "pairwise_without_kway":
        if cell.n_groups != 3:
            raise ValueError("pairwise-without-K-way cells require exactly three groups")
        codes = np.arange(cell.n) % 3
        rng.shuffle(codes)
        support = ((0.0, 1.0), (1.0, 2.0), (0.0, 2.0))
        x[:, 0] = np.array([rng.choice(support[code]) for code in codes])
        effects = np.zeros(3)
        outcomes = 0.6 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(size=cell.n)
        return Stage4Data(
            covariates=x,
            groups=tuple(labels[code] for code in codes),
            outcomes=outcomes,
            row_ids=tuple(range(cell.n)),
            group_effects=dict(zip(labels, effects, strict=True)),
            true_pairs=tuple(itertools.combinations(labels, 2)),
            true_hyperedges=(),
        )
    slopes = np.linspace(-0.7, 0.7, cell.n_groups)
    logits = x[:, [0]] * slopes + 0.3 * x[:, [1]] * slopes[::-1]
    if cell.scenario == "rare_group":
        logits[:, -1] -= 4.0
    if cell.scenario == "pairwise_without_kway" and cell.n_groups >= 3:
        logits[:, 0] -= 1.1 * x[:, 0]
        logits[:, -1] += 1.1 * x[:, 0]
    probabilities = _softmax(logits)
    codes = (rng.random(cell.n)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    if cell.scenario == "rare_group":
        # Exactly two rare rows force one row into each deterministic outer
        # split, exercising the preregistered insufficient-per-split refusal.
        nonrare = _softmax(logits[:, :-1])
        codes = (rng.random(cell.n)[:, None] > np.cumsum(nonrare, axis=1)).sum(axis=1)
        rare_indices = rng.choice(cell.n, size=2, replace=False)
        codes[rare_indices] = cell.n_groups - 1
    effects = np.zeros(cell.n_groups)
    if cell.scenario == "sparse_pairwise_signal":
        effects[-1] = 0.8
    base = 0.6 * x[:, 0] - 0.3 * x[:, 1]
    outcomes = base + effects[codes] + rng.normal(size=cell.n)
    pairs = tuple(itertools.combinations(labels, 2))
    hyperedges = (
        tuple(itertools.combinations(labels, min(3, len(labels))))
        if cell.scenario in ("global_null", "sparse_pairwise_signal", "strong_complete_graph")
        and len(labels) >= 3
        else ()
    )
    if cell.scenario == "rare_group":
        pairs = tuple(pair for pair in pairs if labels[-1] not in pair)
    if cell.scenario == "pairwise_without_kway":
        hyperedges = ()
    return Stage4Data(
        covariates=x,
        groups=tuple(labels[code] for code in codes),
        outcomes=outcomes,
        row_ids=tuple(range(cell.n)),
        group_effects=dict(zip(labels, effects, strict=True)),
        true_pairs=pairs,
        true_hyperedges=hyperedges,
    )


def _all_subsets(labels: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset
        for size in range(3, min(4, len(labels)) + 1)
        for subset in itertools.combinations(labels, size)
    )


def _truth_for_name(name: str, effects: dict[str, float]) -> float:
    contrast = name.rsplit(" @ lambda=", 1)[0]
    if " - " not in contrast:
        return 0.0
    first, second = contrast.split(" - ", 1)
    return effects[first] - effects[second]


def _normalized_pairs(pairs: Any) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(tuple(sorted(map(str, pair))) for pair in pairs))


def _run_one(
    cell: Stage4Cell,
    seed: int,
    thresholds: DiagnosticThresholds,
    bootstrap: int,
    lambdas: tuple[float, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    tracemalloc.start()
    data = generate_stage4_data(cell, seed)
    labels = tuple(sorted(set(data.groups)))
    design_data = OutcomeFreeDesignData.from_arrays(
        data.covariates, data.groups, row_ids=data.row_ids
    )
    declaration = DesignDeclaration(
        group="group",
        covariates=tuple(f"x{index + 1}" for index in range(cell.p)),
        n_splits=2,
        random_state=seed,
        lambdas=lambdas,
        candidate_subsets=_all_subsets(labels),
    )
    engine = SCOVADesign(thresholds=thresholds)
    design = engine.prepare_design(design_data, declaration)
    estimation_ids = design.lock.estimation_row_ids
    result = engine.analyze_outcomes(
        design,
        [float(data.outcomes[row_id]) for row_id in estimation_ids],
        row_ids=estimation_ids,
        n_bootstrap=bootstrap,
        random_state=seed,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    selected_edges = [list(edge) for edge in design.graph.supported_edges]
    selected_hyperedges = [list(edge.groups) for edge in design.graph.supported_maximal_hyperedges]
    rejected = (
        []
        if result.inference is None
        else [item.name for item in result.inference.contrasts if item.rejected]
    )
    coverage = None
    if result.inference is not None:
        coverage = all(
            contrast.simultaneous_confidence_interval[0]
            <= _truth_for_name(contrast.name, data.group_effects)
            <= contrast.simultaneous_confidence_interval[1]
            for contrast in result.inference.contrasts
        )
    mutation_rejected = False
    try:
        changed = DesignDeclaration(
            "group", declaration.covariates, random_state=seed + 1, n_splits=2
        )
        design.lock.verify(changed, design.data)
    except ValueError:
        mutation_rejected = True
    return {
        "status": "completed",
        "selected_edges": selected_edges,
        "selected_hyperedges": selected_hyperedges,
        "refusal_reasons": list(result.refused),
        "accepted": result.inference is not None,
        "simultaneous_coverage": coverage,
        "any_rejection": bool(rejected),
        "rejected_family": rejected,
        "post_lock_mutation_rejected": mutation_rejected,
        "complete_graph_recovered": _normalized_pairs(selected_edges)
        == _normalized_pairs(data.true_pairs),
        "truth": {
            "pairs": [list(pair) for pair in data.true_pairs],
            "hyperedges": [list(edge) for edge in data.true_hyperedges],
            "global_null": cell.scenario == "global_null",
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak,
    }


def _failure(error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "accepted": False,
        "simultaneous_coverage": None,
        "any_rejection": False,
        "selected_edges": [],
        "selected_hyperedges": [],
        "post_lock_mutation_rejected": False,
        "complete_graph_recovered": False,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _expected_rare_group_refusal(error: ValueError) -> dict[str, Any]:
    """Record the preregistered per-split-count safety refusal as completed work."""
    return {
        "status": "completed",
        "accepted": False,
        "expected_refusal": "insufficient_per_split_group_count",
        "refusal_reasons": [str(error)],
        "simultaneous_coverage": None,
        "any_rejection": False,
        "selected_edges": [],
        "selected_hyperedges": [],
        "post_lock_mutation_rejected": None,
        "complete_graph_recovered": False,
    }


def _is_expected_rare_group_refusal(cell: Stage4Cell, error: Exception) -> bool:
    return (
        cell.scenario == "rare_group"
        and isinstance(error, ValueError)
        and str(error) == "each design-split group requires at least n_splits observations"
    )


def _engineering_smoke_thresholds() -> DiagnosticThresholds:
    """Exercise the complete pipeline without representing smoke as calibration."""
    return DiagnosticThresholds(
        version="stage4-engineering-smoke-v1",
        calibrated=True,
        artifact_sha256="engineering-smoke-not-promotable",
        min_group_ess_warning=1.0,
        min_group_ess_refuse=0.0,
        min_target_ess_ratio_warning=0.0,
        min_target_ess_ratio_refuse=0.0,
        max_influence_share_warning=1.0,
        max_influence_share_refuse=1.0,
        max_weight_concentration_warning=1.0,
        max_weight_concentration_refuse=1.0,
        min_propensity_q01_warning=1e-12,
        min_propensity_q01_refuse=1e-14,
        max_calibration_error_warning=1.0,
        max_calibration_error_refuse=1.0,
        max_balance_warning=1_000.0,
        max_balance_refuse=10_000.0,
        max_crossfit_instability_warning=1.0,
        max_crossfit_instability_refuse=1.0,
    )


def run_campaign(
    *,
    tier: str,
    thresholds_path: Path | None,
    shard_index: int,
    shard_count: int,
    output: Path,
    specification_path: Path | None = None,
) -> dict[str, Any]:
    specification_path = specification_path or (
        Path(__file__).with_name("specs") / "stage4_graph_release.json"
    )
    specification_bytes = specification_path.read_bytes()
    specification = load_specification(specification_path)
    if tier not in specification["tiers"]:
        raise ValueError(f"unknown Stage 4 tier: {tier}")
    if not specification["frozen"]:
        raise ValueError("Stage 4 protocol must be frozen before execution")
    tier_spec = specification["tiers"][tier]
    smoke = tier == "engineering_smoke"
    if smoke:
        thresholds = _engineering_smoke_thresholds()
        threshold_hash = None
    else:
        if thresholds_path is None:
            raise ValueError("Stage 4 campaign requires a Stage 3 threshold artifact")
        thresholds = DiagnosticThresholds.from_calibration_artifact(
            json.loads(thresholds_path.read_text())
        )
        if not thresholds.calibrated:
            raise ValueError("Stage 4 campaign requires calibrated Stage 3 thresholds")
        threshold_hash = thresholds.artifact_sha256
    cells = frozen_cells(tier, specification)
    repetitions, bootstrap = int(tier_spec["repetitions"]), int(tier_spec["bootstrap"])
    items = shard_items(cells, repetitions, shard_index, shard_count)
    seed_namespace = int(specification["seed_namespaces"][tier_spec["seed_namespace"]])
    lambdas = tuple(float(value) for value in specification["target_path"]["lambdas"])
    records = []
    for cell_index, repetition in items:
        seed = seed_namespace + cell_index * 100_000 + repetition
        try:
            result = _run_one(cells[cell_index], seed, thresholds, bootstrap, lambdas)
        except Exception as error:  # retain failures as evidence, never reseed
            result = (
                _expected_rare_group_refusal(error)
                if _is_expected_rare_group_refusal(cells[cell_index], error)
                else _failure(error)
            )
        records.append(
            {
                "cell_index": cell_index,
                "repetition": repetition,
                "seed": seed,
                "cell": asdict(cells[cell_index]),
                **result,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": specification["protocol"],
        "metric_contract": specification.get("metric_contract", "legacy"),
        "tier": tier,
        "specification_sha256": sha256(specification_bytes).hexdigest(),
        "catalog_sha256": _digest([asdict(cell) for cell in cells]),
        "catalog_definition_sha256": specification.get("catalog_definition_sha256"),
        "threshold_artifact_sha256": threshold_hash,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "repetitions": repetitions,
        "bootstrap": bootstrap,
        "seed_namespace": seed_namespace,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scova": scova.__version__,
        },
        "records": records,
    }
    payload["sha256"] = _digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmarks/specs/stage4_graph_release.json")
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_campaign(
        tier=args.tier,
        thresholds_path=args.thresholds,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        output=args.output,
        specification_path=args.spec,
    )


if __name__ == "__main__":
    main()
