"""Outcome-free pairwise comparability graphs from Stage 3 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .declaration import DesignDeclaration, JsonLabel
from .experimental.gates import (
    DiagnosticThresholds,
    GateDecision,
    GateStatus,
    evaluate_design_gates,
)


def _label_sort_key(value: JsonLabel) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, value)


@dataclass(frozen=True, slots=True)
class PairwiseDiagnosticInput:
    """Stage 3's design-safe diagnostics for one pairwise path.

    ``diagnostics`` is the existing Stage 3 diagnostics payload.  Graph
    construction deliberately reads only its (X, A)-measurable fields.
    This adapter lets the later design stage emit the same schema without
    requiring a fitted outcome model.
    """

    group_labels: tuple[JsonLabel, ...]
    diagnostics: Mapping[str, Any]

    @classmethod
    def from_stage3_path(cls, path_result: Any) -> PairwiseDiagnosticInput:
        return cls(tuple(path_result.group_labels), path_result.diagnostics)


@dataclass(frozen=True, slots=True)
class PairwiseEdge:
    """All design evidence for a requested pair, including refusal evidence."""

    groups: tuple[JsonLabel, JsonLabel]
    supported_lambdas: tuple[float, ...]
    decisions: tuple[GateDecision, ...]
    refusal_reasons: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return bool(self.supported_lambdas)


@dataclass(frozen=True, slots=True)
class ComparabilityGraphResult:
    """Deterministic pairwise graph and its complete design-stage evidence."""

    group_labels: tuple[JsonLabel, ...]
    lambdas: tuple[float, ...]
    edges: tuple[PairwiseEdge, ...]
    threshold_version: str
    threshold_calibrated: bool

    @property
    def supported_edges(self) -> tuple[tuple[JsonLabel, JsonLabel], ...]:
        return tuple(edge.groups for edge in self.edges if edge.supported)

    @property
    def refused_edges(self) -> tuple[PairwiseEdge, ...]:
        return tuple(edge for edge in self.edges if not edge.supported)

    def edge_for(self, first: JsonLabel, second: JsonLabel) -> PairwiseEdge:
        target = tuple(sorted((first, second), key=_label_sort_key))
        for edge in self.edges:
            if edge.groups == target:
                return edge
        raise ValueError(f"unknown graph pair: {target}")


def _canonical_pair(values: Sequence[JsonLabel]) -> tuple[JsonLabel, JsonLabel]:
    if len(values) != 2 or len(set(values)) != 2:
        raise ValueError("pairwise diagnostic keys must name two distinct groups")
    return tuple(sorted(values, key=_label_sort_key))  # type: ignore[return-value]


def _required_grid_values(
    source: PairwiseDiagnosticInput,
    pair: tuple[JsonLabel, JsonLabel],
    lambdas: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    diagnostics = source.diagnostics
    grid = diagnostics.get("path_gate_grid")
    if not isinstance(grid, Mapping) or grid.get("schema_version") != 1:
        raise ValueError("pairwise diagnostics require Stage 3 path_gate_grid schema 1")
    observed_lambdas = tuple(float(value) for value in grid.get("lambdas", ()))
    if observed_lambdas != lambdas:
        raise ValueError("pairwise diagnostics lambda grid does not match the declaration")
    labels = tuple(source.group_labels)
    if len(set(labels)) != len(labels) or not set(pair).issubset(labels):
        raise ValueError("pairwise diagnostics do not contain the requested groups")
    indices = [labels.index(label) for label in pair]
    count = len(lambdas)
    target_ess = np.asarray(grid["target_ess_ratio"], dtype=float)
    group_ess = np.asarray(grid["group_effective_sample_size"], dtype=float)
    concentration = np.asarray(grid["target_weight_concentration"], dtype=float)
    balance = np.asarray(grid["maximum_weighted_covariate_imbalance"], dtype=float)
    normalization = np.asarray(grid["normalization_finite"], dtype=bool)
    if (
        target_ess.shape != (count,)
        or concentration.shape != (count,)
        or normalization.shape != (count,)
        or group_ess.shape != (count, len(labels))
        or balance.shape != (count, len(labels))
    ):
        raise ValueError("pairwise diagnostics have incompatible Stage 3 grid shapes")
    quantiles = diagnostics.get("propensity_quantiles")
    calibration = diagnostics.get("propensity_calibration")
    if not isinstance(quantiles, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError("pairwise diagnostics are missing propensity diagnostics")
    try:
        min_q01 = min(float(quantiles[str(label)]["q01"]) for label in pair)
        calibration_error = float(calibration["worst_class_expected_calibration_error"])
        instability = float(diagnostics["crossfit_instability"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pairwise diagnostics contain invalid scalar gate values") from error
    return (
        target_ess,
        group_ess[:, indices],
        concentration,
        balance[:, indices],
        normalization,
        min_q01,
        calibration_error,
        instability,
    )


def build_pairwise_comparability_graph(
    declaration: DesignDeclaration,
    group_labels: Sequence[JsonLabel],
    diagnostics_by_pair: Mapping[tuple[JsonLabel, JsonLabel], PairwiseDiagnosticInput],
    *,
    thresholds: DiagnosticThresholds,
) -> ComparabilityGraphResult:
    """Build pairwise edges from design-safe, finite-grid Stage 3 diagnostics.

    A warning or provisional threshold lock is not sufficient to create an
    edge.  Missing diagnostics become a recorded refusal, whereas malformed
    diagnostics are errors because they indicate an invalid design artifact.
    """
    labels = tuple(sorted(group_labels, key=_label_sort_key))
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("graph construction requires at least two distinct group labels")
    normalized_inputs: dict[tuple[JsonLabel, JsonLabel], PairwiseDiagnosticInput] = {}
    for raw_pair, source in diagnostics_by_pair.items():
        pair = _canonical_pair(raw_pair)
        if pair not in tuple(combinations(labels, 2)):
            raise ValueError(f"diagnostics supplied for unknown pair: {pair}")
        if pair in normalized_inputs:
            raise ValueError(f"duplicate diagnostics supplied for pair: {pair}")
        normalized_inputs[pair] = source

    edges: list[PairwiseEdge] = []
    for pair in combinations(labels, 2):
        source = normalized_inputs.get(pair)
        if source is None:
            edges.append(
                PairwiseEdge(pair, (), (), ("pairwise design diagnostics were not supplied",))
            )
            continue
        (
            target_ess,
            group_ess,
            concentration,
            balance,
            normalization,
            min_q01,
            calibration_error,
            instability,
        ) = _required_grid_values(source, pair, declaration.lambdas)
        decisions = tuple(
            evaluate_design_gates(
                min_group_ess=float(np.min(group_ess[index])),
                target_ess_ratio=float(target_ess[index]),
                max_weight_concentration=float(concentration[index]),
                min_propensity_q01=min_q01,
                max_calibration_error=calibration_error,
                max_balance=float(np.max(balance[index])),
                crossfit_instability=instability,
                numerical_valid=bool(normalization[index]),
                thresholds=thresholds,
            )
            for index in range(len(declaration.lambdas))
        )
        supported = tuple(
            lam
            for lam, decision in zip(declaration.lambdas, decisions, strict=True)
            if decision.status is GateStatus.PASS
        )
        reasons: tuple[str, ...] = ()
        if not supported:
            reasons = ("no declared lambda passed all design gates",) + tuple(
                f"lambda={lam:.6g}: {'; '.join(decision.reasons) or decision.status.value}"
                for lam, decision in zip(declaration.lambdas, decisions, strict=True)
            )
        edges.append(PairwiseEdge(pair, supported, decisions, reasons))
    return ComparabilityGraphResult(
        group_labels=labels,
        lambdas=declaration.lambdas,
        edges=tuple(edges),
        threshold_version=thresholds.version,
        threshold_calibrated=thresholds.calibrated,
    )
