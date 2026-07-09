"""Outcome-free data and locking primitives for Stage 4 design selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from .declaration import ContrastSpec, DesignDeclaration, JsonLabel, SCOVADeclaration
from .estimator import SCOVA
from .experimental.gates import DiagnosticThresholds, production_thresholds
from .experimental.path import PathDeclaration, fit_path
from .experimental.tilts import geometric_tilt_and_gradient
from .graph import (
    ComparabilityGraphResult,
    PairwiseDiagnosticInput,
    SubsetDiagnosticInput,
    build_comparability_graph,
)
from .inference import SimultaneousInferenceResult, run_direct_influence_inference

SplitAssignment = Literal["design", "estimation"]


def _json_label(value: Any) -> JsonLabel:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    raise TypeError("Group labels and row IDs must be JSON scalar values")


def _canonical_json(values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("design_metadata must be a mapping")
    try:
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError("design_metadata must be JSON serializable") from error
    if not isinstance(decoded, dict):
        raise TypeError("design_metadata must encode as a JSON object")
    return decoded


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OutcomeFreeDesignData:
    """Validated design input containing only covariates, group labels, and IDs.

    There is deliberately no dataframe constructor and no outcome attribute.
    Passing an outcome-bearing object is therefore structurally impossible at
    this boundary; the later outcome-analysis API must receive outcomes
    separately and only after a lock is created.
    """

    covariates: np.ndarray
    groups: tuple[JsonLabel, ...]
    row_ids: tuple[JsonLabel, ...]

    def __post_init__(self) -> None:
        if hasattr(self.covariates, "columns"):
            raise TypeError(
                "OutcomeFreeDesignData accepts an array, not a dataframe; "
                "supply covariate columns explicitly"
            )
        matrix = np.asarray(self.covariates, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("covariates must be a two-dimensional array with a column")
        if len(matrix) < 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("covariates must contain at least two finite rows")
        groups = tuple(_json_label(value) for value in self.groups)
        row_ids = tuple(_json_label(value) for value in self.row_ids)
        if len(groups) != len(matrix) or len(row_ids) != len(matrix):
            raise ValueError("groups and row_ids must align with covariate rows")
        if len(set(groups)) < 2:
            raise ValueError("design data requires at least two observed groups")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("row_ids must be unique")
        stored = np.array(matrix, dtype=float, copy=True, order="C")
        stored.setflags(write=False)
        object.__setattr__(self, "covariates", stored)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "row_ids", row_ids)

    @classmethod
    def from_arrays(
        cls,
        covariates: np.ndarray | Sequence[Sequence[float]],
        groups: Sequence[JsonLabel],
        *,
        row_ids: Sequence[JsonLabel] | None = None,
    ) -> OutcomeFreeDesignData:
        if hasattr(covariates, "columns"):
            raise TypeError(
                "OutcomeFreeDesignData accepts an array, not a dataframe; "
                "supply covariate columns explicitly"
            )
        matrix = np.asarray(covariates, dtype=float)
        ids: Sequence[JsonLabel] = tuple(range(len(matrix))) if row_ids is None else row_ids
        return cls(matrix, tuple(groups), tuple(ids))

    @property
    def n_observations(self) -> int:
        return len(self.groups)

    @property
    def n_covariates(self) -> int:
        return self.covariates.shape[1]

    @property
    def data_hash(self) -> str:
        digest = sha256()
        digest.update(np.ascontiguousarray(self.covariates, dtype="<f8").tobytes())
        digest.update(
            json.dumps(
                {"groups": self.groups, "row_ids": self.row_ids},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DesignLock:
    """Tamper-evident binding of a declaration to outcome-free design choices."""

    declaration_hash: str
    data_hash: str
    row_ids: tuple[JsonLabel, ...]
    split_assignments: tuple[SplitAssignment, ...]
    design_metadata: Mapping[str, Any]
    lock_hash: str

    def __post_init__(self) -> None:
        row_ids = tuple(_json_label(value) for value in self.row_ids)
        assignments = tuple(self.split_assignments)
        metadata = _canonical_json(self.design_metadata)
        if len(row_ids) == 0 or len(set(row_ids)) != len(row_ids):
            raise ValueError("lock row_ids must be nonempty and unique")
        if len(assignments) != len(row_ids):
            raise ValueError("split assignments must align with lock row_ids")
        if set(assignments).difference({"design", "estimation"}):
            raise ValueError("split assignments must be 'design' or 'estimation'")
        if "design" not in assignments or "estimation" not in assignments:
            raise ValueError("a lock requires both design and estimation rows")
        payload = {
            "declaration_hash": self.declaration_hash,
            "data_hash": self.data_hash,
            "row_ids": list(row_ids),
            "split_assignments": list(assignments),
            "design_metadata": metadata,
        }
        expected = _hash_payload(payload)
        if self.lock_hash != expected:
            raise ValueError("design lock checksum is invalid")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "split_assignments", assignments)
        object.__setattr__(self, "design_metadata", metadata)

    @classmethod
    def create(
        cls,
        declaration: DesignDeclaration,
        data: OutcomeFreeDesignData,
        split_assignments: Sequence[SplitAssignment],
        *,
        design_metadata: Mapping[str, Any],
    ) -> DesignLock:
        assignments = tuple(split_assignments)
        metadata = _canonical_json(design_metadata)
        payload = {
            "declaration_hash": declaration.declaration_hash,
            "data_hash": data.data_hash,
            "row_ids": list(data.row_ids),
            "split_assignments": list(assignments),
            "design_metadata": metadata,
        }
        return cls(
            declaration_hash=declaration.declaration_hash,
            data_hash=data.data_hash,
            row_ids=data.row_ids,
            split_assignments=assignments,
            design_metadata=metadata,
            lock_hash=_hash_payload(payload),
        )

    @property
    def estimation_row_ids(self) -> tuple[JsonLabel, ...]:
        return tuple(
            row_id
            for row_id, assignment in zip(self.row_ids, self.split_assignments, strict=True)
            if assignment == "estimation"
        )

    @property
    def design_row_ids(self) -> tuple[JsonLabel, ...]:
        return tuple(
            row_id
            for row_id, assignment in zip(self.row_ids, self.split_assignments, strict=True)
            if assignment == "design"
        )

    def verify(self, declaration: DesignDeclaration, data: OutcomeFreeDesignData) -> None:
        """Raise if this lock does not bind the exact declaration and design data."""
        if declaration.declaration_hash != self.declaration_hash:
            raise ValueError("design declaration does not match the locked declaration")
        if data.data_hash != self.data_hash or data.row_ids != self.row_ids:
            raise ValueError("outcome-free design data do not match the lock")

    def to_dict(self) -> dict[str, Any]:
        return {
            "declaration_hash": self.declaration_hash,
            "data_hash": self.data_hash,
            "row_ids": list(self.row_ids),
            "split_assignments": list(self.split_assignments),
            "design_metadata": self.design_metadata,
            "lock_hash": self.lock_hash,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DesignLock:
        return cls(
            declaration_hash=str(values["declaration_hash"]),
            data_hash=str(values["data_hash"]),
            row_ids=tuple(values["row_ids"]),
            split_assignments=tuple(values["split_assignments"]),
            design_metadata=values["design_metadata"],
            lock_hash=str(values["lock_hash"]),
        )


def _canonical_labels(values: Sequence[JsonLabel]) -> tuple[JsonLabel, ...]:
    def key(value: JsonLabel) -> tuple[int, Any]:
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float)):
            return (1, float(value))
        return (2, value)

    return tuple(sorted(set(values), key=key))


def _stratified_outer_split(data: OutcomeFreeDesignData, declaration: DesignDeclaration) -> tuple[SplitAssignment, ...]:
    """Deterministically assign each observed group to both outer partitions."""
    assignments = np.empty(data.n_observations, dtype=object)
    groups = np.asarray(data.groups, dtype=object)
    for group in _canonical_labels(data.groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            raise ValueError("every group requires two observations for an outer split")
        hashes = np.array(
            [
                int(
                    sha256(f"{declaration.random_state}:{data.row_ids[index]}".encode()).hexdigest(),
                    16,
                )
                for index in indices
            ],
            dtype=object,
        )
        order = indices[np.argsort(hashes, kind="stable")]
        count = min(max(1, round(len(order) * declaration.design_fraction)), len(order) - 1)
        assignments[order[:count]] = "design"
        assignments[order[count:]] = "estimation"
    return tuple(assignments.tolist())  # type: ignore[return-value]


def _design_diagnostics(
    x: np.ndarray,
    groups: np.ndarray,
    labels: tuple[JsonLabel, ...],
    propensity: np.ndarray,
    active: tuple[JsonLabel, ...],
    lambdas: tuple[float, ...],
) -> dict[str, Any]:
    """Emit the design-safe subset of Stage 3's path_gate_grid schema."""
    codes = np.array([labels.index(value) for value in groups], dtype=int)
    active_codes = tuple(labels.index(value) for value in active)
    tilt, _ = geometric_tilt_and_gradient(propensity, np.asarray(lambdas), active_codes)
    denominator = tilt.sum(axis=0)
    target_ess = np.square(denominator) / np.square(tilt).sum(axis=0)
    group_ess = np.empty((len(lambdas), len(labels)))
    balance = np.empty((len(lambdas), len(labels)))
    scale = np.where(x.std(axis=0, ddof=1) > 0, x.std(axis=0, ddof=1), 1.0)
    target_mean = (tilt / denominator[None, :]).T @ x
    for code in range(len(labels)):
        raw = (codes == code)[:, None] * tilt / propensity[:, code, None]
        sums = raw.sum(axis=0)
        group_ess[:, code] = np.square(sums) / np.square(raw).sum(axis=0)
        normalized = raw / sums[None, :]
        means = normalized.T @ x
        balance[:, code] = np.max(np.abs((means - target_mean) / scale), axis=1)
    observed = np.eye(len(labels))[codes]
    calibration = float(np.max(np.abs(propensity.mean(axis=0) - observed.mean(axis=0))))
    return {
        "path_gate_grid": {
            "schema_version": 1,
            "lambdas": list(lambdas),
            "target_ess_ratio": (target_ess / len(x)).tolist(),
            "group_effective_sample_size": group_ess.tolist(),
            "target_weight_concentration": (
                np.sort(tilt / denominator[None, :], axis=0)[-max(1, int(np.ceil(.01 * len(x)))) :].sum(axis=0)
            ).tolist(),
            "maximum_weighted_covariate_imbalance": balance.tolist(),
            "normalization_finite": np.isfinite(denominator).tolist(),
        },
        "propensity_quantiles": {
            str(label): {"q01": float(np.quantile(propensity[:, code], .01))}
            for code, label in enumerate(labels)
        },
        "propensity_calibration": {"worst_class_expected_calibration_error": calibration},
        "crossfit_instability": 0.0,
    }


@dataclass(slots=True)
class SCOVADesignResult:
    """Locked outcome-free design artifact retained in memory for analysis."""

    declaration: DesignDeclaration
    data: OutcomeFreeDesignData
    lock: DesignLock
    graph: ComparabilityGraphResult
    selected_family: tuple[tuple[str, tuple[JsonLabel, ...], float], ...]
    diagnostics: dict[str, Any]

    def design_report(self) -> dict[str, Any]:
        return {
            "design_lock": self.lock.lock_hash,
            "threshold_version": self.graph.threshold_version,
            "threshold_calibrated": self.graph.threshold_calibrated,
            "supported_edges": [list(edge) for edge in self.graph.supported_edges],
            "supported_hyperedges": [list(edge.groups) for edge in self.graph.supported_maximal_hyperedges],
            "selected_family": [
                {"name": name, "groups": list(groups), "lambda": lam}
                for name, groups, lam in self.selected_family
            ],
        }

    def save(self, path: str | Path) -> None:
        """Persist a design artifact without serializing raw covariates or outcomes."""
        Path(path).write_text(
            json.dumps(self.design_report(), sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class SCOVAGraphResult:
    """Held-out graph-conditional analysis with interpretation kept separate."""

    design_lock: str
    interpretation: str
    reliability: str
    inference: SimultaneousInferenceResult | None
    refused: tuple[str, ...]

    def report(self) -> dict[str, Any]:
        return {
            "design_lock": self.design_lock,
            "interpretation": self.interpretation,
            "reliability": self.reliability,
            "inference_scope": "graph-conditional-outer-split",
            "refused": list(self.refused),
            "inference": None if self.inference is None else self.inference.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.report(), sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )


class SCOVADesign:
    """Stage 4 outer-split design/analysis firewall."""

    def __init__(
        self,
        *,
        propensity_model: BaseEstimator | None = None,
        outcome_model: BaseEstimator | None = None,
        thresholds: DiagnosticThresholds | None = None,
    ) -> None:
        self.propensity_model = propensity_model or LogisticRegression(max_iter=2000)
        self.outcome_model = outcome_model
        self.thresholds = thresholds or production_thresholds()

    def prepare_design(
        self, data: OutcomeFreeDesignData, declaration: DesignDeclaration
    ) -> SCOVADesignResult:
        assignments = _stratified_outer_split(data, declaration)
        design_mask = np.asarray(assignments) == "design"
        x = data.covariates[design_mask]
        groups = np.asarray(data.groups, dtype=object)[design_mask]
        labels = _canonical_labels(data.groups)
        codes = np.array([labels.index(value) for value in groups], dtype=int)
        if any(np.sum(codes == code) < declaration.n_splits for code in range(len(labels))):
            raise ValueError("each design-split group requires at least n_splits observations")
        model = clone(self.propensity_model)
        model.fit(x, codes)
        probability = np.asarray(model.predict_proba(x), dtype=float)
        aligned = np.empty((len(x), len(labels)))
        for column, code in enumerate(np.asarray(model.classes_, dtype=int)):
            aligned[:, code] = probability[:, column]
        pair_inputs = {
            pair: PairwiseDiagnosticInput(labels, _design_diagnostics(x, groups, labels, aligned, pair, declaration.lambdas))
            for pair in __import__("itertools").combinations(labels, 2)
        }
        subset_inputs = {
            tuple(subset): SubsetDiagnosticInput(
                labels,
                _design_diagnostics(x, groups, labels, aligned, tuple(subset), declaration.lambdas),
            )
            for subset in declaration.candidate_subsets
            if len(subset) > 2
        }
        graph = build_comparability_graph(
            declaration, labels, pair_inputs, subset_diagnostics=subset_inputs, thresholds=self.thresholds
        )
        selected: list[tuple[str, tuple[JsonLabel, ...], float]] = []
        for edge in graph.edges:
            for lam in edge.supported_lambdas:
                selected.append((f"{edge.groups[0]} - {edge.groups[1]} @ λ={lam:.2f}", edge.groups, lam))
        for edge in graph.supported_maximal_hyperedges:
            for lam in edge.supported_lambdas:
                selected.append((f"omnibus[{','.join(map(str, edge.groups))}] @ λ={lam:.2f}", edge.groups, lam))
        metadata = {"graph": graph_to_dict(graph), "selected_family": selected}
        lock = DesignLock.create(declaration, data, assignments, design_metadata=metadata)
        return SCOVADesignResult(declaration, data, lock, graph, tuple(selected), {"design_rows": int(design_mask.sum())})

    def analyze_outcomes(
        self,
        design: SCOVADesignResult,
        outcomes: Sequence[float],
        *,
        row_ids: Sequence[JsonLabel],
        confidence_level: float = .95,
        n_bootstrap: int = 1999,
        random_state: int | None = None,
    ) -> SCOVAGraphResult:
        design.lock.verify(design.declaration, design.data)
        expected_ids = design.lock.estimation_row_ids
        if set(row_ids) != set(expected_ids) or len(row_ids) != len(expected_ids):
            raise ValueError("outcome row_ids must exactly match the locked estimation rows")
        values = np.asarray(outcomes, dtype=float)
        if values.shape != (len(row_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("outcomes must be a finite vector aligned with row_ids")
        outcome_by_id = dict(zip(row_ids, values, strict=True))
        positions = [design.data.row_ids.index(row_id) for row_id in expected_ids]
        x = design.data.covariates[positions]
        groups = [design.data.groups[position] for position in positions]
        frame = pd.DataFrame(x, columns=design.declaration.covariates)
        frame[design.declaration.group] = groups
        frame["__scova_outcome__"] = [outcome_by_id[row_id] for row_id in expected_ids]
        labels = _canonical_labels(design.data.groups)
        base = SCOVADeclaration(
            "__scova_outcome__", design.declaration.group, design.declaration.covariates,
            interpretation=design.declaration.interpretation, n_splits=design.declaration.n_splits,
            random_state=design.declaration.random_state,
        )
        flattened: list[tuple[str, float, float, np.ndarray]] = []
        for name, subset, lam in design.selected_family:
            target = "pairwise" if len(subset) == 2 else "subset"
            path = fit_path(
                frame,
                PathDeclaration(base, lambdas=(0.0, lam, 1.0) if lam not in (0.0, 1.0) else (0.0, 1.0), target=target, active_groups=subset),
                estimator=SCOVA(propensity_model=self.propensity_model, outcome_model=self.outcome_model),
            )
            if len(subset) == 2:
                contrast = next(iter(path.contrasts.values()))
                index = int(np.where(np.isclose(path.lambdas, lam))[0][0])
                flattened.append((name, float(contrast.estimates[index]), float(contrast.standard_errors[index]), contrast.influence_values[:, index]))
            else:
                # The maximal-hyperedge omnibus statistic is represented by its pairwise basis.
                for contrast in path.contrasts.values():
                    index = int(np.where(np.isclose(path.lambdas, lam))[0][0])
                    flattened.append((f"{name}: {contrast.name}", float(contrast.estimates[index]), float(contrast.standard_errors[index]), contrast.influence_values[:, index]))
        if not flattened:
            return SCOVAGraphResult(design.lock.lock_hash, design.declaration.interpretation, "refused", None, ("no graph-supported contrasts",))
        family = tuple(item[0] for item in flattened)
        inference = run_direct_influence_inference(
            family=family,
            estimates=np.array([item[1] for item in flattened]),
            standard_errors=np.array([item[2] for item in flattened]),
            influence_values=np.column_stack([item[3] for item in flattened]),
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            random_state=design.declaration.random_state if random_state is None else random_state,
            batch_size=256,
        )
        reliability = "certified-overlap-only" if self.thresholds.calibrated else "exploratory-only"
        return SCOVAGraphResult(design.lock.lock_hash, design.declaration.interpretation, reliability, inference, ())


def graph_to_dict(graph: ComparabilityGraphResult) -> dict[str, Any]:
    return {
        "group_labels": list(graph.group_labels), "lambdas": list(graph.lambdas),
        "edges": [{"groups": list(edge.groups), "supported_lambdas": list(edge.supported_lambdas), "decisions": [decision.to_dict() for decision in edge.decisions], "refusal_reasons": list(edge.refusal_reasons)} for edge in graph.edges],
        "hyperedges": [{"groups": list(edge.groups), "supported_lambdas": list(edge.supported_lambdas), "decisions": [decision.to_dict() for decision in edge.decisions], "refusal_reasons": list(edge.refusal_reasons)} for edge in graph.hyperedges],
        "maximal_pairwise_cliques": [list(item) for item in graph.maximal_pairwise_cliques],
    }
