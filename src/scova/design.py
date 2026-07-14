"""Outcome-free data and locking primitives for Stage 4 design selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

from .anchor import (
    AnchoredBoundsResult,
    LipschitzAnchorResult,
    bounded_pairwise_anchor,
    lipschitz_pairwise_anchor,
)
from .declaration import ContrastSpec, DesignDeclaration, JsonLabel, SCOVADeclaration
from .estimator import SCOVA
from .experimental.gates import DiagnosticThresholds, GateDecision, production_thresholds
from .experimental.path import PathDeclaration, fit_path
from .experimental.tilts import geometric_tilt_and_gradient
from .geometry import (
    fit_support_geometry,
    gaussian_reference_transport,
    soft_k_nearest,
)
from .graph import (
    ComparabilityGraphResult,
    PairwiseDiagnosticInput,
    PairwiseEdge,
    SubsetDiagnosticInput,
    SubsetHyperedge,
    build_comparability_graph,
)
from .inference import SimultaneousInferenceResult, run_direct_influence_inference

SplitAssignment = Literal["design", "estimation"]


def _fold_reference_predictions(
    engine: SCOVADesign,
    x: np.ndarray,
    outcomes: np.ndarray,
    group_codes: np.ndarray,
    folds: np.ndarray,
    references: dict[int, np.ndarray],
    neighbor_indices: dict[int, np.ndarray],
    neighbor_weights: dict[int, np.ndarray],
) -> np.ndarray:
    """Predict frozen reference locations with models excluding each query fold."""
    predictions = np.empty((len(x), len(references)), dtype=float)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        for code, reference_x in references.items():
            model = clone(engine.outcome_model) if engine.outcome_model is not None else None
            if model is None:
                model = SCOVA().outcome_model
            model.fit(x[train & (group_codes == code)], outcomes[train & (group_codes == code)])
            values = np.asarray(model.predict(reference_x), dtype=float)
            selected = values[neighbor_indices[code][test]]
            predictions[test, code] = np.sum(selected * neighbor_weights[code][test], axis=1)
    return predictions


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
        return int(self.covariates.shape[1])

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


def _stratified_outer_split(
    data: OutcomeFreeDesignData, declaration: DesignDeclaration
) -> tuple[SplitAssignment, ...]:
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
                    sha256(
                        f"{declaration.random_state}:{data.row_ids[index]}".encode()
                    ).hexdigest(),
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
    return tuple(cast(SplitAssignment, item) for item in assignments.tolist())


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
                np.sort(tilt / denominator[None, :], axis=0)[
                    -max(1, int(np.ceil(0.01 * len(x)))) :
                ].sum(axis=0)
            ).tolist(),
            "maximum_weighted_covariate_imbalance": balance.tolist(),
            "normalization_finite": np.isfinite(denominator).tolist(),
        },
        "propensity_quantiles": {
            str(label): {"q01": float(np.quantile(propensity[:, code], 0.01))}
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
            "supported_hyperedges": [
                list(edge.groups) for edge in self.graph.supported_maximal_hyperedges
            ],
            "selected_family": [
                {"name": name, "groups": list(groups), "lambda": lam}
                for name, groups, lam in self.selected_family
            ],
        }

    def save(self, path: str | Path) -> None:
        """Persist a replayable design artifact without raw covariates or outcomes."""
        payload = {
            "schema_version": 1,
            "declaration": self.declaration.to_dict(),
            "declaration_hash": self.declaration.declaration_hash,
            "lock": self.lock.to_dict(),
            "graph": graph_to_dict(self.graph),
            "selected_family": [
                {"name": name, "groups": list(groups), "lambda": lam}
                for name, groups, lam in self.selected_family
            ],
            "diagnostics": self.diagnostics,
        }
        Path(path).write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        data: OutcomeFreeDesignData,
        declaration: DesignDeclaration | None = None,
    ) -> SCOVADesignResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Stage 4 design artifact schema")
        stored_declaration = DesignDeclaration.from_dict(payload["declaration"])
        active_declaration = stored_declaration if declaration is None else declaration
        if active_declaration.declaration_hash != payload["declaration_hash"]:
            raise ValueError("supplied declaration does not match the design artifact")
        lock = DesignLock.from_dict(payload["lock"])
        lock.verify(active_declaration, data)
        selected = tuple(
            (str(item["name"]), tuple(item["groups"]), float(item["lambda"]))
            for item in payload["selected_family"]
        )
        return cls(
            active_declaration,
            data,
            lock,
            graph_from_dict(payload["graph"]),
            selected,
            dict(payload["diagnostics"]),
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

    @classmethod
    def load(cls, path: str | Path) -> SCOVAGraphResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        inference = payload.get("inference")
        return cls(
            design_lock=str(payload["design_lock"]),
            interpretation=str(payload["interpretation"]),
            reliability=str(payload["reliability"]),
            inference=None
            if inference is None
            else SimultaneousInferenceResult.from_dict(inference),
            refused=tuple(payload.get("refused", [])),
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
            pair: PairwiseDiagnosticInput(
                labels, _design_diagnostics(x, groups, labels, aligned, pair, declaration.lambdas)
            )
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
            declaration,
            labels,
            pair_inputs,
            subset_diagnostics=subset_inputs,
            thresholds=self.thresholds,
        )
        selected: list[tuple[str, tuple[JsonLabel, ...], float]] = []
        for edge in graph.edges:
            for lam in edge.supported_lambdas:
                selected.append(
                    (f"{edge.groups[0]} - {edge.groups[1]} @ λ={lam:.2f}", edge.groups, lam)
                )
        for hyperedge in graph.supported_maximal_hyperedges:
            for lam in hyperedge.supported_lambdas:
                selected.append(
                    (
                        f"omnibus[{','.join(map(str, hyperedge.groups))}] @ λ={lam:.2f}",
                        hyperedge.groups,
                        lam,
                    )
                )
        # Rebuild from the locked graph using either declared exact-support
        # contrasts or the default pairwise comparison.  The earlier entries
        # are intentionally discarded so display labels cannot affect selection.
        selected = []
        for edge in graph.edges:
            names = [
                contrast.name
                for contrast in declaration.contrasts
                if {label for label, weight in contrast.weights if abs(weight) > 1e-15}
                == set(edge.groups)
            ] or [f"{edge.groups[0]} - {edge.groups[1]}"]
            for lam in edge.supported_lambdas:
                selected.extend((f"{name} @ lambda={lam:.2f}", edge.groups, lam) for name in names)
        for hyperedge in graph.supported_maximal_hyperedges:
            names = [
                contrast.name
                for contrast in declaration.contrasts
                if {label for label, weight in contrast.weights if abs(weight) > 1e-15}
                == set(hyperedge.groups)
            ]
            for lam in hyperedge.supported_lambdas:
                selected.extend(
                    (f"{name} @ lambda={lam:.2f}", hyperedge.groups, lam) for name in names
                )
        metadata: dict[str, Any] = {"graph": graph_to_dict(graph), "selected_family": selected}
        anchor = declaration.anchored_bounds
        if anchor is not None and anchor.support_geometry is not None:
            metadata["support_geometry"] = fit_support_geometry(
                data.covariates,
                data.groups,
                data.row_ids,
                design_mask,
                anchor.support_geometry,
            )
        lock = DesignLock.create(declaration, data, assignments, design_metadata=metadata)
        return SCOVADesignResult(
            declaration, data, lock, graph, tuple(selected), {"design_rows": int(design_mask.sum())}
        )

    def analyze_outcomes(
        self,
        design: SCOVADesignResult,
        outcomes: Sequence[float],
        *,
        row_ids: Sequence[JsonLabel],
        confidence_level: float = 0.95,
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
        base = SCOVADeclaration(
            "__scova_outcome__",
            design.declaration.group,
            design.declaration.covariates,
            interpretation=design.declaration.interpretation,
            n_splits=design.declaration.n_splits,
            random_state=design.declaration.random_state,
            contrasts=design.declaration.contrasts,
        )
        flattened: list[tuple[str, float, float, np.ndarray]] = []
        grouped: dict[tuple[JsonLabel, ...], list[tuple[str, float]]] = {}
        for name, subset, lam in design.selected_family:
            grouped.setdefault(subset, []).append((name, lam))
        for subset, requests in grouped.items():
            target: Literal["pairwise", "subset"] = "pairwise" if len(subset) == 2 else "subset"
            lambdas = tuple(sorted({0.0, 1.0, *(lam for _, lam in requests)}))
            path = fit_path(
                frame,
                PathDeclaration(base, lambdas=lambdas, target=target, active_groups=subset),
                estimator=SCOVA(
                    propensity_model=self.propensity_model, outcome_model=self.outcome_model
                ),
            )
            for name, lam in requests:
                contrast_name = name.rsplit(" @ lambda=", maxsplit=1)[0]
                if contrast_name not in path.contrasts:
                    raise ValueError(
                        f"locked contrast {contrast_name!r} is incompatible with subset {subset}"
                    )
                contrast = path.contrasts[contrast_name]
                index = int(np.where(np.isclose(path.lambdas, lam))[0][0])
                flattened.append(
                    (
                        name,
                        float(contrast.estimates[index]),
                        float(contrast.standard_errors[index]),
                        contrast.influence_values[:, index],
                    )
                )
        if not flattened:
            return SCOVAGraphResult(
                design.lock.lock_hash,
                design.declaration.interpretation,
                "refused",
                None,
                ("no graph-supported contrasts",),
            )
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
        return SCOVAGraphResult(
            design.lock.lock_hash, design.declaration.interpretation, reliability, inference, ()
        )

    def analyze_anchored_bounds(
        self,
        design: SCOVADesignResult,
        outcomes: Sequence[float],
        *,
        row_ids: Sequence[JsonLabel],
        confidence_level: float = 0.95,
    ) -> AnchoredBoundsResult:
        """Run the locked Stage 5A B1 analysis for graph-supported pairs only."""
        design.lock.verify(design.declaration, design.data)
        anchor = design.declaration.anchored_bounds
        seed = design.declaration.random_state
        if anchor is None:
            return AnchoredBoundsResult(
                design.lock.lock_hash,
                design.declaration.interpretation,
                "refused",
                "bounded_outcome",
                "scaled_harmonic_overlap",
                None,
                None,
                confidence_level,
                seed,
                (),
                ("anchored outcome bounds were not declared before design locking",),
            )
        expected_ids = design.lock.estimation_row_ids
        if set(row_ids) != set(expected_ids) or len(row_ids) != len(expected_ids):
            raise ValueError("outcome row_ids must exactly match the locked estimation rows")
        values = np.asarray(outcomes, dtype=float)
        if values.shape != (len(row_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("outcomes must be a finite vector aligned with row_ids")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between 0 and 1")
        if np.any(values < anchor.outcome_lower) or np.any(values > anchor.outcome_upper):
            return AnchoredBoundsResult(
                design.lock.lock_hash,
                design.declaration.interpretation,
                "refused",
                anchor.assumption,
                anchor.support_weight,
                anchor.outcome_lower,
                anchor.outcome_upper,
                confidence_level,
                seed,
                (),
                ("held-out outcomes fall outside the declared bounded-outcome range",),
            )
        pairs = tuple(edge.groups for edge in design.graph.edges if edge.supported)
        if not pairs:
            return AnchoredBoundsResult(
                design.lock.lock_hash,
                design.declaration.interpretation,
                "refused",
                anchor.assumption,
                anchor.support_weight,
                anchor.outcome_lower,
                anchor.outcome_upper,
                confidence_level,
                seed,
                (),
                ("no graph-supported pairwise contrasts are available",),
            )
        outcome_by_id = dict(zip(row_ids, values, strict=True))
        positions = [design.data.row_ids.index(row_id) for row_id in expected_ids]
        x = design.data.covariates[positions]
        groups = [design.data.groups[position] for position in positions]
        frame = pd.DataFrame(x, columns=design.declaration.covariates)
        frame[design.declaration.group] = groups
        frame["__scova_outcome__"] = [outcome_by_id[row_id] for row_id in expected_ids]
        labels = _canonical_labels(design.data.groups)
        contrasts = tuple(
            ContrastSpec(f"{pair[0]} - {pair[1]}", ((pair[0], 1.0), (pair[1], -1.0)))
            for pair in pairs
        )
        base = SCOVADeclaration(
            "__scova_outcome__",
            design.declaration.group,
            design.declaration.covariates,
            interpretation=design.declaration.interpretation,
            n_splits=design.declaration.n_splits,
            random_state=seed,
            contrasts=contrasts,
        )
        fitted = SCOVA(
            propensity_model=self.propensity_model, outcome_model=self.outcome_model
        ).fit(frame, base)
        observed_codes = np.array([labels.index(group) for group in groups], dtype=int)
        results = tuple(
            bounded_pairwise_anchor(
                groups=pair,
                group_codes=observed_codes,
                outcomes=values,
                propensity=fitted.propensity_predictions,
                outcome_predictions=fitted.outcome_predictions,
                active_codes=(labels.index(pair[0]), labels.index(pair[1])),
                outcome_lower=anchor.outcome_lower,
                outcome_upper=anchor.outcome_upper,
                confidence_level=confidence_level,
            )
            for pair in pairs
        )
        return AnchoredBoundsResult(
            design.lock.lock_hash,
            design.declaration.interpretation,
            "interval-only",
            anchor.assumption,
            anchor.support_weight,
            anchor.outcome_lower,
            anchor.outcome_upper,
            confidence_level,
            seed,
            results,
            (),
        )

    def analyze_lipschitz_anchors(
        self,
        design: SCOVADesignResult,
        outcomes: Sequence[float],
        *,
        row_ids: Sequence[JsonLabel],
    ) -> LipschitzAnchorResult:
        """Run experimental B2 bounds using only geometry frozen in the design lock."""
        design.lock.verify(design.declaration, design.data)
        anchor = design.declaration.anchored_bounds
        if anchor is None or anchor.support_geometry is None:
            return LipschitzAnchorResult(
                design.lock.lock_hash, None, "refused", None, None, np.array([]), (),
                ("a locked Stage 5B support geometry declaration is required",),
            )
        geometry = design.lock.design_metadata.get("support_geometry")
        if not isinstance(geometry, Mapping) or not geometry.get("valid", False):
            reason = "support geometry is absent or invalid"
            if isinstance(geometry, Mapping) and isinstance(geometry.get("reason"), str):
                reason = str(geometry["reason"])
            return LipschitzAnchorResult(
                design.lock.lock_hash, None, "refused", anchor.outcome_lower, anchor.outcome_upper,
                np.asarray(anchor.support_geometry.gamma_grid), (), (reason,),
            )
        expected_ids = design.lock.estimation_row_ids
        if set(row_ids) != set(expected_ids) or len(row_ids) != len(expected_ids):
            raise ValueError("outcome row_ids must exactly match the locked estimation rows")
        values = np.asarray(outcomes, dtype=float)
        if values.shape != (len(row_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("outcomes must be a finite vector aligned with row_ids")
        if np.any(values < anchor.outcome_lower) or np.any(values > anchor.outcome_upper):
            return LipschitzAnchorResult(
                design.lock.lock_hash, str(geometry["digest"]), "refused", anchor.outcome_lower,
                anchor.outcome_upper, np.asarray(anchor.support_geometry.gamma_grid), (),
                ("held-out outcomes fall outside the declared bounded-outcome range",),
            )
        pairs = tuple(edge.groups for edge in design.graph.edges if edge.supported)
        if not pairs:
            return LipschitzAnchorResult(
                design.lock.lock_hash, str(geometry["digest"]), "refused", anchor.outcome_lower,
                anchor.outcome_upper, np.asarray(anchor.support_geometry.gamma_grid), (),
                ("no graph-supported pairwise contrasts are available",),
            )
        positions = [design.data.row_ids.index(row_id) for row_id in expected_ids]
        x = design.data.covariates[positions]
        groups = [design.data.groups[position] for position in positions]
        outcome_by_id = dict(zip(row_ids, values, strict=True))
        frame = pd.DataFrame(x, columns=design.declaration.covariates)
        frame[design.declaration.group] = groups
        frame["__scova_outcome__"] = [outcome_by_id[row_id] for row_id in expected_ids]
        labels = _canonical_labels(design.data.groups)
        contrasts = tuple(
            ContrastSpec(f"{pair[0]} - {pair[1]}", ((pair[0], 1.0), (pair[1], -1.0)))
            for pair in pairs
        )
        base = SCOVADeclaration(
            "__scova_outcome__", design.declaration.group, design.declaration.covariates,
            interpretation=design.declaration.interpretation, n_splits=design.declaration.n_splits,
            random_state=design.declaration.random_state, contrasts=contrasts,
        )
        fitted = SCOVA(
            propensity_model=self.propensity_model, outcome_model=self.outcome_model
        ).fit(frame, base)
        observed_codes = np.array([labels.index(group) for group in groups], dtype=int)
        reference_ids = geometry["reference_row_ids"]
        reference_positions = {
            code: np.array(
                [design.data.row_ids.index(row_id) for row_id in reference_ids[str(label)]]
            )
            for code, label in enumerate(labels)
        }
        references = {
            code: design.data.covariates[reference_positions]
            for code, reference_positions in reference_positions.items()
        }
        distances: dict[int, np.ndarray] = {}
        neighbor_indices: dict[int, np.ndarray] = {}
        neighbor_weights: dict[int, np.ndarray] = {}
        transport_tilts: dict[int, np.ndarray] = {}
        for code in range(len(labels)):
            distance, indices, weights = soft_k_nearest(x, references[code], dict(geometry))
            distances[code] = distance
            neighbor_indices[code] = indices
            neighbor_weights[code] = weights
            transport_tilts[code] = gaussian_reference_transport(
                x, references[code], dict(geometry)
            )
        reference_prediction = _fold_reference_predictions(
            self, x, values, observed_codes, fitted.fold_assignments, references,
            neighbor_indices, neighbor_weights,
        )
        gamma_grid = np.asarray(anchor.support_geometry.gamma_grid)
        transport_residuals = np.zeros((len(x), len(labels)), dtype=float)
        transport_ess: dict[str, float] = {}
        for code, label in enumerate(labels):
            observed = observed_codes == code
            if not np.any(observed):
                return LipschitzAnchorResult(
                    design.lock.lock_hash, str(geometry["digest"]), "refused", anchor.outcome_lower,
                    anchor.outcome_upper, gamma_grid, (),
                    (f"transport group {label!r} has no held-out observations",),
                )
            normalized_tilt = transport_tilts[code] / float(
                np.mean(transport_tilts[code][observed])
            )
            group_weights = normalized_tilt[observed]
            ess = float(np.square(group_weights.sum()) / np.square(group_weights).sum())
            transport_ess[str(label)] = ess
            if not np.isfinite(ess) or ess <= self.thresholds.min_group_ess_refuse:
                return LipschitzAnchorResult(
                    design.lock.lock_hash, str(geometry["digest"]), "refused", anchor.outcome_lower,
                    anchor.outcome_upper, gamma_grid, (),
                    (f"transport effective sample size failed for group {label!r}",),
                    confidence_level=design.declaration.confidence_level,
                    inference_method="regularized-transport-eif",
                    transport_diagnostics={"effective_sample_size": transport_ess},
                )
            transport_residuals[:, code] = (
                observed * normalized_tilt * (values - fitted.outcome_predictions[:, code])
                / fitted.propensity_predictions[:, code]
            )
        if not np.all(np.isfinite(transport_residuals)):
            return LipschitzAnchorResult(
                design.lock.lock_hash, str(geometry["digest"]), "refused", anchor.outcome_lower,
                anchor.outcome_upper, gamma_grid, (), ("transport residual score is non-finite",),
                confidence_level=design.declaration.confidence_level,
                inference_method="regularized-transport-eif",
                transport_diagnostics={"effective_sample_size": transport_ess},
            )
        results = []
        for pair in pairs:
            first, second = labels.index(pair[0]), labels.index(pair[1])
            bounded = bounded_pairwise_anchor(
                groups=pair,
                group_codes=observed_codes,
                outcomes=values,
                propensity=fitted.propensity_predictions,
                outcome_predictions=fitted.outcome_predictions,
                active_codes=(first, second),
                outcome_lower=anchor.outcome_lower,
                outcome_upper=anchor.outcome_upper,
                confidence_level=design.declaration.confidence_level,
            )
            results.append(
                lipschitz_pairwise_anchor(
                    bounded=bounded,
                    propensity=fitted.propensity_predictions,
                    active_codes=(first, second),
                    gamma_grid=gamma_grid,
                    smooth_distances=np.column_stack((distances[first], distances[second])),
                    reference_predictions=np.column_stack(
                        (reference_prediction[:, first], reference_prediction[:, second])
                    ),
                    outcome_lower=anchor.outcome_lower,
                    outcome_upper=anchor.outcome_upper,
                    transport_residuals=np.column_stack(
                        (transport_residuals[:, first], transport_residuals[:, second])
                    ),
                    confidence_level=design.declaration.confidence_level,
                )
            )
        return LipschitzAnchorResult(
            design.lock.lock_hash, str(geometry["digest"]), "experimental", anchor.outcome_lower,
            anchor.outcome_upper, gamma_grid, tuple(results), (),
            confidence_level=design.declaration.confidence_level,
            inference_method="regularized-transport-eif-im",
            transport_diagnostics={
                "transport": "gaussian-mixture-design-temperature",
                "effective_sample_size": transport_ess,
            },
        )


def graph_to_dict(graph: ComparabilityGraphResult) -> dict[str, Any]:
    return {
        "group_labels": list(graph.group_labels),
        "lambdas": list(graph.lambdas),
        "edges": [
            {
                "groups": list(edge.groups),
                "supported_lambdas": list(edge.supported_lambdas),
                "decisions": [decision.to_dict() for decision in edge.decisions],
                "refusal_reasons": list(edge.refusal_reasons),
            }
            for edge in graph.edges
        ],
        "hyperedges": [
            {
                "groups": list(edge.groups),
                "supported_lambdas": list(edge.supported_lambdas),
                "decisions": [decision.to_dict() for decision in edge.decisions],
                "refusal_reasons": list(edge.refusal_reasons),
            }
            for edge in graph.hyperedges
        ],
        "maximal_pairwise_cliques": [list(item) for item in graph.maximal_pairwise_cliques],
        "supported_maximal_hyperedges": [
            list(item.groups) for item in graph.supported_maximal_hyperedges
        ],
        "threshold_version": graph.threshold_version,
        "threshold_calibrated": graph.threshold_calibrated,
    }


def graph_from_dict(values: Mapping[str, Any]) -> ComparabilityGraphResult:
    def edge(item: Mapping[str, Any]) -> PairwiseEdge:
        return PairwiseEdge(
            tuple(item["groups"]),
            tuple(float(value) for value in item["supported_lambdas"]),
            tuple(GateDecision.from_dict(value) for value in item.get("decisions", [])),
            tuple(item.get("refusal_reasons", [])),
        )

    def hyperedge(item: Mapping[str, Any]) -> SubsetHyperedge:
        return SubsetHyperedge(
            tuple(item["groups"]),
            tuple(float(value) for value in item["supported_lambdas"]),
            tuple(GateDecision.from_dict(value) for value in item.get("decisions", [])),
            tuple(item.get("refusal_reasons", [])),
        )

    hyperedges = tuple(hyperedge(item) for item in values.get("hyperedges", []))
    supported = {item.groups: item for item in hyperedges if item.supported}
    maximal = tuple(
        supported[tuple(groups)]
        for groups in values.get("supported_maximal_hyperedges", [])
        if tuple(groups) in supported
    )
    return ComparabilityGraphResult(
        group_labels=tuple(values["group_labels"]),
        lambdas=tuple(float(value) for value in values["lambdas"]),
        edges=tuple(edge(item) for item in values["edges"]),
        hyperedges=hyperedges,
        maximal_pairwise_cliques=tuple(tuple(item) for item in values["maximal_pairwise_cliques"]),
        supported_maximal_hyperedges=maximal,
        threshold_version=str(values["threshold_version"]),
        threshold_calibrated=bool(values["threshold_calibrated"]),
    )
