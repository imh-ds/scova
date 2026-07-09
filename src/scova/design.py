"""Outcome-free data and locking primitives for Stage 4 design selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .declaration import DesignDeclaration, JsonLabel

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
