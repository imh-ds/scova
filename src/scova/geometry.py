"""Frozen, outcome-blind support geometry for experimental Stage 5B."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from .declaration import JsonLabel, SupportGeometryDeclaration


def fit_support_geometry(
    x: np.ndarray,
    groups: Sequence[JsonLabel],
    row_ids: Sequence[JsonLabel],
    design_mask: np.ndarray,
    declaration: SupportGeometryDeclaration,
) -> dict[str, Any]:
    """Fit and serialize geometry from design rows alone, without outcomes."""
    design_x = np.asarray(x, dtype=float)[design_mask]
    design_groups = np.asarray(groups, dtype=object)[design_mask]
    design_ids = tuple(
        row_id for row_id, selected in zip(row_ids, design_mask, strict=True) if selected
    )
    labels = tuple(sorted(set(groups), key=lambda value: (str(type(value)), str(value))))
    references = {
        str(label): [
            row_id
            for row_id, group in zip(design_ids, design_groups, strict=True)
            if group == label
        ]
        for label in labels
    }
    counts = {key: len(value) for key, value in references.items()}
    invalid = [key for key, count in counts.items() if count < declaration.neighbor_count]
    if invalid:
        return {
            "schema_version": 1,
            "valid": False,
            "reason": "insufficient design references for groups: " + ", ".join(invalid),
            "configuration": declaration.to_dict(),
            "reference_row_ids": references,
            "reference_counts": counts,
        }
    location = design_x.mean(axis=0)
    scale = design_x.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (design_x - location) / scale
    covariance = LedoitWolf().fit(standardized).covariance_
    ridge = max(float(np.trace(covariance)) / len(covariance) * 1e-8, 1e-10)
    precision = np.linalg.pinv(covariance + ridge * np.eye(len(covariance)))
    transformed = standardized @ np.linalg.cholesky(precision).T
    nearest: list[float] = []
    for label in labels:
        values = transformed[design_groups == label]
        distances = np.sqrt(np.maximum(np.square(values[:, None] - values[None, :]).sum(axis=2), 0))
        np.fill_diagonal(distances, np.inf)
        nearest.extend(np.min(distances, axis=1).tolist())
    temperature = float(np.median(np.asarray(nearest)))
    if not np.isfinite(temperature) or temperature <= 1e-10:
        return {
            "schema_version": 1,
            "valid": False,
            "reason": "design median soft-min temperature is non-positive",
            "configuration": declaration.to_dict(),
            "reference_row_ids": references,
            "reference_counts": counts,
        }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "valid": True,
        "configuration": declaration.to_dict(),
        "reference_row_ids": references,
        "reference_counts": counts,
        "location": location.tolist(),
        "scale": scale.tolist(),
        "precision": precision.tolist(),
        "temperature": temperature,
    }
    metadata["digest"] = sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return metadata


def soft_k_nearest(
    query: np.ndarray, references: np.ndarray, geometry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return design-locked smooth distances and neighbor weights."""
    location = np.asarray(geometry["location"], dtype=float)
    scale = np.asarray(geometry["scale"], dtype=float)
    precision = np.asarray(geometry["precision"], dtype=float)
    count = int(geometry["configuration"]["neighbor_count"])
    temperature = float(geometry["temperature"])
    if len(references) < count or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("support geometry is not usable for requested references")
    q = (np.asarray(query, dtype=float) - location) / scale
    r = (np.asarray(references, dtype=float) - location) / scale
    delta = q[:, None, :] - r[None, :, :]
    squared = np.einsum("nri,ij,nrj->nr", delta, precision, delta)
    distances = np.sqrt(np.maximum(squared, 0))
    indices = np.argpartition(distances, count - 1, axis=1)[:, :count]
    selected = np.take_along_axis(distances, indices, axis=1)
    minimum = selected.min(axis=1, keepdims=True)
    logits = -(selected - minimum) / temperature
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    smooth = minimum[:, 0] - temperature * np.log(np.exp(logits).mean(axis=1))
    return smooth, indices, weights
