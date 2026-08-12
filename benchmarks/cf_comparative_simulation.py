"""Frozen data-generating process for the two-group comparative methods study."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SPEC_DIRECTORY = Path(__file__).with_name("specs")
_PROTOCOL_FILES = {
    "v2": "cf_two_group_comparative_methods_v2.json",
    "v3": "cf_two_group_comparative_methods_v3.json",
}


@dataclass(frozen=True, slots=True)
class ComparativeData:
    """Observed data paired with simulation truth for one replication."""

    data: pd.DataFrame
    group: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    propensity: np.ndarray
    ate: float
    att: float
    source_metadata: Mapping[str, Any]


def _protocol(protocol_version: str = "v2") -> dict[str, Any]:
    try:
        path = _SPEC_DIRECTORY / _PROTOCOL_FILES[protocol_version]
    except KeyError as error:
        raise ValueError(f"unknown comparative protocol version {protocol_version!r}") from error
    return json.loads(path.read_text(encoding="utf-8"))


def comparative_cells(protocol_version: str = "v2") -> tuple[dict[str, object], ...]:
    """Return the eight declared two-group DGP cells in stable order."""
    protocol = _protocol(protocol_version)
    factors = protocol["factors"]
    cell_prefix = "cmp-v1" if protocol_version == "v2" else f"cmp-{protocol_version}"
    return tuple(
        {
            "cell_id": f"{cell_prefix}-{outcome}-{confounding}-{overlap}",
            "protocol_version": protocol_version,
            "n_groups": protocol["n_groups"],
            "n_covariates": protocol["n_covariates"],
            "n": protocol["units_per_replication"],
            "outcome_surface": outcome,
            "confounding_surface": confounding,
            "overlap": overlap,
        }
        for outcome in factors["outcome_surface"]
        for confounding in factors["confounding_surface"]
        for overlap in factors["overlap"]
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std()


def _propensity(x: np.ndarray, cell: Mapping[str, object]) -> np.ndarray:
    if cell["confounding_surface"] == "linear":
        score = 0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.4 * x[:, 2]
    elif cell["confounding_surface"] == "nonlinear":
        score = np.sin(1.4 * x[:, 0]) + 0.7 * x[:, 1] ** 2 - 0.5 * np.abs(x[:, 2])
    elif cell["confounding_surface"] == "smooth-nonlinear":
        score = (
            np.sin(1.4 * x[:, 0])
            + 0.7 * x[:, 1] ** 2
            - 0.5 * np.abs(x[:, 2])
            + 0.35 * x[:, 3] * x[:, 4]
        )
    elif cell["confounding_surface"] == "threshold":
        score = (
            1.1 * (x[:, 0] > 0)
            - 0.9 * (x[:, 1] > 0.5)
            + 0.7 * (x[:, 2] * x[:, 3] > 0)
            - 0.4 * (x[:, 4] < -0.5)
        )
    else:
        raise ValueError("unknown confounding surface")
    protocol = _protocol(str(cell.get("protocol_version", "v2")))
    scale = float(protocol.get("dgp_contract", {}).get("overlap_scaling", {}).get(
        str(cell["overlap"]), 0.7 if cell["overlap"] == "adequate" else 2.0
    ))
    logits = scale * _standardize(score)
    return 1.0 / (1.0 + np.exp(-logits))


def _potential_outcomes(x: np.ndarray, cell: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    baseline = 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2] + 0.2 * x[:, 3]
    if cell["outcome_surface"] == "linear":
        treatment_effect = 1.0 + 0.25 * x[:, 0]
    elif cell["outcome_surface"] == "interaction":
        baseline += 0.6 * x[:, 0] * x[:, 1] - 0.35 * x[:, 2] ** 2
        treatment_effect = 1.0 + 0.35 * x[:, 0] * x[:, 1] + 0.2 * np.sin(x[:, 3])
    elif cell["outcome_surface"] == "smooth-nonlinear":
        baseline = (
            0.7 * np.sin(1.2 * x[:, 0])
            - 0.45 * x[:, 1] ** 2
            + 0.35 * x[:, 2] * x[:, 3]
            + 0.2 * x[:, 4]
        )
        treatment_effect = 1.0 + 0.3 * np.sin(x[:, 0] * x[:, 1]) + 0.2 * x[:, 2] ** 2
    elif cell["outcome_surface"] == "threshold":
        baseline = (
            0.8 * (x[:, 0] > 0)
            - 0.6 * (x[:, 1] + x[:, 2] > 0)
            + 0.4 * (x[:, 3] * x[:, 4] > 0)
        )
        treatment_effect = (
            1.0 + 0.4 * ((x[:, 0] > 0) & (x[:, 1] > 0)) - 0.25 * (x[:, 2] < -0.5)
        )
    else:
        raise ValueError("unknown outcome surface")
    return baseline, baseline + treatment_effect


def simulate_comparative_cell(cell: Mapping[str, object], seed: int) -> ComparativeData:
    """Simulate observed outcomes and retain the complete potential-outcome truth."""
    protocol_version = str(cell.get("protocol_version", "v2"))
    if cell not in comparative_cells(protocol_version):
        raise ValueError("cell is not part of the frozen comparative design")
    rng = np.random.default_rng(seed)
    n = int(cell["n"])
    x = rng.normal(size=(n, int(cell["n_covariates"])))
    propensity = _propensity(x, cell)
    group = rng.binomial(1, propensity, size=n)
    mu0, mu1 = _potential_outcomes(x, cell)
    outcome = np.where(group == 1, mu1, mu0) + rng.normal(scale=1.0, size=n)
    data = pd.DataFrame(x, columns=[f"x{index}" for index in range(1, x.shape[1] + 1)])
    data["group"] = group
    data["outcome"] = outcome
    effects = mu1 - mu0
    return ComparativeData(
        data=data,
        group=group,
        mu0=mu0,
        mu1=mu1,
        propensity=propensity,
        ate=float(np.mean(effects)),
        att=float(np.mean(effects[group == 1])),
        source_metadata={
            "protocol_id": _protocol(protocol_version)["protocol_id"],
            "cell_id": cell["cell_id"],
            "seed": seed,
            "outcome_formula": str(cell["outcome_surface"]),
            "assignment_formula": str(cell["confounding_surface"]),
        },
    )
