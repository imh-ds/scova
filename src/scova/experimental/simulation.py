"""Production stabilization DGPs and matched target truths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..estimator import NuisancePredictions
from .tilts import geometric_tilt_and_gradient

OverlapRegime = Literal["strong", "moderate", "weak", "near_violation", "pairwise_only"]
OutcomeSurface = Literal["linear", "nonlinear", "threshold", "nonmonotone"]
ImbalanceRegime = Literal["balanced", "rare"]
ErrorRegime = Literal["normal", "heteroskedastic", "heavy_tailed"]
NuisanceRegime = Literal[
    "oracle",
    "correct_both",
    "gps_correct_outcome_wrong",
    "outcome_correct_gps_wrong",
    "both_wrong",
    "flexible",
    "deliberately_inadequate",
]


@dataclass(frozen=True, slots=True)
class StabilizationSpec:
    n_groups: int
    n: int
    p: int
    overlap: OverlapRegime
    outcome: OutcomeSurface
    imbalance: ImbalanceRegime
    error: ErrorRegime = "normal"
    nuisance: NuisanceRegime = "correct_both"


@dataclass(frozen=True, slots=True)
class StabilizationData:
    data: pd.DataFrame
    propensity: np.ndarray
    outcome_regression: np.ndarray
    pseudo_propensity: np.ndarray
    group_labels: tuple[str, ...]

    def target_path(
        self,
        lambdas: np.ndarray,
        weights: np.ndarray,
        *,
        pseudo: bool = False,
    ) -> np.ndarray:
        probability = self.pseudo_propensity if pseudo else self.propensity
        tilt, _ = geometric_tilt_and_gradient(
            probability, lambdas, tuple(range(len(self.group_labels)))
        )
        means = (
            np.einsum("nl,nk->lk", tilt, self.outcome_regression)
            / tilt.sum(axis=0)[:, None]
        )
        return means @ weights

    def nuisance_predictions(self, regime: NuisanceRegime) -> NuisancePredictions | None:
        if regime in ("correct_both", "flexible", "deliberately_inadequate"):
            return None
        zero_outcome = np.zeros_like(self.outcome_regression)
        if regime in ("oracle", "correct_both"):
            propensity, outcome = self.propensity, self.outcome_regression
        elif regime == "gps_correct_outcome_wrong":
            propensity, outcome = self.propensity, zero_outcome
        elif regime == "outcome_correct_gps_wrong":
            propensity, outcome = self.pseudo_propensity, self.outcome_regression
        elif regime == "both_wrong":
            propensity, outcome = self.pseudo_propensity, zero_outcome
        else:
            raise ValueError(f"unknown nuisance regime: {regime}")
        return NuisancePredictions(propensity, outcome, self.group_labels)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def generate_stabilization_data(spec: StabilizationSpec, *, seed: int) -> StabilizationData:
    if spec.n_groups not in (2, 4, 8):
        raise ValueError("stabilization DGP supports K in {2, 4, 8}")
    if spec.n < 100 or spec.p < 2:
        raise ValueError("stabilization DGP requires n >= 100 and p >= 2")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(spec.n, spec.p))
    labels = tuple(f"g{code}" for code in range(spec.n_groups))
    slopes = np.linspace(-1, 1, spec.n_groups)
    overlap_scale = {
        "strong": 0.25,
        "moderate": 0.8,
        "weak": 1.8,
        "near_violation": 3.5,
        "pairwise_only": 2.5,
    }[spec.overlap]
    logits = overlap_scale * (
        x[:, [0]] * slopes[None, :]
        + 0.4 * x[:, [1]] * np.cos(np.arange(spec.n_groups))[None, :]
    )
    if spec.overlap == "pairwise_only" and spec.n_groups >= 4:
        left = np.arange(spec.n_groups) < spec.n_groups // 2
        logits[:, left] += 3 * (x[:, [0]] < 0)
        logits[:, ~left] += 3 * (x[:, [0]] >= 0)
    if spec.imbalance == "rare":
        logits[:, -1] -= 2.5
    propensity = _softmax(logits)
    pseudo_logits = 0.15 * x[:, [0]] * slopes[None, :]
    pseudo_propensity = _softmax(pseudo_logits)

    effects = np.linspace(-0.75, 0.75, spec.n_groups)
    outcome_regression = np.empty((spec.n, spec.n_groups))
    for code, effect in enumerate(effects):
        if spec.outcome == "linear":
            base = 0.7 * x[:, 0] - 0.4 * x[:, 1]
        elif spec.outcome == "nonlinear":
            base = np.sin(x[:, 0]) + 0.4 * np.square(x[:, 1])
        elif spec.outcome == "threshold":
            base = (x[:, 0] > 0).astype(float) - 0.5 * (x[:, 1] > 0.5)
        elif spec.outcome == "nonmonotone":
            base = np.sin(1.5 * x[:, 0]) + 0.25 * np.square(x[:, 1])
            effect = effect * (1 - 1.2 * np.tanh(x[:, 0]))
        else:
            raise ValueError(f"unknown outcome surface: {spec.outcome}")
        interactions = 0.25 * code * x[:, 0]
        if spec.p > 2:
            interactions += 0.1 * x[:, 2]
        outcome_regression[:, code] = base + effect + interactions

    uniform = rng.random(spec.n)
    group_codes = (uniform[:, None] > np.cumsum(propensity, axis=1)).sum(axis=1)
    if spec.error == "normal":
        error = rng.normal(size=spec.n)
    elif spec.error == "heteroskedastic":
        error = rng.normal(scale=0.5 + np.abs(x[:, 0]), size=spec.n)
    elif spec.error == "heavy_tailed":
        error = rng.standard_t(df=3, size=spec.n) / np.sqrt(3)
    else:
        raise ValueError(f"unknown error regime: {spec.error}")
    outcome = outcome_regression[np.arange(spec.n), group_codes] + error
    columns = tuple(f"x{index + 1}" for index in range(spec.p))
    data = pd.DataFrame(x, columns=columns)
    data["group"] = [labels[code] for code in group_codes]
    data["outcome"] = outcome
    return StabilizationData(
        data=data,
        propensity=propensity,
        outcome_regression=outcome_regression,
        pseudo_propensity=pseudo_propensity,
        group_labels=labels,
    )
