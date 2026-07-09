"""Production-candidate nuisance profiles and repeated-fit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from ..estimator import SCOVA
from .path import PathDeclaration, fit_path

LearnerProfileName = Literal["linear", "nonlinear", "ensemble", "deliberately_inadequate"]
CalibrationMethod = Literal["none", "sigmoid", "isotonic"]


class ConvexStackingClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Inner-CV probability stack with nonnegative weights summing to one."""

    def __init__(self, estimators: tuple[object, ...], cv: int = 3, random_state: int = 0):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConvexStackingClassifier:
        self.classes_ = np.unique(y)
        splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        predictions = np.empty((len(y), len(self.estimators), len(self.classes_)))
        class_index = {label: index for index, label in enumerate(self.classes_)}
        for train, test in splitter.split(x, y):
            for learner_index, estimator in enumerate(self.estimators):
                fitted = clone(estimator).fit(x[train], y[train])
                fold_probability = fitted.predict_proba(x[test])
                predictions[test, learner_index] = 0.0
                for column, label in enumerate(fitted.classes_):
                    predictions[test, learner_index, class_index[label]] = fold_probability[
                        :, column
                    ]
        targets = np.searchsorted(self.classes_, y)

        def objective(weights: np.ndarray) -> float:
            probability = np.einsum("m,nmk->nk", weights, predictions)
            chosen = np.clip(probability[np.arange(len(y)), targets], 1e-15, 1.0)
            return float(-np.mean(np.log(chosen)))

        initial = np.repeat(1 / len(self.estimators), len(self.estimators))
        optimization = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(initial),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        )
        if not optimization.success or not np.all(np.isfinite(optimization.x)):
            raise RuntimeError(f"convex classifier stacking failed: {optimization.message}")
        self.weights_ = optimization.x / optimization.x.sum()
        self.estimators_ = tuple(clone(estimator).fit(x, y) for estimator in self.estimators)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities = []
        class_index = {label: index for index, label in enumerate(self.classes_)}
        for estimator in self.estimators_:
            raw = estimator.predict_proba(x)
            aligned = np.zeros((len(x), len(self.classes_)))
            for column, label in enumerate(estimator.classes_):
                aligned[:, class_index[label]] = raw[:, column]
            probabilities.append(aligned)
        return np.einsum("m,mnk->nk", self.weights_, np.stack(probabilities))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


class ConvexStackingRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Inner-CV regression stack with nonnegative weights summing to one."""

    def __init__(self, estimators: tuple[object, ...], cv: int = 3, random_state: int = 0):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConvexStackingRegressor:
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        predictions = np.empty((len(y), len(self.estimators)))
        for train, test in splitter.split(x):
            for learner_index, estimator in enumerate(self.estimators):
                fitted = clone(estimator).fit(x[train], y[train])
                predictions[test, learner_index] = fitted.predict(x[test])

        def objective(weights: np.ndarray) -> float:
            residual = y - predictions @ weights
            return float(np.mean(np.square(residual)))

        initial = np.repeat(1 / len(self.estimators), len(self.estimators))
        optimization = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(initial),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        )
        if not optimization.success or not np.all(np.isfinite(optimization.x)):
            raise RuntimeError(f"convex regressor stacking failed: {optimization.message}")
        self.weights_ = optimization.x / optimization.x.sum()
        self.estimators_ = tuple(clone(estimator).fit(x, y) for estimator in self.estimators)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.column_stack([estimator.predict(x) for estimator in self.estimators_])
        return predictions @ self.weights_


@dataclass(frozen=True, slots=True)
class LearnerProfile:
    name: LearnerProfileName
    propensity_model: object
    outcome_model: object
    calibration: CalibrationMethod

    def estimator(self) -> SCOVA:
        return SCOVA(
            propensity_model=self.propensity_model,
            outcome_model=self.outcome_model,
        )


@dataclass(frozen=True, slots=True)
class CrossFitStability:
    seeds: tuple[int, ...]
    maximum_standardized_range: float
    estimate_paths: np.ndarray


def make_learner_profile(
    name: LearnerProfileName,
    *,
    calibration: CalibrationMethod = "none",
    random_state: int = 0,
) -> LearnerProfile:
    if name == "linear":
        propensity: object = LogisticRegression(max_iter=2000, random_state=random_state)
        outcome: object = Ridge(alpha=1.0)
    elif name == "nonlinear":
        propensity = HistGradientBoostingClassifier(random_state=random_state)
        outcome = HistGradientBoostingRegressor(random_state=random_state)
    elif name == "ensemble":
        propensity = ConvexStackingClassifier(
            estimators=(
                LogisticRegression(max_iter=2000, random_state=random_state),
                HistGradientBoostingClassifier(random_state=random_state),
            ),
            cv=3,
            random_state=random_state,
        )
        outcome = ConvexStackingRegressor(
            estimators=(
                Ridge(alpha=1.0),
                HistGradientBoostingRegressor(random_state=random_state),
            ),
            cv=3,
            random_state=random_state,
        )
    elif name == "deliberately_inadequate":
        propensity = LogisticRegression(
            max_iter=2000,
            C=1e-6,
            random_state=random_state,
        )
        outcome = Ridge(alpha=1e6)
    else:
        raise ValueError(f"unknown learner profile: {name}")
    if calibration not in ("none", "sigmoid", "isotonic"):
        raise ValueError(f"unknown calibration method: {calibration}")
    if calibration != "none":
        propensity = CalibratedClassifierCV(
            estimator=propensity,
            method=calibration,
            cv=3,
        )
    return LearnerProfile(name, propensity, outcome, calibration)


def assess_crossfit_stability(
    data: pd.DataFrame,
    declaration: PathDeclaration,
    profile: LearnerProfile,
    *,
    seeds: tuple[int, ...] = (101, 211, 307),
) -> CrossFitStability:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("stability assessment requires at least two distinct seeds")
    paths: list[np.ndarray] = []
    errors: np.ndarray | None = None
    for seed in seeds:
        base = replace(declaration.base, random_state=seed)
        candidate = replace(declaration, base=base, random_state=seed)
        result = fit_path(data, candidate, estimator=profile.estimator())
        paths.append(np.concatenate([contrast.estimates for contrast in result.contrasts.values()]))
        if errors is None:
            errors = np.concatenate(
                [contrast.standard_errors for contrast in result.contrasts.values()]
            )
    estimate_paths = np.stack(paths)
    assert errors is not None
    safe_errors = np.where(errors > 0, errors, np.nan)
    standardized_range = np.ptp(estimate_paths, axis=0) / safe_errors
    return CrossFitStability(
        seeds=seeds,
        maximum_standardized_range=float(np.nanmax(standardized_range)),
        estimate_paths=estimate_paths,
    )
