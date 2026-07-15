"""Cross-fitted fixed-target AIPW estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error

from ._version import __version__
from .declaration import JsonLabel, SCOVADeclaration
from .diagnostics import compute_diagnostics
from .result import SCOVAResult, Verdict


@dataclass(frozen=True, slots=True)
class NuisancePredictions:
    """Externally supplied, observation-aligned oracle nuisance predictions."""

    propensity: np.ndarray
    outcome_regression: np.ndarray
    group_labels: tuple[JsonLabel, ...]


def _native_label(value: Any) -> JsonLabel:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    raise TypeError("Group labels must be strings, integers, floats, or booleans")


def _label_sort_key(value: JsonLabel) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, float(value))
    return (2, value)


def _validate_probabilities(probability: np.ndarray, n: int, k: int) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.shape != (n, k):
        raise ValueError(f"Propensity predictions must have shape {(n, k)}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Propensity predictions must be finite")
    if np.any(values <= 0) or np.any(values > 1):
        raise ValueError("Propensity predictions must be strictly positive and at most one")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("Each propensity prediction row must sum to one")
    return values


def _assemble_aipw(
    outcome: np.ndarray,
    group_codes: np.ndarray,
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-target means, influence rows, and estimator covariance."""
    n, n_groups = propensity.shape
    if outcome_regression.shape != (n, n_groups):
        raise ValueError(f"Outcome predictions must have shape {(n, n_groups)}")
    if not np.all(np.isfinite(outcome_regression)):
        raise ValueError("Outcome predictions must be finite")
    observed = np.eye(n_groups, dtype=float)[group_codes]
    signal = outcome_regression + observed / propensity * (outcome[:, None] - outcome_regression)
    means = signal.mean(axis=0)
    influence = signal - means
    covariance = np.cov(influence, rowvar=False, ddof=1) / n
    covariance = np.atleast_2d(covariance)
    covariance = (covariance + covariance.T) / 2
    return means, influence, covariance


class SCOVA:
    """Fixed-target, cross-fitted multi-group AIPW estimator."""

    def __init__(
        self,
        *,
        propensity_model: BaseEstimator | None = None,
        outcome_model: BaseEstimator | None = None,
        nuisance_strategy: Literal["adaptive", "linear", "custom"] = "adaptive",
    ) -> None:
        if nuisance_strategy not in {"adaptive", "linear", "custom"}:
            raise ValueError("nuisance_strategy must be 'adaptive', 'linear', or 'custom'")
        if (propensity_model is None) != (outcome_model is None):
            raise ValueError("propensity_model and outcome_model must be supplied together")
        if nuisance_strategy == "custom" and propensity_model is None:
            raise ValueError("custom nuisance_strategy requires both nuisance models")
        if nuisance_strategy != "custom" and propensity_model is not None:
            if nuisance_strategy != "adaptive":
                raise ValueError("explicit nuisance models require nuisance_strategy='custom'")
            nuisance_strategy = "custom"
        self.nuisance_strategy = nuisance_strategy
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model

    @staticmethod
    def _linear_propensity_model() -> BaseEstimator:
        return LogisticRegression(max_iter=2000)

    @staticmethod
    def _linear_outcome_model() -> BaseEstimator:
        return Ridge(alpha=1.0)

    @staticmethod
    def _adaptive_propensity_candidates() -> dict[str, BaseEstimator]:
        return {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
                learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
            ),
        }

    @staticmethod
    def _adaptive_outcome_candidates() -> dict[str, BaseEstimator]:
        return {
            "Ridge": Ridge(alpha=1.0),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
            ),
        }

    @staticmethod
    def _inner_folds(group_codes: np.ndarray, n_splits: int = 3) -> np.ndarray:
        """Deterministic stratified folds used only to select nuisance candidates."""
        counts = np.bincount(group_codes)
        usable_splits = min(n_splits, int(np.min(counts)))
        if usable_splits < 2:
            return np.zeros(len(group_codes), dtype=int)
        folds = np.empty(len(group_codes), dtype=int)
        for code in np.unique(group_codes):
            indices = np.flatnonzero(group_codes == code)
            folds[indices] = np.arange(len(indices)) % usable_splits
        return folds

    @classmethod
    def _select_propensity_model(
        cls, x: np.ndarray, group_codes: np.ndarray
    ) -> tuple[BaseEstimator, str, dict[str, float]]:
        """Choose a probability learner by deterministic inner-fold log loss."""
        candidates = cls._adaptive_propensity_candidates()
        folds = cls._inner_folds(group_codes)
        scores: dict[str, float] = {}
        for name, candidate in candidates.items():
            if len(np.unique(folds)) < 2:
                scores[name] = float("inf")
                continue
            predicted = np.empty((len(x), len(np.unique(group_codes))))
            for fold in np.unique(folds):
                train = folds != fold
                test = ~train
                model = clone(candidate)
                model.fit(x[train], group_codes[train])
                probability = np.asarray(model.predict_proba(x[test]), dtype=float)
                for column, code in enumerate(np.asarray(model.classes_, dtype=int)):
                    predicted[test, code] = probability[:, column]
            scores[name] = float(
                log_loss(group_codes, predicted, labels=np.arange(predicted.shape[1]))
            )
        selected_name = min(scores, key=scores.__getitem__)
        return clone(candidates[selected_name]), selected_name, scores

    @classmethod
    def _select_outcome_model(
        cls, x: np.ndarray, outcome: np.ndarray
    ) -> tuple[BaseEstimator, str, dict[str, float]]:
        """Choose an outcome learner by deterministic inner-fold squared error."""
        candidates = cls._adaptive_outcome_candidates()
        folds = np.arange(len(outcome)) % min(3, len(outcome))
        scores: dict[str, float] = {}
        for name, candidate in candidates.items():
            if len(np.unique(folds)) < 2:
                scores[name] = float("inf")
                continue
            predicted = np.empty(len(outcome))
            for fold in np.unique(folds):
                train = folds != fold
                test = ~train
                model = clone(candidate)
                model.fit(x[train], outcome[train])
                predicted[test] = np.asarray(model.predict(x[test]), dtype=float)
            scores[name] = float(mean_squared_error(outcome, predicted))
        selected_name = min(scores, key=scores.__getitem__)
        return clone(candidates[selected_name]), selected_name, scores

    @staticmethod
    def _validate_data(
        data: pd.DataFrame, declaration: SCOVADeclaration
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[JsonLabel, ...]]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        required = [declaration.outcome, declaration.group, *declaration.covariates]
        missing_columns = [column for column in required if column not in data.columns]
        if missing_columns:
            raise ValueError(f"Data is missing declared columns: {missing_columns}")
        selected = data.loc[:, required]
        if selected.isna().any().any():
            raise ValueError("Declared analysis columns cannot contain missing values")
        try:
            x = data.loc[:, declaration.covariates].to_numpy(dtype=float)
            outcome = data.loc[:, declaration.outcome].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Outcome and covariates must be numeric") from error
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(outcome)):
            raise ValueError("Outcome and covariates must be finite")
        raw_labels = [_native_label(value) for value in pd.unique(data[declaration.group])]
        labels = tuple(sorted(raw_labels, key=_label_sort_key))
        if len(labels) < 2:
            raise ValueError("SCOVA requires at least two observed groups")
        label_to_code = {label: code for code, label in enumerate(labels)}
        group_codes = np.array(
            [label_to_code[_native_label(value)] for value in data[declaration.group]], dtype=int
        )
        counts = np.bincount(group_codes, minlength=len(labels))
        if np.any(counts < declaration.n_splits):
            too_small = {
                str(labels[code]): int(count)
                for code, count in enumerate(counts)
                if count < declaration.n_splits
            }
            raise ValueError(
                f"Every group needs at least n_splits observations; too small: {too_small}"
            )
        return x, outcome, group_codes, labels

    @staticmethod
    def _design_folds(
        data: pd.DataFrame,
        declaration: SCOVADeclaration,
        group_codes: np.ndarray,
    ) -> np.ndarray:
        design = data.loc[:, [declaration.group, *declaration.covariates]]
        hashes = pd.util.hash_pandas_object(design, index=False, categorize=True).to_numpy(
            dtype=np.uint64
        )
        salt = np.uint64(declaration.random_state % (2**32)) * np.uint64(0x9E3779B1)
        hashes = hashes ^ salt
        folds = np.empty(len(data), dtype=int)
        for code in np.unique(group_codes):
            indices = np.flatnonzero(group_codes == code)
            order = indices[np.argsort(hashes[indices], kind="stable")]
            folds[order] = np.arange(len(order)) % declaration.n_splits
        return folds

    def _cross_fit(
        self,
        x: np.ndarray,
        group_codes: np.ndarray,
        outcome: np.ndarray,
        folds: np.ndarray,
        n_groups: int,
        labels: tuple[JsonLabel, ...],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        propensity = np.empty((len(outcome), n_groups), dtype=float)
        outcome_regression = np.empty((len(outcome), n_groups), dtype=float)
        propensity_selected: list[dict[str, Any]] = []
        outcome_selected: dict[str, list[dict[str, Any]]] = {str(label): [] for label in labels}
        for fold in sorted(np.unique(folds)):
            test = folds == fold
            train = ~test
            if self.nuisance_strategy == "adaptive":
                (
                    propensity_model,
                    propensity_name,
                    propensity_scores,
                ) = self._select_propensity_model(x[train], group_codes[train])
                propensity_selected.append(
                    {"fold": int(fold), "selected": propensity_name, "scores": propensity_scores}
                )
            elif self.nuisance_strategy == "linear":
                propensity_model = self._linear_propensity_model()
            else:
                assert self.propensity_model is not None
                propensity_model = clone(self.propensity_model)
            propensity_model.fit(x[train], group_codes[train])
            raw_probability = np.asarray(propensity_model.predict_proba(x[test]), dtype=float)
            classes = np.asarray(propensity_model.classes_, dtype=int)
            if set(classes.tolist()) != set(range(n_groups)):
                raise ValueError("Every propensity training fold must contain every group")
            aligned = np.empty((test.sum(), n_groups), dtype=float)
            for column, group_code in enumerate(classes):
                aligned[:, group_code] = raw_probability[:, column]
            propensity[test] = aligned
            for code in range(n_groups):
                group_train = train & (group_codes == code)
                if self.nuisance_strategy == "adaptive":
                    model, outcome_name, outcome_scores = self._select_outcome_model(
                        x[group_train], outcome[group_train]
                    )
                    outcome_selected[str(labels[code])].append(
                        {"fold": int(fold), "selected": outcome_name, "scores": outcome_scores}
                    )
                elif self.nuisance_strategy == "linear":
                    model = self._linear_outcome_model()
                else:
                    assert self.outcome_model is not None
                    model = clone(self.outcome_model)
                model.fit(x[group_train], outcome[group_train])
                outcome_regression[test, code] = np.asarray(model.predict(x[test]), dtype=float)
        metadata: dict[str, Any] = {
            "source": "cross-fitted",
            "nuisance_strategy": self.nuisance_strategy,
            "propensity_model": self._metadata_model_name("propensity"),
            "outcome_model": self._metadata_model_name("outcome"),
        }
        if self.nuisance_strategy == "adaptive":
            metadata["selection"] = {
                "criterion": {"propensity": "log_loss", "outcome": "mean_squared_error"},
                "inner_folds": 3,
                "propensity": propensity_selected,
                "outcome": outcome_selected,
            }
        return propensity, outcome_regression, metadata

    def _metadata_model_name(self, nuisance: Literal["propensity", "outcome"]) -> str:
        if self.nuisance_strategy == "adaptive":
            return "adaptive"
        if self.nuisance_strategy == "linear":
            return "LogisticRegression" if nuisance == "propensity" else "Ridge"
        model = self.propensity_model if nuisance == "propensity" else self.outcome_model
        assert model is not None
        return type(model).__name__

    def fit(
        self,
        data: pd.DataFrame,
        declaration: SCOVADeclaration,
        *,
        nuisance_predictions: NuisancePredictions | None = None,
    ) -> SCOVAResult:
        """Fit SCOVA or assemble its estimator from supplied oracle nuisances."""
        x, outcome, group_codes, labels = self._validate_data(data, declaration)
        n, n_groups = len(data), len(labels)
        folds = self._design_folds(data, declaration, group_codes)
        if nuisance_predictions is None:
            propensity, outcome_regression, nuisance_metadata = self._cross_fit(
                x, group_codes, outcome, folds, n_groups, labels
            )
        else:
            supplied_labels = tuple(
                _native_label(label) for label in nuisance_predictions.group_labels
            )
            if supplied_labels != labels:
                raise ValueError(
                    "Nuisance prediction group_labels must exactly match SCOVA's canonical order "
                    f"{labels}"
                )
            propensity = np.asarray(nuisance_predictions.propensity, dtype=float)
            outcome_regression = np.asarray(nuisance_predictions.outcome_regression, dtype=float)
            nuisance_metadata = {
                "source": "supplied",
                "nuisance_strategy": "supplied",
                "propensity_model": None,
                "outcome_model": None,
            }
        propensity = _validate_probabilities(propensity, n, n_groups)
        means, influence, covariance = _assemble_aipw(
            outcome, group_codes, propensity, outcome_regression
        )
        diagnostics = compute_diagnostics(
            x,
            group_codes,
            propensity,
            influence,
            folds,
            declaration.covariates,
            labels,
        )
        verdict = (
            Verdict.EXPLORATORY_ONLY
            if declaration.interpretation == "causal"
            else Verdict.DESCRIPTIVE_ONLY
        )
        result = SCOVAResult(
            group_labels=labels,
            covariate_names=declaration.covariates,
            group_means=means,
            influence_values=influence,
            covariance=covariance,
            fold_assignments=folds,
            propensity_predictions=propensity,
            outcome_predictions=outcome_regression,
            diagnostics=diagnostics,
            declaration_hash=declaration.declaration_hash,
            nuisance_metadata=nuisance_metadata,
            interpretation=declaration.interpretation,
            random_state=declaration.random_state,
            verdict=verdict,
            package_version=__version__,
        )
        for left in range(n_groups):
            for right in range(left + 1, n_groups):
                weights = np.zeros(n_groups)
                weights[left] = 1.0
                weights[right] = -1.0
                result.contrast(weights, name=f"{labels[left]} - {labels[right]}")
        for specification in declaration.contrasts:
            result.contrast(dict(specification.weights), name=specification.name)
        return result
