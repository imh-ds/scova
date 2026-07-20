"""Pinned validation-only adapters for DoubleMLAPOS and EconML DRLearner."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True, slots=True)
class ExternalAgreement:
    implementation: str
    version: str
    status: str
    estimates: tuple[float, ...] = ()
    standard_errors: tuple[float, ...] = ()
    influence: np.ndarray | None = None
    covariance: np.ndarray | None = None
    raw_standard_errors: tuple[float, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "version": self.version,
            "status": self.status,
            "estimates": list(self.estimates),
            "standard_errors": list(self.standard_errors),
            "raw_standard_errors": list(self.raw_standard_errors),
            "detail": self.detail,
        }


class KnownRandomizationClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible fixed multinomial assignment model for randomized fixtures."""

    def __init__(self, probabilities: tuple[float, ...]) -> None:
        self.probabilities = probabilities

    def fit(self, x: np.ndarray, y: np.ndarray) -> KnownRandomizationClassifier:
        self.classes_ = np.sort(np.unique(y))
        if tuple(int(value) for value in self.classes_) != tuple(range(len(self.probabilities))):
            raise ValueError("Known randomization levels do not match observed treatment levels")
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Known randomization probabilities must sum to one")
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(np.asarray(self.probabilities, dtype=float), (len(x), 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the modal declared arm for APIs that require classifier labels."""
        probability = self.predict_proba(x)
        return self.classes_[np.argmax(probability, axis=1)]


class SelectedOutcomeRegressor(RegressorMixin, BaseEstimator):
    """Validation-only replica of the declared per-arm outcome learner policy.

    DoubleML fits its outcome nuisance separately for each treatment level.  This
    wrapper makes that fit choose the same candidate family and inner-fold rule
    as SCOVA-CF's declared ``linear`` or ``adaptive`` outcome policy.
    """

    def __init__(self, learner_policy: str) -> None:
        self.learner_policy = learner_policy

    def fit(self, x: np.ndarray, y: np.ndarray) -> SelectedOutcomeRegressor:
        features = np.asarray(x, dtype=float)
        outcome = np.asarray(y, dtype=float)
        if self.learner_policy == "linear":
            self.selected_name_ = "Ridge"
            self.selection_scores_ = {"Ridge": 0.0}
            self.model_ = Ridge(alpha=1.0).fit(features, outcome)
            return self
        if self.learner_policy != "adaptive":
            raise ValueError(f"Unknown learner policy: {self.learner_policy}")
        candidates = {
            "Ridge": Ridge(alpha=1.0),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=0,
            ),
        }
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
                model = clone(candidate).fit(features[train], outcome[train])
                predicted[test] = np.asarray(model.predict(features[test]), dtype=float)
            scores[name] = float(mean_squared_error(outcome, predicted))
        self.selected_name_ = min(scores, key=scores.__getitem__)
        self.selection_scores_ = scores
        self.model_ = clone(candidates[self.selected_name_]).fit(features, outcome)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)


class TreatmentSpecificOutcomeRegressor(RegressorMixin, BaseEstimator):
    """Fit the declared outcome policy separately for each discrete treatment.

    EconML supplies covariates followed by one-hot non-reference treatment
    columns to ``model_regression``.  Splitting those rows restores the
    treatment-specific nuisance structure used by SCOVA-CF while EconML still
    constructs and aggregates its own doubly robust pseudo-outcomes.
    """

    def __init__(self, n_groups: int, learner_policy: str) -> None:
        self.n_groups = n_groups
        self.learner_policy = learner_policy

    def _codes(self, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(design, dtype=float)
        treatment_columns = self.n_groups - 1
        if values.ndim != 2 or values.shape[1] < treatment_columns:
            raise ValueError("EconML outcome design has incompatible treatment columns")
        features = values[:, : values.shape[1] - treatment_columns]
        encoded = values[:, values.shape[1] - treatment_columns :]
        codes = np.where(np.any(encoded != 0.0, axis=1), np.argmax(encoded, axis=1) + 1, 0)
        return features, codes.astype(int)

    def fit(self, x: np.ndarray, y: np.ndarray) -> TreatmentSpecificOutcomeRegressor:
        features, codes = self._codes(x)
        outcome = np.asarray(y, dtype=float)
        self.models_ = []
        for code in range(self.n_groups):
            mask = codes == code
            if not np.any(mask):
                raise ValueError(f"No observations for treatment level {code}")
            self.models_.append(
                SelectedOutcomeRegressor(self.learner_policy).fit(features[mask], outcome[mask])
            )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        features, codes = self._codes(x)
        predicted = np.empty(len(features), dtype=float)
        for code, model in enumerate(self.models_):
            mask = codes == code
            if np.any(mask):
                predicted[mask] = model.predict(features[mask])
        return predicted


def fixed_nuisance_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent literal implementation of unnormalized AIPW."""
    y = np.asarray(outcome, dtype=float)
    d = np.asarray(treatment, dtype=int)
    e = np.asarray(propensity, dtype=float)
    m = np.asarray(outcome_regression, dtype=float)
    n, k = e.shape
    if y.shape != (n,) or d.shape != (n,) or m.shape != (n, k):
        raise ValueError("Fixed-nuisance arrays have incompatible shapes")
    signal = np.empty((n, k), dtype=float)
    for group in range(k):
        observed = (d == group).astype(float)
        signal[:, group] = m[:, group] + observed * (y - m[:, group]) / e[:, group]
    means = signal.mean(axis=0)
    influence = signal - means
    covariance = (influence.T @ influence) / (n * (n - 1))
    return means, influence, covariance


def _splits(folds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(folds, dtype=int)
    return [
        (np.flatnonzero(values != fold), np.flatnonzero(values == fold))
        for fold in sorted(np.unique(values))
    ]


def _learners(policy: str) -> tuple[Any, Any]:
    if policy == "linear":
        return LogisticRegression(max_iter=2000), Ridge(alpha=1.0)
    if policy == "adaptive":
        return (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=0,
            ),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=0,
            ),
        )
    raise ValueError(f"Unknown learner policy: {policy}")


def doubleml_shared_score(
    x: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    folds: np.ndarray,
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
) -> ExternalAgreement:
    """Run DoubleMLAPOS with SCOVA folds and externally supplied nuisances."""
    try:
        import doubleml as dml
        from doubleml.utils import PSProcessorConfig
    except ImportError:
        return ExternalAgreement("DoubleMLAPOS", "not-installed", "blocked/missing-dependency")
    try:
        levels = tuple(int(value) for value in np.unique(treatment))
        data = dml.DoubleMLData.from_arrays(x, outcome, treatment)
        model = dml.DoubleMLAPOS(
            data,
            ml_g=Ridge(alpha=1.0),
            ml_m=LogisticRegression(max_iter=2000),
            treatment_levels=levels,
            n_folds=len(np.unique(folds)),
            normalize_ipw=False,
            ps_processor_config=PSProcessorConfig(
                clipping_threshold=1e-12,
                extreme_threshold=1e-12,
            ),
            draw_sample_splitting=False,
        )
        model.set_sample_splitting(_splits(folds))
        external = {
            level: {
                "ml_g_d_lvl0": outcome_regression[:, code, None],
                "ml_g_d_lvl1": outcome_regression[:, code, None],
                "ml_m": propensity[:, code, None],
            }
            for code, level in enumerate(levels)
        }
        model.fit(external_predictions=external)
        # DoubleML's scaled_psi is the negative of the centered AIPW row because
        # its linear-score derivative is -1. Recompute the SCOVA n-1 covariance
        # from those rows while retaining DoubleML's raw n-denominator SE below.
        influence = -np.asarray(model.framework.scaled_psi[:, :, 0], dtype=float)
        n = len(outcome)
        covariance = (influence.T @ influence) / (n * (n - 1))
        return ExternalAgreement(
            "DoubleMLAPOS",
            version("doubleml"),
            "complete",
            tuple(float(value) for value in np.ravel(model.coef)),
            tuple(float(value) for value in np.sqrt(np.diag(covariance))),
            influence=influence,
            covariance=covariance,
            raw_standard_errors=tuple(float(value) for value in np.ravel(model.se)),
            detail="Raw DoubleML SE uses n; aligned SE/covariance uses SCOVA's n-1 convention",
        )
    except (Exception, PackageNotFoundError) as error:  # pragma: no cover - external API
        return ExternalAgreement(
            "DoubleMLAPOS",
            "unknown",
            "blocked/external-error",
            detail=f"{type(error).__name__}: {error}",
        )


def doubleml_apos(
    x: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    folds: np.ndarray,
    *,
    learner_policy: str,
    known_probabilities: tuple[float, ...],
) -> ExternalAgreement:
    """Fit DoubleML outcome nuisances with known randomized assignment probabilities."""
    try:
        import doubleml as dml
        from doubleml.utils import PSProcessorConfig
    except ImportError:
        return ExternalAgreement("DoubleMLAPOS", "not-installed", "blocked/missing-dependency")
    try:
        propensity_model, _ = _learners(learner_policy)
        levels = tuple(int(value) for value in np.unique(treatment))
        model = dml.DoubleMLAPOS(
            dml.DoubleMLData.from_arrays(x, outcome, treatment),
            ml_g=SelectedOutcomeRegressor(learner_policy),
            ml_m=propensity_model,
            treatment_levels=levels,
            n_folds=len(np.unique(folds)),
            normalize_ipw=False,
            ps_processor_config=PSProcessorConfig(
                clipping_threshold=1e-12,
                extreme_threshold=1e-12,
            ),
            draw_sample_splitting=False,
        )
        model.set_sample_splitting(_splits(folds))
        external = {
            level: {"ml_m": np.full((len(outcome), 1), known_probabilities[code])}
            for code, level in enumerate(levels)
        }
        model.fit(external_predictions=external)
        return ExternalAgreement(
            "DoubleMLAPOS",
            version("doubleml"),
            "complete",
            tuple(float(value) for value in np.ravel(model.coef)),
            tuple(float(value) for value in np.ravel(model.se)),
            detail="Known-design propensities; independently fitted DoubleML outcome nuisances",
        )
    except (Exception, PackageNotFoundError) as error:  # pragma: no cover - external API
        return ExternalAgreement(
            "DoubleMLAPOS",
            "unknown",
            "blocked/external-error",
            detail=f"{type(error).__name__}: {error}",
        )


def econml_drlearner(
    x: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    folds: np.ndarray,
    *,
    learner_policy: str,
    known_probabilities: tuple[float, ...],
) -> ExternalAgreement:
    """Fit EconML with X=None, covariates in W, and frozen fold indices."""
    try:
        from econml.dr import DRLearner
    except ImportError:
        return ExternalAgreement("EconML.DRLearner", "not-installed", "blocked/missing-dependency")
    try:
        levels = tuple(int(value) for value in np.unique(treatment))
        model = DRLearner(
            model_propensity=KnownRandomizationClassifier(known_probabilities),
            model_regression=TreatmentSpecificOutcomeRegressor(
                n_groups=len(levels), learner_policy=learner_policy
            ),
            categories=levels,
            cv=_splits(folds),
            min_propensity=1e-12,
            random_state=17,
        )
        model.fit(outcome, treatment, X=None, W=x)
        estimates = tuple(
            float(np.asarray(model.ate(X=None, T0=levels[0], T1=level)).squeeze())
            for level in levels[1:]
        )
        return ExternalAgreement(
            "EconML.DRLearner",
            version("econml"),
            "complete",
            estimates,
            detail="Known-design propensities; X=None; W=covariates; intercept-only final model",
        )
    except (Exception, PackageNotFoundError) as error:  # pragma: no cover - external API
        return ExternalAgreement(
            "EconML.DRLearner",
            "unknown",
            "blocked/external-error",
            detail=f"{type(error).__name__}: {error}",
        )
