"""Pinned validation-only adapters for DoubleMLAPOS and EconML DRLearner."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


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
) -> ExternalAgreement:
    """Fit DoubleMLAPOS end-to-end with frozen folds and learner classes."""
    try:
        import doubleml as dml
        from doubleml.utils import PSProcessorConfig
    except ImportError:
        return ExternalAgreement("DoubleMLAPOS", "not-installed", "blocked/missing-dependency")
    try:
        propensity_model, outcome_model = _learners(learner_policy)
        levels = tuple(int(value) for value in np.unique(treatment))
        model = dml.DoubleMLAPOS(
            dml.DoubleMLData.from_arrays(x, outcome, treatment),
            ml_g=outcome_model,
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
        model.fit()
        return ExternalAgreement(
            "DoubleMLAPOS",
            version("doubleml"),
            "complete",
            tuple(float(value) for value in np.ravel(model.coef)),
            tuple(float(value) for value in np.ravel(model.se)),
            detail="End-to-end one-vs-rest nuisance fits; 1e-12 propensity floor",
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
) -> ExternalAgreement:
    """Fit EconML with X=None, covariates in W, and frozen fold indices."""
    try:
        from econml.dr import DRLearner
    except ImportError:
        return ExternalAgreement("EconML.DRLearner", "not-installed", "blocked/missing-dependency")
    try:
        propensity_model, outcome_model = _learners(learner_policy)
        levels = tuple(int(value) for value in np.unique(treatment))
        model = DRLearner(
            model_propensity=propensity_model,
            model_regression=outcome_model,
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
            detail="X=None; W=covariates; intercept-only final model; 1e-12 floor",
        )
    except (Exception, PackageNotFoundError) as error:  # pragma: no cover - external API
        return ExternalAgreement(
            "EconML.DRLearner",
            "unknown",
            "blocked/external-error",
            detail=f"{type(error).__name__}: {error}",
        )
