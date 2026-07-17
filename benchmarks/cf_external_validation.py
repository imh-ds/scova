"""Optional external numerical checks for the SCOVA-CF promotion gate.

The external packages are deliberately validation extras rather than SCOVA runtime
dependencies. A missing or failed comparison is recorded as a blocked gate, never as
agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass(frozen=True, slots=True)
class ExternalAgreement:
    implementation: str
    version: str
    status: str
    estimates: tuple[float, ...] = ()
    standard_errors: tuple[float, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "version": self.version,
            "status": self.status,
            "estimates": list(self.estimates),
            "standard_errors": list(self.standard_errors),
            "detail": self.detail,
        }


def fixed_nuisance_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: np.ndarray,
    outcome_regression: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent literal implementation of the unnormalized AIPW definition."""
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


def doubleml_apos(
    x: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, *, n_folds: int = 3
) -> ExternalAgreement:
    """Fit DoubleMLAPOS and return its average-potential-outcome vector."""
    try:
        import doubleml as dml
    except ImportError:
        return ExternalAgreement(
            "DoubleMLAPOS", "not-installed", "blocked/missing-dependency"
        )
    try:
        data = dml.DoubleMLData.from_arrays(x, outcome, treatment)
        model = dml.DoubleMLAPOS(
            data,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(max_iter=2000),
            treatment_levels=tuple(int(value) for value in np.unique(treatment)),
            n_folds=n_folds,
        )
        model.fit()
        return ExternalAgreement(
            "DoubleMLAPOS",
            version("doubleml"),
            "complete",
            tuple(float(value) for value in np.ravel(model.coef)),
            tuple(float(value) for value in np.ravel(model.se)),
        )
    except Exception as error:  # pragma: no cover - depends on external API/version
        return ExternalAgreement(
            "DoubleMLAPOS",
            version("doubleml"),
            "blocked/external-error",
            detail=f"{type(error).__name__}: {error}",
        )


def econml_drlearner(
    x: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, *, n_folds: int = 3
) -> ExternalAgreement:
    """Fit EconML DRLearner and return each nonreference ATE contrast."""
    try:
        from econml.dr import DRLearner
    except ImportError:
        return ExternalAgreement(
            "EconML.DRLearner", "not-installed", "blocked/missing-dependency"
        )
    try:
        levels = tuple(int(value) for value in np.unique(treatment))
        model = DRLearner(
            model_propensity=LogisticRegression(max_iter=2000),
            model_regression=LinearRegression(),
            categories=levels,
            cv=n_folds,
            # Clipping is effectively inactive in the prespecified strong-support
            # fixtures while satisfying EconML versions that require a positive floor.
            min_propensity=1e-12,
            random_state=17,
        )
        model.fit(outcome, treatment, X=x)
        estimates = tuple(
            float(np.asarray(model.ate(X=x, T0=levels[0], T1=level)).squeeze())
            for level in levels[1:]
        )
        return ExternalAgreement(
            "EconML.DRLearner",
            version("econml"),
            "complete",
            estimates,
            detail="Contrasts are relative to the first treatment level",
        )
    except (Exception, PackageNotFoundError) as error:  # pragma: no cover
        return ExternalAgreement(
            "EconML.DRLearner",
            "unknown",
            "blocked/external-error",
            detail=f"{type(error).__name__}: {error}",
        )
