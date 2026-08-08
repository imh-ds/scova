"""Cross-fitted fixed-target AIPW estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._aipw import assemble_aipw, validate_probability_matrix
from ._version import __version__
from .declaration import JsonLabel, SCOVADeclaration
from .diagnostics import compute_diagnostics
from .result import SCOVAResult, Verdict

# Preserve the existing private import surface used by the repository's
# regression suite while keeping the implementation centralized.
_validate_probabilities = validate_probability_matrix
_assemble_aipw = assemble_aipw

# The adaptive strategy always cross-fits this propensity learner; see
# SCOVA._adaptive_propensity_model for why the choice is not data-dependent.
_FLEXIBLE_PROPENSITY = "HistGradientBoostingClassifier"


def _scaled(name: str, model: BaseEstimator) -> Pipeline:
    """Standardize covariates before a linear learner.

    Neither LogisticRegression nor Ridge is scale free. On covariates whose
    columns differ by orders of magnitude the fit is badly conditioned: the
    solver's path then depends on floating-point ordering, so the same input on
    different hardware lands on materially different coefficients. Measured on
    the breast-cancer plasmode cells, whose columns span a ~215,000x scale
    ratio, a third of contrasts failed to reproduce between two runs of an
    identical campaign.

    Scaling also fixes what the penalty means. Ridge's alpha and logistic's L2
    shrink coefficients toward zero in the units they are expressed in, so on
    raw columns they crush small-scale covariates and barely touch large-scale
    ones -- the penalty ends up depending on whether a covariate was recorded in
    millimetres or kilometres.

    Tree learners are left alone: they split on order, not magnitude, and are
    already invariant. Only the linear learners are wrapped.
    """
    return Pipeline([("scale", StandardScaler()), (name, model)])


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


class SCOVA:
    """Fixed-target, cross-fitted multi-group AIPW estimator."""

    def __init__(
        self,
        *,
        propensity_model: BaseEstimator | None = None,
        outcome_model: BaseEstimator | None = None,
        nuisance_strategy: Literal["adaptive", "linear", "custom"] = "adaptive",
        propensity_parameterization: Literal["multiclass", "one-vs-rest"] = "multiclass",
    ) -> None:
        if nuisance_strategy not in {"adaptive", "linear", "custom"}:
            raise ValueError("nuisance_strategy must be 'adaptive', 'linear', or 'custom'")
        if propensity_parameterization not in {"multiclass", "one-vs-rest"}:
            raise ValueError("propensity_parameterization must be 'multiclass' or 'one-vs-rest'")
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
        # Default stays "multiclass" deliberately. One-vs-rest columns do not
        # sum to one, and not every consumer of a fitted propensity is an AIPW
        # arm equation: `bounded_pairwise_anchor` scores `one_hot - propensity`,
        # the multinomial score residual, which is only a residual on the
        # simplex. SCOVA-CF opts in because its arm equations qualify; the
        # anchor path must not be switched without deriving its score again.
        self.propensity_parameterization = propensity_parameterization

    @staticmethod
    def _linear_propensity_model() -> BaseEstimator:
        return _scaled("LogisticRegression", LogisticRegression(max_iter=2000))

    @staticmethod
    def _linear_outcome_model() -> BaseEstimator:
        return _scaled("Ridge", Ridge(alpha=1.0))

    @staticmethod
    def _adaptive_propensity_candidates() -> dict[str, BaseEstimator]:
        return {
            "LogisticRegression": _scaled("LogisticRegression", LogisticRegression(max_iter=2000)),
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
                learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
            ),
        }

    @staticmethod
    def _adaptive_outcome_candidates() -> dict[str, BaseEstimator]:
        return {
            "Ridge": _scaled("Ridge", Ridge(alpha=1.0)),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
            ),
        }

    @classmethod
    def _adaptive_propensity_model(cls) -> tuple[BaseEstimator, str]:
        """Return the flexible propensity learner, unconditionally.

        This deliberately does not hold a contest.  Selecting the propensity by
        predictive fit is unsound: log loss scores prediction of group
        membership, but the propensity's job in AIPW is confounding control, and
        the two disagree exactly where it matters.  Under weak-to-moderate
        nonlinear confounding the nonlinear signal is too faint for boosting to
        win on log loss, so the misspecified linear learner was chosen almost
        always -- measured at 96% of folds with two groups and 100% with three
        -- and its bias passed into the estimate while the intervals stayed
        narrow.  No margin or tie-break repairs a contest that one-sided.

        The trade is asymmetric, which is why it is worth making.  Choosing the
        linear learner wrongly is a validity failure: coverage fell to 0.53-0.74
        against a nominal 0.95.  Choosing the flexible one wrongly costs only
        precision -- on genuinely linear propensities it stays at or slightly
        above nominal coverage (0.95-0.99) with wider intervals, up to 1.4x RMSE
        with two groups and 2.8x with three at small samples, decaying to <=1.2x
        by 800 per group.  Callers who know the propensity is linear can still
        say so with ``nuisance_strategy="linear"``.

        Evidence: benchmarks/selector_study.py, Actions run 30188022106
        (64 design cells, 400 replications, pinned numerical stack).
        """
        return clone(cls._adaptive_propensity_candidates()[_FLEXIBLE_PROPENSITY]), (
            _FLEXIBLE_PROPENSITY
        )

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
        known_propensity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        propensity = np.empty((len(outcome), n_groups), dtype=float)
        outcome_regression = np.empty((len(outcome), n_groups), dtype=float)
        propensity_selected: list[dict[str, Any]] = []
        outcome_selected: dict[str, list[dict[str, Any]]] = {str(label): [] for label in labels}
        if known_propensity is not None:
            known_propensity = validate_probability_matrix(known_propensity, len(outcome), n_groups)
        for fold in sorted(np.unique(folds)):
            test = folds == fold
            train = ~test
            if known_propensity is not None:
                propensity[test] = known_propensity[test]
            elif self.nuisance_strategy == "adaptive":
                propensity_model, propensity_name = self._adaptive_propensity_model()
                propensity_selected.append({"fold": int(fold), "selected": propensity_name})
            elif known_propensity is None and self.nuisance_strategy == "linear":
                propensity_model = self._linear_propensity_model()
            elif known_propensity is None:
                assert self.propensity_model is not None
                propensity_model = clone(self.propensity_model)
            if known_propensity is None and self._fits_one_vs_rest(n_groups):
                # One binary model per arm, each cloned from the SAME spec the
                # multiclass path would have used -- matching the declared
                # learner family is not optional. Fitting one-vs-rest with a
                # different family than the cell declares produces a comparison
                # of learners masquerading as a comparison of parameterizations.
                for code in range(n_groups):
                    membership = (group_codes[train] == code).astype(int)
                    if membership.min() == membership.max():
                        raise ValueError("Every propensity training fold must contain every group")
                    arm_model = clone(propensity_model)
                    arm_model.fit(x[train], membership)
                    arm_classes = np.asarray(arm_model.classes_, dtype=int)
                    positive = int(np.flatnonzero(arm_classes == 1)[0])
                    propensity[test, code] = np.asarray(
                        arm_model.predict_proba(x[test]), dtype=float
                    )[:, positive]
            elif known_propensity is None:
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
            "source": (
                "cross-fitted-outcome-known-assignment"
                if known_propensity is not None
                else "cross-fitted"
            ),
            "nuisance_strategy": self.nuisance_strategy,
            "propensity_model": (
                "known-design"
                if known_propensity is not None
                else self._metadata_model_name("propensity")
            ),
            "outcome_model": self._metadata_model_name("outcome"),
            # Recorded even when it did not bind, so a stored result says which
            # parameterization produced it rather than leaving it to be inferred
            # from the column sums.
            "propensity_parameterization": (
                "known-design"
                if known_propensity is not None
                else ("one-vs-rest" if self._fits_one_vs_rest(n_groups) else "multiclass")
            ),
        }
        if self.nuisance_strategy == "adaptive":
            metadata["selection"] = {
                # The propensity is fixed, not scored: a predictive criterion
                # selects the wrong learner for confounding control. Only the
                # outcome regression is still chosen from the data.
                "criterion": {
                    "propensity": "fixed-flexible",
                    "outcome": "mean_squared_error",
                },
                "inner_folds": 3,
                "propensity": propensity_selected if known_propensity is None else [],
                "outcome": outcome_selected,
            }
        return propensity, outcome_regression, metadata

    def _fits_one_vs_rest(self, n_groups: int) -> bool:
        """One-vs-rest only bites at three or more arms.

        At two arms the parameterizations are the same estimator, so there is
        nothing to gain -- but refitting as two separate binary problems would
        still perturb the last bits, because a solver run on ``y`` and on
        ``1 - y`` need not return exactly negated coefficients. Keeping k=2 on
        the single multiclass fit makes two-arm output BIT-identical rather
        than merely equal to machine precision, which is what lets v3-v9's
        two-arm numbers be checked rather than argued about.
        """
        return self.propensity_parameterization == "one-vs-rest" and n_groups > 2

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
        # Only columns this estimator fitted one-vs-rest are exempt from the
        # simplex; supplied predictions carry no such key and stay strict.
        propensity = validate_probability_matrix(
            propensity,
            n,
            n_groups,
            require_simplex=nuisance_metadata.get("propensity_parameterization") != "one-vs-rest",
        )
        means, influence, covariance = assemble_aipw(
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
