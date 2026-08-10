"""Independent estimators used by the descriptive SCOVA-CF comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from benchmarks.cf_comparative_simulation import ComparativeData
from scova import ContrastSpec
from scova.cf import (
    SCOVACF,
    AnalysisMode,
    EstimatedAssignment,
    SCOVACFDeclaration,
    SCOVACFRefusal,
)


@dataclass(frozen=True, slots=True)
class MethodEstimate:
    """One comparator's result for a replication, including its estimand."""

    name: str
    estimand: str
    estimate: float | None
    standard_error: float | None
    status: str
    details: dict[str, Any]


def _arrays(dgp: ComparativeData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = dgp.data.loc[:, [f"x{index}" for index in range(1, 6)]].to_numpy()
    return x, dgp.group.astype(int), dgp.data["outcome"].to_numpy(dtype=float)


def _scova_declaration(seed: int) -> SCOVACFDeclaration:
    covariates = tuple(f"x{index}" for index in range(1, 6))
    return SCOVACFDeclaration(
        outcome="outcome",
        group="group",
        covariates=covariates,
        mode=AnalysisMode.OBSERVATIONAL_CAUSAL,
        scientific_question="What is the study-population standardized group contrast?",
        eligibility="All simulated study units",
        target_population="Eligible simulated study-unit population",
        group_definitions=((0, "untreated group"), (1, "treated group")),
        outcome_time="simulated follow-up",
        outcome_units="outcome units",
        covariate_rationales=tuple((name, "simulated baseline covariate") for name in covariates),
        assignment=EstimatedAssignment(nuisance_strategy="adaptive"),
        outcome_nuisance_strategy="adaptive",
        n_splits=5,
        random_state=seed,
        contrasts=(ContrastSpec("treated-minus-untreated", ((1, 1.0), (0, -1.0))),),
        sensitivity_analysis="Not evaluated in this descriptive simulation study",
    )


def fit_scova_cf(dgp: ComparativeData, seed: int) -> MethodEstimate:
    """Fit SCOVA-CF in its retired, assumption-dependent observational mode."""
    result = SCOVACF().analyze(dgp.data, _scova_declaration(seed))
    if isinstance(result, SCOVACFRefusal):
        return MethodEstimate("scova-cf", "ate", None, None, result.status.code, result.to_dict())
    contrast = result.contrasts["treated-minus-untreated"]
    return MethodEstimate(
        "scova-cf",
        "ate",
        contrast.estimate,
        contrast.standard_error,
        result.status.code,
        {
            "qualification_status": result.status.qualification_status.value,
            "qualification_reason": result.status.qualification_reason,
            "support": result.status.support.value,
        },
    )


def fit_linear_ancova(dgp: ComparativeData, seed: int) -> MethodEstimate:
    """Fit interacting linear ANCOVA and standardize its contrast over all rows."""
    del seed
    x, group, outcome = _arrays(dgp)
    design = np.column_stack((np.ones(len(x)), group, x, group[:, None] * x))
    coefficients, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    contrast_rows = np.column_stack((np.zeros(len(x)), np.ones(len(x)), np.zeros_like(x), x))
    estimate = float(np.mean(contrast_rows @ coefficients))
    residuals = outcome - design @ coefficients
    degrees_of_freedom = max(len(x) - design.shape[1], 1)
    covariance = (residuals @ residuals / degrees_of_freedom) * np.linalg.pinv(design.T @ design)
    contrast_vector = np.mean(contrast_rows, axis=0)
    standard_error = float(np.sqrt(max(contrast_vector @ covariance @ contrast_vector, 0.0)))
    return MethodEstimate("linear-ancova", "ate", estimate, standard_error, "ok", {})


def fit_independent_aipw(dgp: ComparativeData, seed: int) -> MethodEstimate:
    """Cross-fit AIPW without calling SCOVA-CF nuisance or score assembly code."""
    x, group, outcome = _arrays(dgp)
    propensity = np.empty(len(x))
    mu0 = np.empty(len(x))
    mu1 = np.empty(len(x))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in folds.split(x, group):
        propensity_model = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
        propensity_model.fit(x[train], group[train])
        propensity[test] = propensity_model.predict_proba(x[test])[:, 1]
        for value, target in ((0, mu0), (1, mu1)):
            model = HistGradientBoostingRegressor(random_state=seed + value, max_iter=100)
            arm_train = train[group[train] == value]
            model.fit(x[arm_train], outcome[arm_train])
            target[test] = model.predict(x[test])
    bounded_propensity = np.clip(propensity, 1e-3, 1.0 - 1e-3)
    scores = (
        mu1
        - mu0
        + group * (outcome - mu1) / bounded_propensity
        - (1 - group) * (outcome - mu0) / (1 - bounded_propensity)
    )
    return MethodEstimate(
        "independent-aipw",
        "ate",
        float(np.mean(scores)),
        float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
        "ok",
        {"n_splits": 5, "propensity_floor": 1e-3},
    )


def fit_matching_att(dgp: ComparativeData, seed: int) -> MethodEstimate:
    """Match treated units 1:1 without replacement; retain an ATT-only result."""
    del seed
    x, group, outcome = _arrays(dgp)
    score = LogisticRegression(max_iter=1000).fit(x, group).predict_proba(x)[:, 1]
    logit = np.log(score / (1.0 - score))
    treated = np.flatnonzero(group == 1)
    available_controls = set(np.flatnonzero(group == 0).tolist())
    caliper = 0.2 * float(np.std(logit))
    pairs: list[tuple[int, int]] = []
    for treated_index in treated[np.argsort(logit[treated])]:
        candidates = np.array(sorted(available_controls), dtype=int)
        if not len(candidates):
            break
        candidate = int(candidates[np.argmin(np.abs(logit[candidates] - logit[treated_index]))])
        if abs(logit[candidate] - logit[treated_index]) <= caliper:
            pairs.append((int(treated_index), candidate))
            available_controls.remove(candidate)
    retained_fraction = len(pairs) / len(treated)
    if not pairs:
        return MethodEstimate(
            "psm-att", "att", None, None, "limited/no-matches", {"treated_retained_fraction": 0.0}
        )
    differences = np.array([outcome[case] - outcome[control] for case, control in pairs])
    return MethodEstimate(
        "psm-att",
        "att",
        float(np.mean(differences)),
        (
            float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
            if len(differences) > 1
            else None
        ),
        "ok",
        {
            "treated_retained_fraction": retained_fraction,
            "matched_pairs": len(pairs),
            "caliper": caliper,
        },
    )


def fit_econml_drlearner(dgp: ComparativeData, seed: int) -> MethodEstimate:
    """Fit optional EconML DRLearner, returning a recorded block if unavailable."""
    try:
        from econml.dr import DRLearner
    except ImportError:
        return MethodEstimate("econml-drlearner", "ate", None, None, "blocked/missing-econml", {})
    x, group, outcome = _arrays(dgp)
    learner = DRLearner(random_state=seed)
    learner.fit(outcome, group, X=x)
    effects = learner.effect(x)
    return MethodEstimate(
        "econml-drlearner",
        "ate",
        float(np.mean(effects)),
        float(np.std(effects, ddof=1) / np.sqrt(len(effects))),
        "ok",
        {},
    )


def score_replication(dgp: ComparativeData, seed: int) -> list[dict[str, Any]]:
    """Serialize comparator results while preserving each method's estimand."""
    methods = (
        fit_scova_cf,
        fit_linear_ancova,
        fit_independent_aipw,
        fit_matching_att,
        fit_econml_drlearner,
    )
    records: list[dict[str, Any]] = []
    for fit in methods:
        result = fit(dgp, seed)
        truth = dgp.ate if result.estimand == "ate" else dgp.att
        records.append(
            {
                "method": result.name,
                "estimand": result.estimand,
                "estimate": result.estimate,
                "standard_error": result.standard_error,
                "truth": truth,
                "status": result.status,
                "details": result.details,
            }
        )
    return records
