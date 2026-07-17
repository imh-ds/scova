"""Governed SCOVA-CF estimation built on SCOVA's shared numerical engine."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .._aipw import assemble_aipw, validate_probability_matrix
from .._version import __version__
from ..declaration import JsonLabel
from ..estimator import SCOVA
from .benchmarks import lin_interacted_benchmark, unadjusted_benchmark
from .declaration import (
    AnalysisMode,
    EstimatedAssignment,
    KnownAssignment,
    SCOVACFDeclaration,
)
from .result import CFDesignLock, SCOVACFResult, guarded_omnibus
from .status import SCOVACFRefusal, SCOVACFStatus, SupportStatus
from .support import assess_support, influence_concentration


@dataclass(frozen=True, slots=True)
class SCOVACFNuisancePredictions:
    """Observation-aligned nuisance predictions for oracle/reference checks."""

    outcome_regression: np.ndarray
    group_labels: tuple[JsonLabel, ...]
    propensity: np.ndarray | None = None


class SCOVACF:
    """Population-counterfactual SCOVA feature with governed output semantics."""

    def __init__(
        self,
        *,
        propensity_model: BaseEstimator | None = None,
        outcome_model: BaseEstimator | None = None,
    ) -> None:
        if (propensity_model is None) != (outcome_model is None):
            raise ValueError("Custom propensity and outcome models must be supplied together")
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model

    @staticmethod
    def _refusal(
        declaration: SCOVACFDeclaration,
        *,
        code: str,
        reason: str,
        support: SupportStatus = SupportStatus.UNSUPPORTED,
        details: dict[str, Any] | None = None,
    ) -> SCOVACFRefusal:
        return SCOVACFRefusal(
            declaration_hash=declaration.declaration_hash,
            mode=declaration.mode,
            claim_class=declaration.claim_class,
            status=SCOVACFStatus(
                support=support,
                code=code,
                reason=reason,
                confirmatory=False,
            ),
            details={} if details is None else details,
        )

    @staticmethod
    def _native_label(value: Any) -> JsonLabel:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int, float)):
            return value
        raise TypeError("Group labels must be strings, integers, floats, or booleans")

    @staticmethod
    def _label_sort_key(value: JsonLabel) -> tuple[int, Any]:
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float)):
            return (1, float(value))
        return (2, value)

    @staticmethod
    def _known_probability_matrix(
        data: pd.DataFrame,
        assignment: KnownAssignment,
        group_labels: tuple[JsonLabel, ...],
    ) -> np.ndarray:
        label_to_column = {label: code for code, label in enumerate(group_labels)}
        matrix = np.empty((len(data), len(group_labels)), dtype=float)
        if assignment.probabilities:
            mapping = dict(assignment.probabilities)
            for label, code in label_to_column.items():
                matrix[:, code] = mapping[label]
            return matrix
        assert assignment.stratum_column is not None
        by_stratum = {
            stratum: dict(probabilities)
            for stratum, probabilities in assignment.stratum_probabilities
        }
        for row, stratum in enumerate(data[assignment.stratum_column]):
            probabilities = by_stratum[stratum]
            for label, code in label_to_column.items():
                matrix[row, code] = probabilities[label]
        return matrix

    @staticmethod
    def _design_folds(
        data: pd.DataFrame,
        declaration: SCOVACFDeclaration,
        group_codes: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        assignment = declaration.assignment
        strata_column = (
            assignment.stratum_column if isinstance(assignment, KnownAssignment) else None
        )
        columns = [declaration.group, *declaration.covariates]
        if strata_column is not None and strata_column not in columns:
            columns.append(strata_column)
        design = data.loc[:, columns]
        hashes = pd.util.hash_pandas_object(design, index=False, categorize=True).to_numpy(
            dtype=np.uint64
        )
        salt = np.uint64(declaration.random_state % (2**32)) * np.uint64(0x9E3779B1)
        hashes = hashes ^ salt
        strata_codes: np.ndarray | None = None
        stratified_by_design = False
        if strata_column is not None:
            strata_codes = pd.factorize(data[strata_column], sort=True)[0]
            cell_counts = np.bincount(
                group_codes * (int(strata_codes.max()) + 1) + strata_codes
            )
            stratified_by_design = bool(
                np.all(cell_counts[cell_counts > 0] >= declaration.n_splits)
            )
        fold_cells = (
            group_codes * (int(strata_codes.max()) + 1) + strata_codes
            if stratified_by_design and strata_codes is not None
            else group_codes
        )
        folds = np.empty(len(data), dtype=int)
        for cell in np.unique(fold_cells):
            indices = np.flatnonzero(fold_cells == cell)
            order = indices[np.argsort(hashes[indices], kind="stable")]
            folds[order] = np.arange(len(order)) % declaration.n_splits
        return folds, stratified_by_design

    @staticmethod
    def _design_lock(
        data: pd.DataFrame,
        declaration: SCOVACFDeclaration,
        folds: np.ndarray,
    ) -> CFDesignLock:
        assignment = declaration.assignment
        strata_column = (
            assignment.stratum_column if isinstance(assignment, KnownAssignment) else None
        )
        columns = [declaration.group, *declaration.covariates]
        if strata_column is not None and strata_column not in columns:
            columns.append(strata_column)
        design_hashes = pd.util.hash_pandas_object(
            data.loc[:, columns], index=True, categorize=True
        ).to_numpy(dtype=np.uint64)
        digest = sha256()
        digest.update(declaration.declaration_hash.encode("ascii"))
        digest.update(design_hashes.tobytes())
        return CFDesignLock(
            declaration_hash=declaration.declaration_hash,
            design_hash=digest.hexdigest(),
            fold_hash=sha256(np.asarray(folds, dtype=np.int64).tobytes()).hexdigest(),
            row_count=len(data),
        )

    def analyze(
        self,
        data: pd.DataFrame,
        declaration: SCOVACFDeclaration,
        *,
        nuisance_predictions: SCOVACFNuisancePredictions | None = None,
    ) -> SCOVACFResult | SCOVACFRefusal:
        """Run SCOVA-CF or return a typed refusal for an unmet prerequisite."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if declaration.post_treatment_covariates:
            return self._refusal(
                declaration,
                code="refused/post-treatment-covariate",
                reason=(
                    "Post-treatment covariates are prohibited in the core total-effect analysis: "
                    f"{list(declaration.post_treatment_covariates)}"
                ),
            )
        required = [declaration.outcome, declaration.group, *declaration.covariates]
        assignment = declaration.assignment
        if isinstance(assignment, KnownAssignment) and assignment.stratum_column is not None:
            required.append(assignment.stratum_column)
        missing_columns = [column for column in required if column not in data]
        if missing_columns:
            return self._refusal(
                declaration,
                code="refused/missing-column",
                reason=f"Data is missing declared columns: {missing_columns}",
                details={"missing_columns": missing_columns},
            )
        design_value_columns = [declaration.group, *declaration.covariates]
        if isinstance(assignment, KnownAssignment) and assignment.stratum_column is not None:
            design_value_columns.append(assignment.stratum_column)
        if data.loc[:, design_value_columns].isna().any().any():
            return self._refusal(
                declaration,
                code="refused/missing-analysis-values",
                reason="Group and baseline covariates must be complete in this release",
            )
        try:
            x = data.loc[:, declaration.covariates].to_numpy(dtype=float)
            raw_labels = [
                self._native_label(value) for value in pd.unique(data[declaration.group])
            ]
        except (TypeError, ValueError) as error:
            return self._refusal(
                declaration,
                code="refused/invalid-analysis-data",
                reason=f"Baseline covariates must be finite numeric values: {error}",
            )
        if not np.all(np.isfinite(x)):
            return self._refusal(
                declaration,
                code="refused/invalid-analysis-data",
                reason="Baseline covariates must be finite",
            )
        labels = tuple(sorted(raw_labels, key=self._label_sort_key))
        label_to_code = {label: code for code, label in enumerate(labels)}
        group_codes = np.array(
            [self._native_label(value) for value in data[declaration.group]], dtype=object
        )
        group_codes = np.array([label_to_code[value] for value in group_codes], dtype=int)
        observed_labels = set(labels)
        declared_labels = set(declaration.declared_group_labels)
        if observed_labels != declared_labels:
            missing_groups = sorted(map(str, declared_labels - observed_labels))
            unexpected_groups = sorted(map(str, observed_labels - declared_labels))
            code = "refused/empty-cell" if missing_groups else "refused/invalid-groups"
            return self._refusal(
                declaration,
                code=code,
                reason="Observed and declared groups do not match",
                details={
                    "missing_declared_groups": missing_groups,
                    "unexpected_groups": unexpected_groups,
                },
            )
        if isinstance(assignment, KnownAssignment):
            if set(assignment.group_labels) != declared_labels:
                return self._refusal(
                    declaration,
                    code="refused/invalid-assignment-mechanism",
                    reason="Known assignment groups must exactly match declared groups",
                )
            if assignment.stratum_column is not None:
                declared_strata = {value for value, _ in assignment.stratum_probabilities}
                observed_strata = set(data[assignment.stratum_column].tolist())
                if not observed_strata.issubset(declared_strata):
                    return self._refusal(
                        declaration,
                        code="refused/invalid-assignment-mechanism",
                        reason="Observed randomization strata lack declared probabilities",
                    )
        for specification in declaration.contrasts:
            if not set(dict(specification.weights)).issubset(declared_labels):
                return self._refusal(
                    declaration,
                    code="refused/invalid-contrast",
                    reason=f"Contrast {specification.name!r} contains an undeclared group",
                )
        counts = np.bincount(group_codes, minlength=len(labels))
        if np.any(counts < declaration.n_splits):
            too_small = {
                str(labels[code]): int(count)
                for code, count in enumerate(counts)
                if count < declaration.n_splits
            }
            return self._refusal(
                declaration,
                code="limited/small-sample-restricted-library",
                reason=(
                    "Every group needs at least n_splits observations for cross-fitting; "
                    f"too small: {too_small}"
                ),
                support=SupportStatus.UNSTABLE,
            )
        # Lock outcome-free design inputs and folds before inspecting outcome values.
        folds, design_stratified = self._design_folds(data, declaration, group_codes)
        design_lock = self._design_lock(data, declaration, folds)
        if data[declaration.outcome].isna().any():
            return self._refusal(
                declaration,
                code="limited/missing-outcomes",
                reason="The reference SCOVA-CF estimator requires complete declared outcomes",
            )
        try:
            outcome = data[declaration.outcome].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            return self._refusal(
                declaration,
                code="refused/invalid-analysis-data",
                reason=f"Outcome must be numeric: {error}",
            )
        if not np.all(np.isfinite(outcome)):
            return self._refusal(
                declaration,
                code="refused/invalid-analysis-data",
                reason="Outcome must be finite",
            )
        engine_strategy = declaration.outcome_nuisance_strategy
        if isinstance(assignment, EstimatedAssignment):
            if assignment.nuisance_strategy != engine_strategy:
                return self._refusal(
                    declaration,
                    code="refused/incompatible-nuisance-policy",
                    reason=(
                        "This reference release requires the same strategy for propensity and "
                        "outcome nuisance learners"
                    ),
                )
            engine_strategy = assignment.nuisance_strategy
        if engine_strategy == "custom" and self.propensity_model is None:
            return self._refusal(
                declaration,
                code="refused/missing-custom-learner",
                reason="The declaration requires custom nuisance learners but none were supplied",
            )
        if engine_strategy != "custom" and self.propensity_model is not None:
            return self._refusal(
                declaration,
                code="refused/incompatible-nuisance-policy",
                reason="Custom learners were supplied but the declaration did not select them",
            )
        known_propensity = (
            self._known_probability_matrix(data, assignment, labels)
            if isinstance(assignment, KnownAssignment)
            else None
        )
        model = SCOVA(
            propensity_model=self.propensity_model,
            outcome_model=self.outcome_model,
            nuisance_strategy=engine_strategy,
        )
        if nuisance_predictions is None:
            try:
                propensity, outcome_regression, nuisance_metadata = model._cross_fit(
                    x,
                    group_codes,
                    outcome,
                    folds,
                    len(labels),
                    labels,
                    known_propensity=known_propensity,
                )
            except (TypeError, ValueError) as error:
                return self._refusal(
                    declaration,
                    code="refused/nuisance-fit",
                    reason=str(error),
                )
        else:
            if tuple(nuisance_predictions.group_labels) != labels:
                return self._refusal(
                    declaration,
                    code="refused/nuisance-labels",
                    reason="Nuisance group labels must match canonical analysis labels",
                )
            outcome_regression = np.asarray(
                nuisance_predictions.outcome_regression, dtype=float
            )
            if known_propensity is not None:
                propensity = known_propensity
                propensity_source = "known-design"
            elif nuisance_predictions.propensity is not None:
                propensity = np.asarray(nuisance_predictions.propensity, dtype=float)
                propensity_source = "supplied"
            else:
                return self._refusal(
                    declaration,
                    code="refused/missing-propensity",
                    reason="Estimated-assignment oracle input requires propensity predictions",
                )
            nuisance_metadata = {
                "source": "supplied",
                "propensity_model": propensity_source,
                "outcome_model": "supplied",
            }
        try:
            propensity = validate_probability_matrix(propensity, len(data), len(labels))
        except ValueError as error:
            return self._refusal(
                declaration,
                code="refused/positivity",
                reason=str(error),
            )
        support = assess_support(
            x=x,
            group_codes=group_codes,
            propensity=propensity,
            folds=folds,
            covariate_names=declaration.covariates,
            group_labels=labels,
            policy=declaration.support_policy,
            assignment_source=("known-design" if known_propensity is not None else "estimated"),
        )
        status = support.status
        if (
            declaration.mode is AnalysisMode.OBSERVATIONAL_CAUSAL
            and not declaration.sensitivity_analysis
        ):
            status = SCOVACFStatus(
                support=SupportStatus.UNSTABLE,
                code="limited/required-sensitivity-analysis",
                reason=(
                    f"{status.reason}; observational-causal promotion requires a prespecified "
                    "quantitative sensitivity analysis"
                ),
                confirmatory=False,
            )
        try:
            means, influence, covariance = assemble_aipw(
                outcome, group_codes, propensity, outcome_regression
            )
        except ValueError as error:
            return self._refusal(
                declaration,
                code="refused/nuisance-predictions",
                reason=str(error),
            )
        diagnostics = {
            "support": support.diagnostics,
            "influence_concentration": influence_concentration(influence, labels),
            "design_stratified_folds": design_stratified,
        }
        benchmarks = {
            "unadjusted": unadjusted_benchmark(outcome, group_codes, labels),
            "lin_interacted": lin_interacted_benchmark(outcome, x, group_codes, labels),
        }
        evidence_card = {
            "mode": declaration.mode.value,
            "claim_class": declaration.claim_class.value,
            "question": declaration.scientific_question,
            "population": declaration.target_population,
            "eligibility": declaration.eligibility,
            "groups": [
                {"label": label, "definition": definition}
                for label, definition in declaration.group_definitions
            ],
            "outcome": {
                "column": declaration.outcome,
                "time": declaration.outcome_time,
                "units": declaration.outcome_units,
                "direction": declaration.outcome_direction,
            },
            "estimand": {
                "id": declaration.estimand_id,
                "mathematical": "psi_g = E_PX[E(Y | G=g, X)]",
                "plain_language": (
                    "Each group mean is standardized to the same declared target population"
                ),
            },
            "contrasts": [contrast.to_dict() for contrast in declaration.contrasts],
            "support_status": status.to_dict(),
            "independent_unit": "row",
            "estimator": declaration.estimator,
            "missingness": declaration.missing_outcome_policy,
            "scientific_boundary": (
                "Population counterfactual means; no person-specific missing outcomes or paired "
                "testing of predictions"
            ),
        }
        omnibus = guarded_omnibus(
            means=means,
            covariance=covariance,
            mode=declaration.mode,
            claim_class=declaration.claim_class,
            status=status,
        )
        result = SCOVACFResult(
            group_labels=labels,
            covariate_names=declaration.covariates,
            group_means=means,
            influence_values=influence,
            covariance=covariance,
            fold_assignments=folds,
            propensity_predictions=propensity,
            outcome_predictions=outcome_regression,
            diagnostics=diagnostics,
            declaration=declaration.to_dict(),
            declaration_hash=declaration.declaration_hash,
            design_lock=design_lock,
            nuisance_metadata=nuisance_metadata,
            mode=declaration.mode,
            claim_class=declaration.claim_class,
            status=status,
            estimand_id=declaration.estimand_id,
            target_population=declaration.target_population,
            outcome_units=declaration.outcome_units,
            benchmarks=benchmarks,
            evidence_card=evidence_card,
            omnibus=omnibus,
            random_state=declaration.random_state,
            package_version=__version__,
        )
        for specification in declaration.contrasts:
            result.contrast(dict(specification.weights), name=specification.name)
        return result
