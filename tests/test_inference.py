import json
from dataclasses import replace

import numpy as np
import pytest

from scova import (
    SCOVA,
    InferenceStatus,
    NuisancePredictions,
    SCOVADeclaration,
    SCOVAResult,
    Verdict,
)
from scova.simulate import generate_data


def fitted_result(*, causal: bool = False) -> SCOVAResult:
    simulation = generate_data("randomized", n=360, n_groups=3, seed=13)
    nuisance = NuisancePredictions(
        simulation.propensity, simulation.outcome_regression, simulation.group_labels
    )
    declaration = SCOVADeclaration(
        "outcome",
        "group",
        ("x1", "x2", "x3"),
        interpretation="causal" if causal else "descriptive",
        n_splits=3,
        random_state=77,
    )
    return SCOVA().fit(simulation.data, declaration, nuisance_predictions=nuisance)


def test_bootstrap_matches_direct_unbatched_algebra() -> None:
    result = fitted_result()
    family = ("g0 - g1", "g0 - g2")
    inference = result.infer(
        family, confidence_level=0.9, n_bootstrap=31, random_state=5, batch_size=31
    )
    selected = [result.contrasts[name] for name in family]
    influence = np.column_stack([contrast.influence_values for contrast in selected])
    errors = np.array([contrast.standard_error for contrast in selected])
    rng = np.random.default_rng(5)
    multipliers = rng.normal(size=(31, influence.shape[0]))
    multipliers -= multipliers.mean(axis=1, keepdims=True)
    bootstrap = np.max(np.abs((multipliers @ influence) / len(influence) / errors), axis=1)
    expected_critical = np.quantile(bootstrap, 0.9, method="higher")
    assert inference.critical_value == pytest.approx(expected_critical)
    for column, name in enumerate(family):
        observed = abs(selected[column].estimate / errors[column])
        expected_p = (1 + np.sum(bootstrap >= observed)) / 32
        assert inference.contrast(name).adjusted_p_value == pytest.approx(expected_p)


def test_determinism_batching_and_family_order() -> None:
    result = fitted_result()
    family = tuple(result.contrasts)
    first = result.infer(family, n_bootstrap=127, random_state=9, batch_size=17)
    second = result.infer(family, n_bootstrap=127, random_state=9, batch_size=127)
    assert first.to_dict() == second.to_dict()
    reversed_result = result.infer(
        tuple(reversed(family)), n_bootstrap=127, random_state=9, batch_size=31
    )
    assert first.critical_value == pytest.approx(reversed_result.critical_value)
    for name in family:
        assert first.contrast(name).adjusted_p_value == pytest.approx(
            reversed_result.contrast(name).adjusted_p_value
        )


def test_redundant_pairwise_family_has_rank_k_minus_one() -> None:
    result = fitted_result()
    inference = result.infer(n_bootstrap=49)
    assert inference.family == tuple(result.contrasts)
    assert inference.global_test.wald_degrees_of_freedom == 2
    assert 0 <= inference.global_test.max_t_p_value <= 1
    assert 0 <= inference.global_test.wald_p_value <= 1
    assert inference.status is InferenceStatus.COMPLETE


def test_family_validation_and_zero_variance_refusal() -> None:
    result = fitted_result()
    with pytest.raises(ValueError, match="at least one"):
        result.infer(())
    with pytest.raises(ValueError, match="duplicate"):
        result.infer(("g0 - g1", "g0 - g1"))
    with pytest.raises(ValueError, match="Unknown"):
        result.infer(("not fitted",))
    original = result.contrasts["g0 - g1"]
    result.contrasts["zero"] = replace(
        original,
        name="zero",
        standard_error=0.0,
        influence_values=np.zeros_like(original.influence_values),
    )
    with pytest.raises(ValueError, match="strictly positive"):
        result.infer(("zero",))


def test_inference_round_trip_and_recomputation(tmp_path) -> None:
    result = fitted_result()
    inference = result.infer(n_bootstrap=63)
    path = tmp_path / "with-inference.scova"
    result.save(path)
    loaded = SCOVAResult.load(path)
    assert loaded.random_state == 77
    assert loaded.interpretation == "descriptive"
    assert loaded.inferences[inference.configuration_key].to_dict() == inference.to_dict()
    recomputed = loaded.infer(n_bootstrap=63)
    assert recomputed.to_dict() == inference.to_dict()


def test_schema_one_migration_and_causal_semantics(tmp_path) -> None:
    result = fitted_result(causal=True)
    assert result.verdict is Verdict.EXPLORATORY_ONLY
    result.verdict = Verdict.CERTIFIED
    metadata = {
        "schema_version": 1,
        "package_version": "0.1.0",
        "group_labels": list(result.group_labels),
        "covariate_names": list(result.covariate_names),
        "diagnostics": result.diagnostics,
        "declaration_hash": result.declaration_hash,
        "nuisance_metadata": result.nuisance_metadata,
        "verdict": "certified",
        "contrasts": {
            name: {
                "weights": contrast.weights.tolist(),
                "estimate": contrast.estimate,
                "standard_error": contrast.standard_error,
                "confidence_interval": list(contrast.confidence_interval),
                "z_statistic": contrast.z_statistic,
                "p_value": contrast.p_value,
            }
            for name, contrast in result.contrasts.items()
        },
    }
    arrays = {
        "metadata": np.array(json.dumps(metadata)),
        "group_means": result.group_means,
        "influence_values": result.influence_values,
        "covariance": result.covariance,
        "fold_assignments": result.fold_assignments,
        "propensity_predictions": result.propensity_predictions,
        "outcome_predictions": result.outcome_predictions,
    }
    for name, contrast in result.contrasts.items():
        arrays[f"contrast_influence::{name}"] = contrast.influence_values
    path = tmp_path / "schema-one.scova"
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    migrated = SCOVAResult.load(path)
    assert migrated.schema_version == 3
    assert migrated.random_state == 0
    assert migrated.interpretation == "causal"
    assert migrated.verdict is Verdict.EXPLORATORY_ONLY
    assert migrated.inferences == {}

    metadata["schema_version"] = 2
    metadata["interpretation"] = "causal"
    metadata["random_state"] = 41
    metadata["inferences"] = {}
    arrays["metadata"] = np.array(json.dumps(metadata))
    schema_two_path = tmp_path / "schema-two.scova"
    with schema_two_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    migrated_two = SCOVAResult.load(schema_two_path)
    assert migrated_two.schema_version == 3
    assert migrated_two.random_state == 41


def test_diagnostic_warning_propagates() -> None:
    result = fitted_result()
    result.diagnostics["warnings"] = ["support diagnostic requires review"]
    inference = result.infer(n_bootstrap=19)
    assert inference.status is InferenceStatus.WARNING
    assert inference.reasons == ("support diagnostic requires review",)
