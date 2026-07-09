import numpy as np

from scova import SCOVA, NuisancePredictions, SCOVADeclaration
from scova.simulate import generate_data


def _declaration(seed: int = 31) -> SCOVADeclaration:
    return SCOVADeclaration("outcome", "group", ("x1", "x2", "x3"), n_splits=3, random_state=seed)


def test_oracle_double_robustness_cases() -> None:
    simulation = generate_data("observational", n=6000, seed=17)
    uniform_propensity = np.full_like(simulation.propensity, 1 / len(simulation.group_labels))
    zero_outcome = np.zeros_like(simulation.outcome_regression)
    cases = {
        "correct-both": (simulation.propensity, simulation.outcome_regression),
        "correct-outcome-only": (uniform_propensity, simulation.outcome_regression),
        "correct-propensity-only": (simulation.propensity, zero_outcome),
        "both-wrong": (uniform_propensity, zero_outcome),
    }
    errors: dict[str, float] = {}
    for name, (propensity, outcome_regression) in cases.items():
        nuisance = NuisancePredictions(propensity, outcome_regression, simulation.group_labels)
        result = SCOVA().fit(simulation.data, _declaration(), nuisance_predictions=nuisance)
        errors[name] = float(np.max(np.abs(result.group_means - simulation.true_group_means)))
    assert errors["correct-both"] < 0.12
    assert errors["correct-outcome-only"] < 0.12
    # IPW-only behavior has higher finite-sample variance than the OR-correct cases.
    assert errors["correct-propensity-only"] < 0.20
    assert errors["both-wrong"] > min(
        errors["correct-outcome-only"], errors["correct-propensity-only"]
    )


def test_seeded_oracle_coverage_smoke() -> None:
    covered = 0
    repetitions = 60
    for seed in range(repetitions):
        simulation = generate_data("randomized", n=300, seed=10_000 + seed)
        nuisance = NuisancePredictions(
            simulation.propensity, simulation.outcome_regression, simulation.group_labels
        )
        result = SCOVA().fit(simulation.data, _declaration(seed), nuisance_predictions=nuisance)
        interval = result.contrasts["g0 - g1"].confidence_interval
        truth = float(simulation.true_group_means[0] - simulation.true_group_means[1])
        covered += int(interval[0] <= truth <= interval[1])
    coverage = covered / repetitions
    assert 0.85 <= coverage <= 1.0


def test_fold_assignment_uses_design_columns_only() -> None:
    simulation = generate_data("randomized", n=300, seed=22)
    declaration = _declaration()
    x, _, codes, _ = SCOVA._validate_data(simulation.data, declaration)
    del x
    original = SCOVA._design_folds(simulation.data, declaration, codes)
    changed = simulation.data.copy()
    changed["outcome"] = np.random.default_rng(99).normal(size=len(changed))
    modified = SCOVA._design_folds(changed, declaration, codes)
    np.testing.assert_array_equal(original, modified)
