import numpy as np

from examples.scova_cf_nonlinear_benefit import run_demo


def test_nonlinear_cf_example_targets_common_population_and_improves_accuracy() -> None:
    summary = run_demo(n=1200, seed=2)
    errors = summary["rmse_against_truth"]
    assert errors["scova_cf"] < errors["raw_anova_style"]
    assert errors["scova_cf"] < errors["linear_ancova"]
    assert summary["estimand_id"] == "study-population-standardized-means"
    assert summary["claim_class"] == "randomization-supported"
    assert summary["known_assignment_used"] is True
    assert summary["support_status"]["confirmatory"] is False
    assert len(summary["core_benefits"]) == 4
    assert "not guaranteed" in summary["interpretation_caveat"]
    assert np.all(np.isfinite(summary["scova_cf_means"]))
