import numpy as np

from scova.inference import run_simultaneous_inference


def test_global_null_fwer_smoke_across_group_counts() -> None:
    repetitions = 30
    for n_groups in (2, 4, 8):
        rejections = 0
        for repetition in range(repetitions):
            rng = np.random.default_rng(10_000 * n_groups + repetition)
            n = 300
            group_signal = rng.normal(size=(n, n_groups))
            means = group_signal.mean(axis=0)
            influence = group_signal - means
            covariance = np.cov(influence, rowvar=False, ddof=1) / n
            weights = []
            names = []
            for left in range(n_groups):
                for right in range(left + 1, n_groups):
                    weight = np.zeros(n_groups)
                    weight[left] = 1
                    weight[right] = -1
                    weights.append(weight)
                    names.append(f"{left} - {right}")
            weight_matrix = np.vstack(weights)
            estimates = weight_matrix @ means
            contrast_influence = influence @ weight_matrix.T
            standard_errors = np.sqrt(np.diag(weight_matrix @ covariance @ weight_matrix.T))
            inference = run_simultaneous_inference(
                family=tuple(names),
                estimates=estimates,
                standard_errors=standard_errors,
                influence_values=contrast_influence,
                weights=weight_matrix,
                group_covariance=covariance,
                confidence_level=0.95,
                n_bootstrap=499,
                random_state=repetition,
                batch_size=100,
            )
            rejections += inference.global_test.max_t_p_value <= 0.05
        assert rejections / repetitions <= 0.20


def test_simultaneous_coverage_and_power_smoke() -> None:
    repetitions = 25
    family_covered = 0
    global_rejections = 0
    n_groups = 4
    weights = []
    names = []
    for left in range(n_groups):
        for right in range(left + 1, n_groups):
            weight = np.zeros(n_groups)
            weight[left] = 1
            weight[right] = -1
            weights.append(weight)
            names.append(f"{left} - {right}")
    weight_matrix = np.vstack(weights)
    truth_means = np.linspace(-0.75, 0.75, n_groups)
    truth = weight_matrix @ truth_means
    for repetition in range(repetitions):
        rng = np.random.default_rng(50_000 + repetition)
        n = 350
        group_signal = rng.normal(loc=truth_means, size=(n, n_groups))
        means = group_signal.mean(axis=0)
        influence = group_signal - means
        covariance = np.cov(influence, rowvar=False, ddof=1) / n
        estimates = weight_matrix @ means
        contrast_influence = influence @ weight_matrix.T
        standard_errors = np.sqrt(np.diag(weight_matrix @ covariance @ weight_matrix.T))
        inference = run_simultaneous_inference(
            family=tuple(names),
            estimates=estimates,
            standard_errors=standard_errors,
            influence_values=contrast_influence,
            weights=weight_matrix,
            group_covariance=covariance,
            confidence_level=0.95,
            n_bootstrap=499,
            random_state=repetition,
            batch_size=100,
        )
        family_covered += all(
            result.simultaneous_confidence_interval[0]
            <= truth[column]
            <= result.simultaneous_confidence_interval[1]
            for column, result in enumerate(inference.contrasts)
        )
        global_rejections += inference.global_test.max_t_p_value <= 0.05
    assert family_covered / repetitions >= 0.80
    assert global_rejections / repetitions >= 0.80
