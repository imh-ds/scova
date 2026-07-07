"""Experimental finite-grid overlap-path example."""

from scova import SCOVADeclaration
from scova.experimental import PathDeclaration, fit_path
from scova.simulate import generate_data


def main() -> None:
    simulation = generate_data("observational", n=1200, seed=2026)
    base = SCOVADeclaration(
        outcome="outcome",
        group="group",
        covariates=("x1", "x2", "x3"),
        n_splits=5,
        random_state=2026,
    )
    declaration = PathDeclaration(base)
    result = fit_path(simulation.data, declaration)
    inference = result.infer()
    contrast = result.contrasts["g0 - g2"]
    certificate = inference.sign_certificates[inference.family.index("g0 - g2")]
    stability = inference.stability_certificates[inference.family.index("g0 - g2")]
    print("Lambda grid:", result.lambdas)
    print("g0 - g2 path:", contrast.estimates)
    print("Positive certified grid:", certificate.positive_lambdas)
    print("Negative suffix starts at:", certificate.negative_suffix_start)
    print("Stability upper bound:", stability.simultaneous_upper_bound)
    print("Target ESS path:", result.drift.target_effective_sample_size)
    result.save("overlap_path_result.scova")


if __name__ == "__main__":
    main()

