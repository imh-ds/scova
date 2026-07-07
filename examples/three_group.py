"""Reproducible three-group SCOVA vertical-slice example."""

from scova import SCOVA, SCOVADeclaration
from scova.simulate import generate_data


def main() -> None:
    simulation = generate_data("observational", n=1200, seed=2026)
    declaration = SCOVADeclaration(
        outcome="outcome",
        group="group",
        covariates=("x1", "x2", "x3"),
        n_splits=5,
        random_state=2026,
    )
    result = SCOVA().fit(simulation.data, declaration)
    simultaneous = result.infer()
    means = dict(zip(result.group_labels, result.group_means, strict=True))
    contrast = result.contrasts["g0 - g2"]
    print("Standardized means:", means)
    print("g0 - g2 estimate:", contrast.estimate)
    print("95% interval:", contrast.confidence_interval)
    print(
        "Simultaneous 95% interval:",
        simultaneous.contrast("g0 - g2").simultaneous_confidence_interval,
    )
    print("Global tests:", simultaneous.global_test)
    print("ESS:", result.diagnostics["effective_sample_sizes"])
    result.save("three_group_result.scova")


if __name__ == "__main__":
    main()
