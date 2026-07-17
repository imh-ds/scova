"""Randomized SCOVA-CF population-counterfactual mean example."""

from scova import ContrastSpec
from scova.cf import SCOVACF, AnalysisMode, KnownAssignment, SCOVACFDeclaration
from scova.simulate import generate_data


def main() -> None:
    simulation = generate_data("randomized", n=600, seed=42)
    declaration = SCOVACFDeclaration(
        outcome="outcome",
        group="group",
        covariates=("x1", "x2", "x3"),
        mode=AnalysisMode.RANDOMIZED,
        scientific_question="What would the population mean be under each group?",
        eligibility="All generated study units",
        target_population="Eligible generated study-unit population",
        group_definitions=(
            ("g0", "randomized group zero"),
            ("g1", "randomized group one"),
            ("g2", "randomized group two"),
        ),
        outcome_time="end of follow-up",
        outcome_units="points",
        covariate_rationales=(
            ("x1", "baseline prognostic factor"),
            ("x2", "baseline prognostic factor"),
            ("x3", "baseline prognostic factor"),
        ),
        assignment=KnownAssignment(
            probabilities=(("g0", 1 / 3), ("g1", 1 / 3), ("g2", 1 / 3))
        ),
        contrasts=(
            ContrastSpec("g0 - g1", (("g0", 1.0), ("g1", -1.0))),
            ContrastSpec("g2 - g1", (("g2", 1.0), ("g1", -1.0))),
        ),
        n_splits=5,
        random_state=42,
    )
    analysis = SCOVACF().analyze(simulation.data, declaration)
    if not hasattr(analysis, "group_means"):
        print(analysis.to_dict())
        return
    print(analysis.evidence_card)
    print(analysis.contrasts["g0 - g1"].to_dict())
    print(analysis.infer(n_bootstrap=499).to_dict())


if __name__ == "__main__":
    main()
