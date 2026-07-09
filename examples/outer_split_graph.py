"""Stage 4 outcome-blind design followed by graph-conditional inference."""

from scova import DesignDeclaration, OutcomeFreeDesignData, SCOVADesign
from scova.simulate import generate_data

simulation = generate_data("observational", n=1_000, seed=42)
data = simulation.data
design_data = OutcomeFreeDesignData.from_arrays(
    data.loc[:, ["x1", "x2", "x3"]].to_numpy(),
    data["group"].tolist(),
    row_ids=list(range(len(data))),
)
declaration = DesignDeclaration(
    group="group",
    covariates=("x1", "x2", "x3"),
    random_state=42,
    candidate_subsets=(("g0", "g1", "g2"),),
)

engine = SCOVADesign()
locked = engine.prepare_design(design_data, declaration)
estimation_ids = locked.lock.estimation_row_ids
outcomes = [data.loc[row_id, "outcome"] for row_id in estimation_ids]
result = engine.analyze_outcomes(locked, outcomes, row_ids=estimation_ids, n_bootstrap=499)

print(locked.design_report())
print(result.report())
