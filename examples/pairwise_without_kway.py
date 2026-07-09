"""Inspect a graph where pairwise support need not justify a K-way claim."""

from scova import DesignDeclaration, OutcomeFreeDesignData, SCOVADesign
from scova.experimental import StabilizationSpec, generate_stabilization_data


generated = generate_stabilization_data(
    StabilizationSpec(n_groups=4, n=2_000, p=5, overlap="pairwise_only", outcome="linear", imbalance="balanced"),
    seed=17,
)
data = generated.data
design_data = OutcomeFreeDesignData.from_arrays(
    data.loc[:, ["x1", "x2", "x3", "x4", "x5"]].to_numpy(), data["group"].tolist()
)
declaration = DesignDeclaration(
    group="group",
    covariates=("x1", "x2", "x3", "x4", "x5"),
    random_state=17,
    candidate_subsets=(tuple(generated.group_labels),),
)

locked = SCOVADesign().prepare_design(design_data, declaration)
print("pairwise edges:", locked.graph.supported_edges)
print("supported K-way hyperedges:", [item.groups for item in locked.graph.supported_maximal_hyperedges])
