"""SCOVA's fixed-target, multi-group AIPW core."""

from ._version import __version__
from .anchor import (
    AnchoredBoundsResult,
    AnchoredContrastResult,
    LipschitzAnchorResult,
    LipschitzContrastResult,
)
from .declaration import (
    AnchoredBoundsDeclaration,
    ContrastSpec,
    DesignDeclaration,
    SCOVADeclaration,
    SupportGeometryDeclaration,
)
from .design import (
    DesignLock,
    OutcomeFreeDesignData,
    SCOVADesign,
    SCOVADesignResult,
    SCOVAGraphResult,
)
from .estimator import SCOVA, NuisancePredictions
from .graph import (
    ComparabilityGraphResult,
    PairwiseDiagnosticInput,
    PairwiseEdge,
    SubsetDiagnosticInput,
    SubsetHyperedge,
    build_comparability_graph,
    build_pairwise_comparability_graph,
)
from .inference import (
    GlobalTestResult,
    InferenceStatus,
    SimultaneousContrastResult,
    SimultaneousInferenceResult,
)
from .result import ContrastEstimate, SCOVAResult, Verdict

__all__ = [
    "ContrastEstimate",
    "AnchoredBoundsDeclaration",
    "AnchoredBoundsResult",
    "AnchoredContrastResult",
    "LipschitzAnchorResult",
    "LipschitzContrastResult",
    "ContrastSpec",
    "ComparabilityGraphResult",
    "DesignDeclaration",
    "DesignLock",
    "GlobalTestResult",
    "InferenceStatus",
    "NuisancePredictions",
    "OutcomeFreeDesignData",
    "PairwiseDiagnosticInput",
    "PairwiseEdge",
    "SCOVA",
    "SCOVADesign",
    "SCOVADesignResult",
    "SCOVAGraphResult",
    "SCOVADeclaration",
    "SCOVAResult",
    "SimultaneousContrastResult",
    "SimultaneousInferenceResult",
    "SubsetDiagnosticInput",
    "SubsetHyperedge",
    "SupportGeometryDeclaration",
    "Verdict",
    "__version__",
    "build_pairwise_comparability_graph",
    "build_comparability_graph",
]
