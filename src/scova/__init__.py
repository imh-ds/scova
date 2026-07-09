"""SCOVA's fixed-target, multi-group AIPW core."""

from ._version import __version__
from .declaration import ContrastSpec, DesignDeclaration, SCOVADeclaration
from .design import DesignLock, OutcomeFreeDesignData, SCOVADesign, SCOVADesignResult, SCOVAGraphResult
from .graph import (
    ComparabilityGraphResult,
    PairwiseDiagnosticInput,
    PairwiseEdge,
    SubsetDiagnosticInput,
    SubsetHyperedge,
    build_comparability_graph,
    build_pairwise_comparability_graph,
)
from .estimator import SCOVA, NuisancePredictions
from .inference import (
    GlobalTestResult,
    InferenceStatus,
    SimultaneousContrastResult,
    SimultaneousInferenceResult,
)
from .result import ContrastEstimate, SCOVAResult, Verdict

__all__ = [
    "ContrastEstimate",
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
    "Verdict",
    "__version__",
    "build_pairwise_comparability_graph",
    "build_comparability_graph",
]
