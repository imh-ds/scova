"""SCOVA's fixed-target, multi-group AIPW core."""

from ._version import __version__
from .declaration import ContrastSpec, DesignDeclaration, SCOVADeclaration
from .design import DesignLock, OutcomeFreeDesignData
from .graph import (
    ComparabilityGraphResult,
    PairwiseDiagnosticInput,
    PairwiseEdge,
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
    "SCOVADeclaration",
    "SCOVAResult",
    "SimultaneousContrastResult",
    "SimultaneousInferenceResult",
    "Verdict",
    "__version__",
    "build_pairwise_comparability_graph",
]
