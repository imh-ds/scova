"""SCOVA's fixed-target, multi-group AIPW core."""

from ._version import __version__
from .declaration import ContrastSpec, SCOVADeclaration
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
    "GlobalTestResult",
    "InferenceStatus",
    "NuisancePredictions",
    "SCOVA",
    "SCOVADeclaration",
    "SCOVAResult",
    "SimultaneousContrastResult",
    "SimultaneousInferenceResult",
    "Verdict",
    "__version__",
]
