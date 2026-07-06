"""SCOVA's fixed-target, multi-group AIPW core."""

from ._version import __version__
from .declaration import ContrastSpec, SCOVADeclaration
from .estimator import SCOVA, NuisancePredictions
from .result import ContrastEstimate, SCOVAResult, Verdict

__all__ = [
    "ContrastEstimate",
    "ContrastSpec",
    "NuisancePredictions",
    "SCOVA",
    "SCOVADeclaration",
    "SCOVAResult",
    "Verdict",
    "__version__",
]
