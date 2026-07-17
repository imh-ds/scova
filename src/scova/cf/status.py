"""Typed analysis statuses for SCOVA-CF."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .declaration import AnalysisMode, ClaimClass


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSTABLE = "unstable"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class SCOVACFStatus:
    support: SupportStatus
    code: str
    reason: str
    confirmatory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "support": self.support.value,
            "code": self.code,
            "reason": self.reason,
            "confirmatory": self.confirmatory,
        }


@dataclass(frozen=True, slots=True)
class SCOVACFRefusal:
    """Machine-readable refusal returned instead of a best-effort CF number."""

    declaration_hash: str
    mode: AnalysisMode
    claim_class: ClaimClass
    status: SCOVACFStatus
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "scova-cf-refusal",
            "declaration_hash": self.declaration_hash,
            "mode": self.mode.value,
            "claim_class": self.claim_class.value,
            "status": self.status.to_dict(),
            "details": self.details,
        }
