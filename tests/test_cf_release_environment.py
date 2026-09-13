from __future__ import annotations

from pathlib import Path

from scova.cf import CFValidationProtocol
from scripts.check_cf_validation_environment import environment_reasons

SPEC = Path("benchmarks/specs/cf_reference_v9.json")


def test_environment_check_reports_pinned_identity_mismatches() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    installed = {name: value for name, value in protocol.software.items()}
    installed["scikit-learn"] = "1.9.0"

    reasons = environment_reasons(
        protocol,
        installed_versions=installed,
        lock_checksum="wrong-lock",
        source_checksums=dict(protocol.dataset_checksums or {}),
    )

    assert "scikit-learn==1.6.1 is required; found 1.9.0" in reasons
    assert "dependency lock checksum does not match the frozen protocol" in reasons


def test_environment_check_accepts_the_frozen_identity() -> None:
    protocol = CFValidationProtocol.load(SPEC)
    installed = dict(protocol.software)

    assert not environment_reasons(
        protocol,
        installed_versions=installed,
        lock_checksum=protocol.dependency_lock_checksum,
        source_checksums=dict(protocol.dataset_checksums or {}),
        python_version=protocol.software["python"],
    )
