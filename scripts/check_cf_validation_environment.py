"""Fail fast when a SCOVA-CF validation runner is not the frozen environment."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmarks.cf_reference_campaign import dependency_lock_checksum, plasmode_source_checksum
from scova.cf import CFValidationProtocol


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def environment_reasons(
    protocol: CFValidationProtocol,
    *,
    installed_versions: Mapping[str, str] | None = None,
    lock_checksum: str | None = None,
    source_checksums: Mapping[str, str] | None = None,
    python_version: str | None = None,
) -> list[str]:
    """Return every mismatch between the runner and a frozen protocol.

    The optional arguments make the comparison deterministic and independently
    testable. With no overrides, values are read from the current runner.
    """
    expected_versions = dict(protocol.software)
    if installed_versions is None:
        actual_versions = {
            name: _installed_version(name)
            for name in expected_versions
            if name != "python"
        }
    else:
        actual_versions = dict(installed_versions)
    actual_python = platform.python_version() if python_version is None else python_version
    reasons: list[str] = []
    expected_python = expected_versions.get("python")
    if expected_python is not None and actual_python != expected_python:
        reasons.append(f"python=={expected_python} is required; found {actual_python}")
    for name, expected in expected_versions.items():
        if name == "python":
            continue
        found = actual_versions.get(name, "not-installed")
        if found != expected:
            reasons.append(f"{name}=={expected} is required; found {found}")

    actual_lock = lock_checksum if lock_checksum is not None else dependency_lock_checksum()
    if actual_lock != protocol.dependency_lock_checksum:
        reasons.append("dependency lock checksum does not match the frozen protocol")

    expected_sources = dict(protocol.dataset_checksums or {})
    if source_checksums is None:
        actual_sources = {}
        for name in expected_sources:
            try:
                actual_sources[name] = plasmode_source_checksum(name)
            except Exception:  # pragma: no cover - defensive runner diagnostic
                actual_sources[name] = "not-available"
    else:
        actual_sources = dict(source_checksums)
    for name, expected in expected_sources.items():
        found = actual_sources.get(name, "not-available")
        if found != expected:
            reasons.append(f"{name} source checksum does not match the frozen protocol")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check the exact interpreter, dependencies, lock, and data sources "
            "for SCOVA-CF."
        )
    )
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    reasons = environment_reasons(CFValidationProtocol.load(args.spec))
    if reasons:
        raise SystemExit("SCOVA-CF validation environment check failed:\n- " + "\n- ".join(reasons))


if __name__ == "__main__":
    main()
