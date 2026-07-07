"""Enforce branch-aware coverage floors for Stage-3 critical modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CRITICAL_SUFFIXES = (
    "scova/experimental/path.py",
    "scova/experimental/gates.py",
    "scova/inference.py",
    "scova/result.py",
)


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _covered_fraction(summary: dict[str, int | float]) -> float:
    statements = int(summary["num_statements"])
    branches = int(summary.get("num_branches", 0))
    covered = int(summary["covered_lines"]) + int(summary.get("covered_branches", 0))
    denominator = statements + branches
    return 1.0 if denominator == 0 else covered / denominator


def coverage_failures(
    report: dict,
    *,
    critical_floor: float = 0.95,
    package_floor: float = 0.92,
) -> list[str]:
    failures: list[str] = []
    normalized = {_normalize(path): payload for path, payload in report["files"].items()}
    for suffix in CRITICAL_SUFFIXES:
        matches = [payload for path, payload in normalized.items() if path.endswith(suffix)]
        if len(matches) != 1:
            failures.append(f"expected exactly one coverage entry ending in {suffix!r}")
            continue
        fraction = _covered_fraction(matches[0]["summary"])
        if fraction < critical_floor:
            failures.append(f"{suffix}: {fraction:.3%} is below {critical_floor:.1%}")
    package_fraction = _covered_fraction(report["totals"])
    if package_fraction < package_floor:
        failures.append(f"package: {package_fraction:.3%} is below {package_floor:.1%}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = coverage_failures(report)
    if failures:
        raise SystemExit("Coverage gate failed:\n- " + "\n- ".join(failures))
    print("Stage-3 critical and package coverage floors passed.")


if __name__ == "__main__":
    main()
