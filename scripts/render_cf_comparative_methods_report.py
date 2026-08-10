"""Render a concise Markdown report for a comparative methods artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _table(summaries: dict[str, Any], *, estimand: str) -> list[str]:
    del estimand
    lines = [
        "| Method | Bias | RMSE | Coverage | Failure rate | Retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in sorted(summaries.items()):
        lines.append(
            "| {name} | {bias} | {rmse} | {coverage} | {failure} | {retention} |".format(
                name=name,
                bias=_format(summary["bias"]),
                rmse=_format(summary["rmse"]),
                coverage=_format(summary["coverage"]),
                failure=_format(summary["failure_rate"]),
                retention=_format(summary["treated_retained_fraction"]),
            )
        )
    return lines


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render(artifact: dict[str, Any]) -> str:
    """Render a methods-only report without qualification language."""
    if artifact.get("program_type") != "methods":
        raise ValueError("comparative report requires a methods artifact")
    lines = [
        "# SCOVA-CF two-group comparative methods study",
        "",
        artifact["interpretation"],
        "",
        f"- Protocol checksum: `{artifact['protocol_checksum']}`",
        f"- Dependency-lock checksum: `{artifact['dependency_lock_checksum']}`",
        f"- Frozen commit: `{artifact['git_commit']}`",
        f"- Execution completeness: `{artifact['complete']}`",
        "",
        "## Study-population ATE estimators",
        "",
        *_table(artifact["ate_summaries"], estimand="ate"),
        "",
        "## Matched-treated ATT estimator",
        "",
        (
            "PSM estimates the ATT among retained matched treated units; it is not ranked "
            "against the ATE estimators."
        ),
        "",
        *_table(artifact["att_summaries"], estimand="att"),
        "",
        "Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    args.output.write_text(render(artifact), encoding="utf-8")


if __name__ == "__main__":
    main()
