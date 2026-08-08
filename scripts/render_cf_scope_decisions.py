"""Render the authoritative scope-decision registry for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scova.cf import validate_registry


def render(values: dict[str, Any]) -> str:
    decisions = validate_registry(values)
    lines = ["# SCOVA-CF scope-decision log", ""]
    for decision in decisions.values():
        record = decision.payload
        lines.extend(
            [
                f"## {decision.decision_id}",
                "",
                f"- Status: `{decision.status}`",
                f"- Path: `{decision.path}`",
                f"- Evidence: {', '.join(record['evidence_ids'])}",
                f"- Affected protocols: {', '.join(record['affected_protocols'])}",
                f"- Uncertainty: {record['uncertainty']}",
                f"- Rationale: {record['rationale']}",
                f"- Prior evidence: {record['prior_evidence_consequences']}",
                f"- Record checksum: `{decision.checksum}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Review rule",
            "",
            "A resolved decision records an approved governance choice. It does not establish "
            "exchangeability, positivity, or causal validity in an applied dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.registry.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(values), encoding="utf-8")


if __name__ == "__main__":
    main()
