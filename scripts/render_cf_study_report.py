"""Render concise, provenance-first reports for the separated study programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scova.cf import StudyProgram, assert_program_artifact


def render(artifact: dict[str, Any]) -> str:
    program = StudyProgram(artifact["program_type"])
    assert_program_artifact(artifact, program)
    lines = [
        f"# SCOVA-CF {program.value} report",
        "",
        f"- Design checksum: `{artifact['design_checksum']}`",
        f"- Dependency lock: `{artifact['dependency_lock_checksum']}`",
        f"- Frozen commit: `{artifact.get('git_commit', 'not-recorded')}`",
        f"- Planned replications: {artifact.get('planned_replications', 'not-recorded')}",
        f"- Completed replications: {artifact.get('completed_replications', 'not-recorded')}",
        f"- Source evidence IDs: {', '.join(artifact.get('source_evidence_ids', [])) or 'none'}",
        *(
            []
            if artifact.get("decision_manifest_checksum") is None
            else [f"- Scope-decision manifest: `{artifact['decision_manifest_checksum']}`"]
        ),
        "",
    ]
    if program is StudyProgram.QUALIFICATION:
        lines.extend(
            [
                "## Interpretation",
                "",
                "This prospective evidence may create a candidate support profile only. "
                "It cannot promote a profile without independent held-out validation and "
                "recorded human owner and independent-reviewer approval. A resolved decision "
                "record documents governance; it does not prove exchangeability, positivity, "
                "or causal validity in an applied dataset.",
            ]
        )
        if artifact.get("verification_lane"):
            lines.extend(
                [
                    "",
                    "## Verification-lane limits",
                    "",
                    f"- Lane: `{artifact['verification_lane']}`",
                    f"- Permitted claim: {artifact.get('permitted_claim', 'not recorded')}",
                    f"- Prohibited claim: {artifact.get('prohibited_claim', 'not recorded')}",
                ]
            )
    else:
        lines.extend(
            [
                "## Interpretation",
                "",
                "These are continuous simulation summaries with Monte-Carlo uncertainty "
                "inside their declared DGPs. They do not create a support profile or a "
                "qualification claim.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(artifact), encoding="utf-8")


if __name__ == "__main__":
    main()
