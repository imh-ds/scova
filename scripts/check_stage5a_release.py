"""Validate a versioned Stage 5A bounded-anchor evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def blocking_reasons(root: Path) -> list[str]:
    specification = root / "benchmarks/specs/stage5a_bounded_anchor.json"
    spec = json.loads(specification.read_text(encoding="utf-8"))
    evidence_path = root / "release/artifacts/stage5a-bounded-anchor-evidence.json"
    if not evidence_path.exists():
        return ["Stage 5A bounded-anchor evidence artifact is missing"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("protocol") != spec["protocol"]:
        return ["evidence protocol does not match frozen Stage 5A specification"]
    if evidence.get("status") != "pass":
        return ["Stage 5A evidence status is not pass"]
    criteria = evidence.get("criteria", {})
    failed = [name for name in spec["required_criteria"] if criteria.get(name) is not True]
    return [] if not failed else ["failed criteria: " + ", ".join(failed)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    reasons = blocking_reasons(args.root)
    if reasons:
        raise SystemExit("Stage 5A promotion blocked: " + "; ".join(reasons))
    print("Stage 5A bounded-anchor evidence passes the frozen gate")


if __name__ == "__main__":
    main()
