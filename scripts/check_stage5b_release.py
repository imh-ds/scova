"""Block Stage 5B promotion until frozen experimental evidence passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def blocking_reasons(root: Path) -> list[str]:
    specification = root / "benchmarks/specs/stage5b_lipschitz_anchor.json"
    spec = json.loads(specification.read_text(encoding="utf-8"))
    evidence_path = root / "release/artifacts/stage5b-lipschitz-anchor-evidence.json"
    if not evidence_path.exists():
        return ["Stage 5B Lipschitz evidence artifact is missing"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("protocol") != spec["protocol"]:
        return ["evidence protocol does not match frozen Stage 5B specification"]
    if evidence.get("status") != "pass":
        return ["Stage 5B evidence status is not pass"]
    criteria = evidence.get("criteria", {})
    failed = [name for name in spec["required_criteria"] if criteria.get(name) is not True]
    return [] if not failed else ["failed criteria: " + ", ".join(failed)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    reasons = blocking_reasons(args.root)
    if reasons:
        raise SystemExit("Stage 5B promotion blocked: " + "; ".join(reasons))
    print("Stage 5B Lipschitz evidence passes the frozen experimental gate")


if __name__ == "__main__":
    main()
