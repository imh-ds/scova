"""Reject Stage 4 promotion unless complete, passing frozen evidence exists."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = ROOT / "benchmarks" / "specs" / "stage4_graph_release.json"
EVIDENCE = ROOT / "release" / "artifacts" / "stage4-evidence.json"
THRESHOLDS = ROOT / "release" / "artifacts" / "stage3-directional-thresholds.json"


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not THRESHOLDS.exists():
        raise SystemExit("Stage 4 promotion blocked: locked Stage 3 threshold artifact is missing")
    if not EVIDENCE.exists():
        raise SystemExit("Stage 4 promotion blocked: Stage 4 evidence artifact is missing")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence.get("protocol") != spec["protocol"]:
        raise SystemExit("Stage 4 promotion blocked: evidence protocol does not match frozen spec")
    if evidence.get("status") != "pass":
        raise SystemExit("Stage 4 promotion blocked: evidence status is not pass")
    criteria = spec["directional_pass_criteria"]
    failed = [name for name in criteria if evidence.get("criteria", {}).get(name) is not True]
    if failed:
        raise SystemExit("Stage 4 promotion blocked: failed criteria " + ", ".join(failed))
    print("Stage 4 promotion evidence passes the frozen directional gate")


if __name__ == "__main__":
    main()
