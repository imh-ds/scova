"""Require a complete full-campaign Stage 5B promotion-audit evidence package."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def blocking_reasons(root: Path) -> list[str]:
    specification = json.loads(
        (root / "benchmarks/specs/stage5b_promotion_audit.json").read_text(encoding="utf-8")
    )
    audit_path = root / "release/stage5b_promotion_audit.json"
    evidence_path = root / "release/artifacts/stage5b-promotion-evidence.json"
    if not audit_path.exists() or not evidence_path.exists():
        return ["Stage 5B promotion audit manifest or evidence is missing"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    claimed = evidence.pop("sha256", None)
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    if claimed != sha256(encoded).hexdigest():
        return ["Stage 5B promotion evidence checksum is invalid"]
    if audit.get("protocol") != specification["protocol"]:
        return ["promotion audit manifest protocol does not match specification"]
    if audit.get("public_verdict") != "experimental":
        return ["promotion audit must retain the experimental public verdict"]
    if evidence.get("audit_manifest_sha256") != sha256(audit_path.read_bytes()).hexdigest():
        return ["promotion evidence is not linked to the audit manifest"]
    if evidence.get("protocol") != specification["protocol"] or evidence.get("status") != "pass":
        return ["Stage 5B promotion evidence is not a passing matching artifact"]
    repetitions = evidence.get("metrics", {}).get("repetitions_per_arm")
    if repetitions != specification["repetitions_per_arm"]:
        return ["promotion evidence does not contain the frozen full campaign size"]
    criteria = evidence.get("criteria", {})
    failed = [name for name in specification["required_criteria"] if criteria.get(name) is not True]
    return [] if not failed else ["failed promotion criteria: " + ", ".join(failed)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    reasons = blocking_reasons(args.root)
    if reasons:
        raise SystemExit("Stage 5B promotion audit blocked: " + "; ".join(reasons))
    print("Stage 5B promotion audit evidence passes; public B2 verdict remains experimental")


if __name__ == "__main__":
    main()
