"""Write a checksum-bound decision manifest for a qualification freeze."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scova.cf import (
    QUALIFICATION_REQUIRED_DECISION_IDS,
    CFValidationProtocol,
    build_qualification_manifest,
)


def contract_version(path: Path) -> str:
    match = re.search(r"^\*\*Version:\*\*\s*([^\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError("Methodological contract has no version")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matrix-id", default="cf-observational-provisional-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_qualification_manifest(
        CFValidationProtocol.load(args.spec),
        contract_version=contract_version(args.contract),
        matrix_id=args.matrix_id,
        required_decision_ids=QUALIFICATION_REQUIRED_DECISION_IDS,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
