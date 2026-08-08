"""Fail closed when a qualification dispatch lacks resolved scope decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scova.cf import CFValidationProtocol, validate_manifest
from scripts.write_cf_scope_manifest import contract_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matrix-id", default="cf-observational-provisional-v1")
    args = parser.parse_args()
    validate_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        CFValidationProtocol.load(args.spec),
        contract_version=contract_version(args.contract),
        matrix_id=args.matrix_id,
    )


if __name__ == "__main__":
    main()
