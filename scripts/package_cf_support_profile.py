"""Package a promoted profile only after the complete evidence audit passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_cf_reference_release import blocking_reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reasons = blocking_reasons(args.evidence_root, args.spec)
    if reasons:
        raise SystemExit("SCOVA-CF profile packaging blocked:\n- " + "\n- ".join(reasons))
    profile = json.loads(
        (args.evidence_root / "cf-reference-support-profile.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {"schema_version": 1, "profiles": [profile]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
