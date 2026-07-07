"""Record checksums for successfully built Stage-3 distributions."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def _record(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"build output is missing or empty: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "passed": True,
        "wheel": _record(args.wheel),
        "sdist": _record(args.sdist),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
