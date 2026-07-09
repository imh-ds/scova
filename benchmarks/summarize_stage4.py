"""Verify Stage 4 shard provenance and write a non-promoting summary."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.shards:
        row = json.loads(path.read_text(encoding="utf-8"))
        claimed = row.pop("sha256", None)
        actual = sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if claimed != actual:
            raise SystemExit(f"invalid shard checksum: {path}")
        row["sha256"] = claimed
        rows.append(row)
    protocols = {row["protocol"] for row in rows}
    tiers = {row["tier"] for row in rows}
    if len(protocols) != 1 or len(tiers) != 1:
        raise SystemExit("shards must share one protocol and tier")
    summary = {
        "protocol": protocols.pop(),
        "tier": tiers.pop(),
        "shards": rows,
        "status": (
            "incomplete" if any(row["status"] != "completed" for row in rows) else "completed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
