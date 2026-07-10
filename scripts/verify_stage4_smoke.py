"""Fail the engineering-smoke workflow unless every fixed smoke item exercised inference."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.stage4_campaign import frozen_cells, load_specification, shard_items  # noqa: E402


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def verify(paths: list[Path], specification: dict[str, Any]) -> dict[str, int]:
    tier = "engineering_smoke"
    cells = frozen_cells(tier, specification)
    expected = {item for shard in range(4) for item in shard_items(cells, 5, shard, 4)}
    observed: set[tuple[int, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("sha256", None)
        if claimed != _digest(payload):
            raise ValueError(f"invalid smoke shard checksum: {path}")
        if payload.get("tier") != tier or payload.get("shard_count") != 4:
            raise ValueError("smoke artifact has an unexpected tier or shard count")
        for record in payload["records"]:
            key = (int(record["cell_index"]), int(record["repetition"]))
            scenario = cells[key[0]].scenario
            if key not in expected or key in observed:
                raise ValueError("smoke artifact has an unexpected or duplicate work item")
            if record.get("status") != "completed":
                raise ValueError(f"smoke work item {key} did not complete")
            if scenario == "rare_group":
                if (
                    record.get("accepted")
                    or record.get("expected_refusal") != "insufficient_per_split_group_count"
                ):
                    raise ValueError(f"smoke rare-group work item {key} did not refuse safely")
            else:
                if not record.get("accepted"):
                    raise ValueError(f"smoke work item {key} did not exercise held-out inference")
                if not record.get("post_lock_mutation_rejected"):
                    raise ValueError(f"smoke work item {key} accepted post-lock mutation")
            observed.add(key)
    if observed != expected:
        raise ValueError(
            f"incomplete smoke evidence: expected {len(expected)}, found {len(observed)}"
        )
    return {"records": len(observed), "cells": len(cells)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("benchmarks/specs/stage4_graph_release.json"),
    )
    args = parser.parse_args()
    result = verify(args.inputs, load_specification(args.spec))
    print(f"Verified {result['records']} Stage 4 smoke records across {result['cells']} cells")


if __name__ == "__main__":
    main()
