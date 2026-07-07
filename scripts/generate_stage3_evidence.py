"""Generate a checksummed Stage-3 release-evidence dossier."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def _file_record(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "present": False, "sha256": None}
    return {
        "path": str(path),
        "present": True,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _artifact_record(role: str, path: Path | None) -> dict[str, object]:
    if path is None:
        return {"role": role, "path": None, "present": False, "sha256": None}
    return {"role": role, **_file_record(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("release/stage3_promotion.json"))
    parser.add_argument("--simulation-summary", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--jax-minimum", type=Path)
    parser.add_argument("--jax-latest", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = [
        ("theorem", Path("docs/theory/stage3_appendix.md")),
        ("notation_map", Path("docs/theory/stage3_notation_map.md")),
        ("review_1", Path("docs/reviews/review-1.md")),
        ("review_2", Path("docs/reviews/review-2.md")),
        ("simulation_summary", args.simulation_summary),
        ("locked_thresholds", args.thresholds),
        ("jax_minimum", args.jax_minimum),
        ("jax_latest", args.jax_latest),
        ("coverage", args.coverage),
    ]
    files = [_artifact_record(role, path) for role, path in required]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "manifest": manifest,
        "artifacts": files,
        "all_artifacts_present": all(record["present"] for record in files),
        "signature": _file_record(args.signature) if args.signature else None,
        "signature_status": "provided-unverified" if args.signature else "pending",
        "known_limitations": [
            "finite declared grids only",
            "i.i.d. continuous outcomes only",
            "built-in smooth tilts only",
            "causal interpretation requires untestable exchangeability assumptions",
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    payload["report_sha256"] = sha256(encoded.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
