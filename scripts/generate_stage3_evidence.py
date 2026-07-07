"""Generate a checksummed Stage-3 directional evidence dossier."""

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
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--robustness-summary", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--packaged-thresholds", type=Path)
    parser.add_argument("--jax-minimum", type=Path)
    parser.add_argument("--jax-maximum", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = [
        ("theorem", Path("docs/theory/stage3_appendix.md")),
        ("notation_map", Path("docs/theory/stage3_notation_map.md")),
        ("frozen_protocol", Path("benchmarks/specs/stage3_release.json")),
        ("threshold_candidates", Path("benchmarks/specs/stage3_threshold_candidates.json")),
        ("validation_summary", args.validation_summary),
        ("robustness_summary", args.robustness_summary),
        ("locked_thresholds", args.thresholds),
        ("packaged_thresholds", args.packaged_thresholds),
        ("jax_minimum", args.jax_minimum),
        ("jax_maximum", args.jax_maximum),
        ("memory", args.memory),
        ("coverage", args.coverage),
        ("build", args.build),
    ]
    files = [_artifact_record(role, path) for role, path in required]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 2,
        "validation_level": "directional",
        "manifest": manifest,
        "artifacts": files,
        "complete": all(record["present"] for record in files),
        "known_limitations": [
            "directional engineering validation only; publication campaign pending",
            "finite declared grids only",
            "i.i.d. continuous outcomes only",
            "built-in smooth tilts only",
            "causal interpretation requires untestable exchangeability assumptions",
            "certified verdicts remain unavailable",
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    payload["report_sha256"] = sha256(encoded.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
