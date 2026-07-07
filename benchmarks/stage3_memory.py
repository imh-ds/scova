"""Measure peak allocations of the batched Stage-3 multiplier process."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from hashlib import sha256
from pathlib import Path

from scova import SCOVADeclaration
from scova.experimental import (
    PathDeclaration,
    StabilizationSpec,
    fit_path,
    generate_stabilization_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--bootstrap", type=int, default=1_999)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = StabilizationSpec(
        n_groups=8,
        n=args.n,
        p=20,
        overlap="strong",
        outcome="nonlinear",
        imbalance="balanced",
        nuisance="oracle",
    )
    generated = generate_stabilization_data(specification, seed=880_301)
    declaration = PathDeclaration(
        SCOVADeclaration(
            "outcome",
            "group",
            tuple(f"x{index}" for index in range(1, 21)),
            n_splits=5,
            random_state=880_301,
        )
    )
    # Nuisances are supplied once. The persisted result's infer method has no learner handle.
    result = fit_path(
        generated.data,
        declaration,
        nuisance_predictions=generated.nuisance_predictions("oracle"),
    )
    tracemalloc.start()
    started = time.perf_counter()
    benchmark_family = (next(iter(result.contrasts)),)
    inference = result.infer(
        benchmark_family,
        n_bootstrap=args.bootstrap,
        random_state=880_302,
        batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    columns = len(inference.family) * len(inference.lambdas)
    unbatched_multiplier_bytes = args.bootstrap * args.n * 8
    payload = {
        "schema_version": 1,
        "n": args.n,
        "groups": 8,
        "grid_points": len(inference.lambdas),
        "contrasts": len(inference.family),
        "family": list(inference.family),
        "columns": columns,
        "bootstrap": args.bootstrap,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "traced_peak_bytes": peak_bytes,
        "unbatched_multiplier_bytes": unbatched_multiplier_bytes,
        "peak_below_unbatched_multiplier_cube": peak_bytes < unbatched_multiplier_bytes,
        "nuisance_refit_possible_from_result": False,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = sha256(encoded).hexdigest()
    if not payload["peak_below_unbatched_multiplier_cube"]:
        raise RuntimeError("batched inference exceeded the unbatched multiplier allocation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
