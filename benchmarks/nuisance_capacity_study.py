"""Methods study: what makes SCOVA-CF's calibration bias gate breach?

Three v11 calibration cells failed the bias gate and all three turned out to
share one mechanism -- AIPW's bias is the PRODUCT of the two nuisance errors,
so it is first-order only when BOTH nuisances are misspecified at once. What
differs between them is which misspecification the fitted learner can absorb.

This runs in the pinned campaign environment on purpose. The question is about
what scikit-learn's estimators can represent, so the scikit-learn version is
part of what is being measured; a local run on a different stack is suggestive
rather than comparable with the frozen CF evidence.

Three designs, all scored through `_screening_cell_gate` on records built by
the same simulate/fit path a campaign shard uses. Nothing here reimplements
bias, coverage, or the gate.

* `misspecification` -- crosses propensity misspecification (confounding_form)
  against outcome misspecification (surface) against learner, everything else
  benign. Separates "nonlinear confounding" from "both nuisances wrong", which
  the campaign grid cannot do because the two only ever co-occur there.
* `cell2-ablation` -- starts from the failing adaptive cell verbatim and
  restores one factor at a time. Whichever restoration collapses the bias is
  the mechanism.
* `capacity` -- sweeps the adaptive outcome learner's capacity on that same
  cell. The flexible learner absorbs a smooth-nonlinear surface but not an
  interaction one; this asks whether that is a hyperparameter or structural.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor

from benchmarks.cf_reference_campaign import fit_campaign_record, simulate_reference_cell
from scova import SCOVA
from scova.cf import CFValidationProtocol
from scripts.calibrate_cf_support import _screening_cell_gate

SPEC = Path("benchmarks/specs/cf_reference_v11.json")

# Benign on every axis, so a variant differs from it only where stated.
BENIGN: dict[str, Any] = {
    "allocation": "balanced",
    "confounding": "moderate",
    "confounding_form": "linear",
    "effect": "partial-null",
    "learner": "linear",
    "n_covariates": 5,
    "n_groups": 2,
    "n_per_group": 150,
    "noise": "normal",
    "overlap": "full",
    "support": "strong",
    "surface": "linear",
}

# v11 calibration cell 2, verbatim: the adaptive cell that failed at 0.352.
CELL_TWO: dict[str, Any] = {
    **BENIGN,
    "confounding_form": "nonlinear",
    "effect": "heterogeneous",
    "learner": "adaptive",
    "n_per_group": 50,
    "noise": "heavy-tailed",
    "overlap": "poor",
    "surface": "interaction",
}


def _stable_seed(label: str, repetition: int) -> int:
    """Seed from a digest, not `hash()`.

    Built-in string hashing is randomized per process unless PYTHONHASHSEED is
    pinned, so seeding from it makes a study reproducible only by accident.
    """
    digest = sha256(f"{label}:{repetition}".encode()).digest()[:6]
    return int.from_bytes(digest, "big")


@contextlib.contextmanager
def _outcome_capacity(max_leaf_nodes: int | None) -> Iterator[None]:
    """Swap the adaptive outcome learner's capacity for the duration.

    The adaptive strategy picks the outcome model from a fixed candidate set by
    inner-fold squared error. Overriding the candidate rather than the selection
    keeps that choice intact and changes only what the flexible option can
    represent. The propensity is deliberately untouched: raising ITS capacity
    was measured making matters worse, and this is a different model.
    """
    if max_leaf_nodes is None:
        yield
        return
    original = SCOVA._adaptive_outcome_candidates

    @staticmethod  # type: ignore[misc]
    def patched() -> dict[str, Any]:
        candidates = original()
        candidates["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=1.0,
            random_state=0,
        )
        return candidates

    SCOVA._adaptive_outcome_candidates = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        SCOVA._adaptive_outcome_candidates = original  # type: ignore[method-assign]


Variant = tuple[str, dict[str, Any], "int | None"]


def variants(design: str, capacities: tuple[int, ...]) -> list[Variant]:
    """(label, cell, outcome_max_leaf_nodes) triples for a named design."""
    if design == "misspecification":
        rows = []
        for form in ("linear", "nonlinear"):
            for surface in ("linear", "smooth-nonlinear"):
                for learner in ("linear", "adaptive"):
                    rows.append(
                        (
                            f"form={form},surface={surface},learner={learner}",
                            {**BENIGN, "confounding_form": form, "surface": surface,
                             "learner": learner},
                            None,
                        )
                    )
        return rows
    if design == "cell2-ablation":
        restorations: list[tuple[str, dict[str, Any]]] = [
            ("cell2-as-is", {}),
            ("overlap->full", {"overlap": "full"}),
            ("n_per_group->150", {"n_per_group": 150}),
            ("noise->normal", {"noise": "normal"}),
            ("surface->linear", {"surface": "linear"}),
            ("confounding_form->linear", {"confounding_form": "linear"}),
            ("effect->partial-null", {"effect": "partial-null"}),
            ("overlap+n_per_group", {"overlap": "full", "n_per_group": 150}),
        ]
        return [(label, {**CELL_TWO, **override}, None) for label, override in restorations]
    if design == "capacity":
        return [
            (f"cell2,max_leaf_nodes={capacity}", dict(CELL_TWO), capacity)
            for capacity in capacities
        ]
    raise ValueError(f"unknown design: {design}")


def run_variant(
    label: str,
    cell: dict[str, Any],
    capacity: int | None,
    *,
    reps: int,
    metrics: Any,
) -> dict[str, Any]:
    records, refused = [], 0
    with _outcome_capacity(capacity):
        for repetition in range(reps):
            seed = _stable_seed(label, repetition)
            generated = simulate_reference_cell(cell, seed=seed)
            fitted = fit_campaign_record(generated, cell, include_stability=False, seed=seed)
            if fitted.get("refused"):
                refused += 1
            else:
                records.append(fitted)
    passed, audit = _screening_cell_gate(records, metrics)
    deviation = audit.get("empirical_standard_deviation")
    return {
        "label": label,
        "cell": cell,
        "outcome_max_leaf_nodes": capacity,
        "replications": reps,
        "refused": refused,
        "passed": passed,
        "bias_over_sd": (
            abs(audit["bias"]) / deviation if deviation else None
        ),
        "audit": audit,
    }


def _row(result: dict[str, Any], limit: float) -> str:
    ratio = result["bias_over_sd"]
    audit = result["audit"]
    return (
        f"{result['label']:34} "
        f"{'n/a' if ratio is None else format(ratio, '8.3f'):>8} "
        f"{audit.get('coverage', float('nan')):7.4f} "
        f"{audit.get('standard_error_ratio', float('nan')):9.3f} "
        f"{result['refused']:7d}  "
        f"{'PASS' if result['passed'] else 'FAIL'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design", choices=("misspecification", "cell2-ablation", "capacity"), required=True
    )
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--capacities", default="15,31,63,127")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--results-dir", default="study_cells")
    parser.add_argument("--out", default="nuisance-capacity-study.json")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    capacities = tuple(int(value) for value in args.capacities.split(",") if value)
    planned = variants(args.design, capacities)
    protocol = CFValidationProtocol.load(SPEC)
    metrics = protocol.calibration_gate_metrics
    limit = float(metrics["maximum_standardized_bias"])
    results_dir = Path(args.results_dir)

    header = (
        f"{'variant':34} {'bias/sd':>8} {'cov':>7} {'se_ratio':>9} {'refuse':>7}  verdict"
    )
    if args.merge:
        # Report the whole planned grid, so a shard that timed out or crashed
        # shows up as missing rather than silently narrowing the study.
        print(f"design {args.design}   bias limit {limit}")
        print(header)
        collected, missing = [], []
        for label, _cell, _capacity in planned:
            path = results_dir / (sha256(label.encode()).hexdigest()[:16] + ".json")
            if not path.exists():
                missing.append(label)
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            collected.append(result)
            print(_row(result, limit))
        for label in missing:
            print(f"{label:34} {'MISSING':>8}")
        summary = {
            "design": args.design,
            "bias_limit": limit,
            "protocol_checksum": protocol.checksum,
            "results": collected,
            "missing": missing,
        }
        Path(args.out).write_text(
            json.dumps(summary, indent=1, sort_keys=True, default=float), encoding="utf-8"
        )
        if missing:
            raise SystemExit(f"{len(missing)} variant(s) missing from the study grid")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"design {args.design}   reps {args.reps}   bias limit {limit}")
    print(header)
    for index, (label, cell, capacity) in enumerate(planned):
        if index % args.shard_count != args.shard_index:
            continue
        result = run_variant(label, cell, capacity, reps=args.reps, metrics=metrics)
        print(_row(result, limit))
        path = results_dir / (sha256(label.encode()).hexdigest()[:16] + ".json")
        path.write_text(
            json.dumps(result, indent=1, sort_keys=True, default=float), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
