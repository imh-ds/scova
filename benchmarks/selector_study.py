"""Methods study: how should SCOVA-CF pick its propensity learner?

The rule this study evaluated picked between logistic regression and gradient
boosting by inner-fold log loss.  Log loss scores
*prediction* of group membership, but the propensity's job in AIPW is
*confounding control*, so under weak nonlinear confounding the misspecified
linear learner wins the contest and bias leaks through with no warning.

The estimator now fixes the flexible learner instead; this harness is what
settled that, and reproduces the old rule as its `adaptive` arm.

It compares candidate rules on identical data.  Both
propensity candidates and the outcome regression are cross-fitted ONCE per
replicate; every rule then reuses those fits and is evaluated through the real
estimator via ``SCOVACFNuisancePredictions``, so differences between rules are
attributable to the selection rule alone and not to refitting noise.

Run one shard:
    python -m benchmarks.selector_study --shard-index 0 --shard-count 16
Merge shards into one summary:
    python -m benchmarks.selector_study --merge --results-dir cells --out study.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

from scova import ContrastSpec
from scova.cf import (
    SCOVACF,
    AnalysisMode,
    EstimatedAssignment,
    SCOVACFDeclaration,
    SCOVACFRefusal,
)
from scova.cf.estimator import SCOVACFNuisancePredictions
from scova.estimator import SCOVA

P = 5
LOGISTIC = "LogisticRegression"
BOOSTING = "HistGradientBoostingClassifier"

# Rules under test.  "adaptive" is the incumbent; "linear"/"flex" are the two
# degenerate bounds; "margin@d" only accepts the linear learner when it wins by
# a relative margin d; "ensemble" averages the two propensity surfaces.
RULES = ("adaptive", "linear", "flex", "margin@0.02", "margin@0.05", "ensemble")


def _basis(x: np.ndarray, form: str) -> np.ndarray:
    """Shared confounding/prognostic signal f(X)."""
    if form == "linear":
        return 0.9 * x[:, 0] - 0.5 * x[:, 1] + 0.3 * x[:, 2]
    return np.sin(1.5 * x[:, 0]) + 0.6 * x[:, 1] ** 2 - 0.4 * np.abs(x[:, 2])


def simulate(*, n_per_group, k, gamma, form, hetero, seed, floor=0.10):
    """Confounded assignment: P(group | X) depends on f(X) with strength gamma."""
    rng = np.random.default_rng(seed)
    n = n_per_group * k
    x = rng.normal(size=(n, P))
    f = _basis(x, form)
    f = (f - f.mean()) / f.std()
    loadings = np.linspace(-1.0, 1.0, k)
    logits = gamma * np.outer(f, loadings)
    logits -= logits.max(axis=1, keepdims=True)
    prop = np.exp(logits)
    prop /= prop.sum(axis=1, keepdims=True)
    if floor > 0:  # enforce positivity, then renormalise
        prop = np.clip(prop, floor, None)
        prop /= prop.sum(axis=1, keepdims=True)
    codes = np.array([rng.choice(k, p=prop[i]) for i in range(n)])
    effects = np.linspace(0.0, 0.5, k)
    mu = np.empty((n, k))
    for g in range(k):
        het = (0.5 * f) if (hetero and g > 0) else 0.0
        mu[:, g] = 1.0 * f + effects[g] + het
    outcome = mu[np.arange(n), codes] + rng.normal(size=n)
    data = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(P)])
    data["group"] = [f"g{c}" for c in codes]
    data["outcome"] = outcome
    return data, codes, x, outcome, mu.mean(axis=0)


def declaration(data, k, strategy="adaptive"):
    covs = tuple(c for c in data.columns if c.startswith("x"))
    labels = tuple(f"g{i}" for i in range(k))
    return SCOVACFDeclaration(
        outcome="outcome",
        group="group",
        covariates=covs,
        mode=AnalysisMode.OBSERVATIONAL_CAUSAL,
        scientific_question="Propensity-selector methods study",
        eligibility="All simulated units",
        target_population="Simulated study population",
        group_definitions=tuple((lbl, f"Naturally occurring group {lbl}") for lbl in labels),
        outcome_time="simulated",
        outcome_units="points",
        covariate_rationales=tuple((c, "Baseline confounder") for c in covs),
        assignment=EstimatedAssignment(nuisance_strategy=strategy),
        outcome_nuisance_strategy=strategy,
        n_splits=3,
        random_state=17,
        contrasts=(
            ContrastSpec(f"{labels[1]} - {labels[0]}", ((labels[1], 1.0), (labels[0], -1.0))),
        ),
    )


def incumbent_scores(x, group_codes, candidates, n_splits=3):
    """Inner-fold log loss per candidate: the rule the estimator used to apply.

    Kept here rather than in ``scova`` because the estimator no longer selects
    the propensity from the data.  Reproducing the historical rule is what lets
    the "adaptive" arm below stay a like-for-like comparison.
    """
    counts = np.bincount(group_codes)
    usable = min(n_splits, int(np.min(counts)))
    if usable < 2:
        return dict.fromkeys(candidates, float("inf"))
    folds = np.empty(len(group_codes), dtype=int)
    for code in np.unique(group_codes):
        indices = np.flatnonzero(group_codes == code)
        folds[indices] = np.arange(len(indices)) % usable
    scores = {}
    for name, candidate in candidates.items():
        predicted = np.empty((len(x), len(np.unique(group_codes))))
        for fold in np.unique(folds):
            train = folds != fold
            test = ~train
            model = clone(candidate)
            model.fit(x[train], group_codes[train])
            probability = np.asarray(model.predict_proba(x[test]), dtype=float)
            for column, code in enumerate(np.asarray(model.classes_, dtype=int)):
                predicted[test, code] = probability[:, column]
        scores[name] = float(log_loss(group_codes, predicted, labels=np.arange(predicted.shape[1])))
    return scores


def shared_nuisance_fits(x, codes, outcome, folds, k, labels):
    """Cross-fit both propensity candidates and the outcome regression once.

    Mirrors ``SCOVA._cross_fit`` exactly, but keeps every candidate's predictions
    instead of collapsing to the selected one, and records the inner-fold log
    loss that the incumbent rule selects on.
    """
    n = len(outcome)
    per_candidate = {name: np.empty((n, k)) for name in (LOGISTIC, BOOSTING)}
    outcome_regression = np.empty((n, k))
    fold_scores: list[dict[str, float]] = []
    candidates = SCOVA._adaptive_propensity_candidates()
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        # Inner-fold scores drive the incumbent rule; compute them once per fold.
        scores = incumbent_scores(x[train], codes[train], candidates)
        fold_scores.append({"fold": int(fold), **scores})
        for name, candidate in candidates.items():
            model = clone(candidate)
            model.fit(x[train], codes[train])
            raw = np.asarray(model.predict_proba(x[test]), dtype=float)
            classes = np.asarray(model.classes_, dtype=int)
            if set(classes.tolist()) != set(range(k)):
                raise ValueError("Every propensity training fold must contain every group")
            aligned = np.empty((int(test.sum()), k))
            for column, code in enumerate(classes):
                aligned[:, code] = raw[:, column]
            per_candidate[name][test] = aligned
        for code in range(k):
            group_train = train & (codes == code)
            model, _, _ = SCOVA._select_outcome_model(x[group_train], outcome[group_train])
            model.fit(x[group_train], outcome[group_train])
            outcome_regression[test, code] = np.asarray(model.predict(x[test]), dtype=float)
    return per_candidate, outcome_regression, fold_scores


def propensity_for_rule(rule, per_candidate, fold_scores, folds):
    """Assemble a cross-fitted propensity matrix under one selection rule."""
    if rule == "linear":
        return per_candidate[LOGISTIC], {LOGISTIC: len(fold_scores)}
    if rule == "flex":
        return per_candidate[BOOSTING], {BOOSTING: len(fold_scores)}
    if rule == "ensemble":
        blended = 0.5 * per_candidate[LOGISTIC] + 0.5 * per_candidate[BOOSTING]
        return blended / blended.sum(axis=1, keepdims=True), {"ensemble": len(fold_scores)}
    margin = 0.0 if rule == "adaptive" else float(rule.split("@")[1])
    matrix = np.empty_like(per_candidate[LOGISTIC])
    picks: dict[str, int] = {}
    for entry in fold_scores:
        fold = entry["fold"]
        # Accept the linear learner only when it beats boosting by the margin.
        chosen = LOGISTIC if entry[LOGISTIC] <= entry[BOOSTING] * (1.0 - margin) else BOOSTING
        picks[chosen] = picks.get(chosen, 0) + 1
        test = folds == fold
        matrix[test] = per_candidate[chosen][test]
    return matrix, picks


def contrast_metrics(result, truth):
    w = np.zeros(len(truth))
    w[1], w[0] = 1.0, -1.0
    means = np.asarray(result.group_means, dtype=float)
    est = float(w @ means)
    se = float(np.sqrt(max(w @ np.asarray(result.covariance, dtype=float) @ w, 0.0)))
    true = float(truth[1] - truth[0])
    return {
        "err": est - true,
        "se": se,
        "covered": bool(abs(est - true) <= 1.96 * se),
    }


def run_replicate(args):
    """One replicate of one design cell, evaluated under every rule."""
    cell, rep = args
    k = cell["k"]
    data, codes, x, outcome, truth = simulate(
        n_per_group=cell["n_per_group"],
        k=k,
        gamma=cell["gamma"],
        form=cell["form"],
        hetero=(cell["form"] == "nonlinear"),
        seed=900_000 + 7919 * rep,
    )
    decl = declaration(data, k)
    labels = tuple(f"g{i}" for i in range(k))
    folds, _ = SCOVACF._design_folds(data, decl, codes)
    try:
        per_candidate, outcome_regression, fold_scores = shared_nuisance_fits(
            x, codes, outcome, folds, k, labels
        )
    except ValueError:
        return cell["id"], {}
    out: dict[str, dict] = {}
    for rule in RULES:
        propensity, picks = propensity_for_rule(rule, per_candidate, fold_scores, folds)
        result = SCOVACF().analyze(
            data,
            decl,
            nuisance_predictions=SCOVACFNuisancePredictions(
                outcome_regression=outcome_regression,
                group_labels=labels,
                propensity=propensity,
            ),
        )
        if isinstance(result, SCOVACFRefusal):
            out[rule] = {"refused": True, "picks": picks}
            continue
        record = contrast_metrics(result, truth)
        record["refused"] = False
        record["picks"] = picks
        out[rule] = record
    return cell["id"], out


def summarise(records):
    """Collapse replicate records into per-rule performance for one cell."""
    summary = {}
    for rule in RULES:
        rows = [r[rule] for r in records if rule in r]
        kept = [r for r in rows if not r["refused"]]
        if not kept:
            summary[rule] = {"n": 0, "refused": len(rows)}
            continue
        err = np.array([r["err"] for r in kept])
        flex = sum(r["picks"].get(BOOSTING, 0) for r in rows)
        total = sum(sum(r["picks"].values()) for r in rows)
        summary[rule] = {
            "n": len(kept),
            "refused": len(rows) - len(kept),
            "bias": float(err.mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "coverage": float(np.mean([r["covered"] for r in kept])),
            "mean_se": float(np.mean([r["se"] for r in kept])),
            "flex_share": float(flex / total) if total else float("nan"),
        }
    return summary


def build_cells(n_per_groups, gammas, forms, k_values):
    cells = []
    for form in forms:
        for gamma in gammas:
            for n in n_per_groups:
                for k in k_values:
                    cells.append(
                        {
                            "id": f"{form}|g{gamma}|n{n}|k{k}",
                            "form": form,
                            "gamma": gamma,
                            "n_per_group": n,
                            "k": k,
                        }
                    )
    return cells


def estimated_cost(cell) -> float:
    """Rough relative runtime of a cell, for balancing shards only.

    Calibrated against observed timings: cost grows about linearly in n and in
    the number of groups (each adds an outcome-model fit per fold), and strong
    confounding makes the boosted fits work harder.
    """
    return cell["n_per_group"] * cell["k"] * (1.0 + cell["gamma"])


def shard_cells(cells, shard_index, shard_count):
    """Assign cells to shards by longest-processing-time-first bin packing.

    Cost spans a ~50x range across the grid, so wall time is set by the
    unluckiest shard. A plain stride is worse than it looks: it aliases with
    the regular structure of the grid (with an even shard count every shard
    lands on one form), so pack by cost instead.
    """
    if shard_count <= 1:
        return cells
    loads = [0.0] * shard_count
    assigned: list[list] = [[] for _ in range(shard_count)]
    for cell in sorted(cells, key=estimated_cost, reverse=True):
        lightest = min(range(shard_count), key=lambda s: (loads[s], s))
        assigned[lightest].append(cell)
        loads[lightest] += estimated_cost(cell)
    return assigned[shard_index]


def checkpoint_path(results_dir: Path, cell_id: str) -> Path:
    """One file per cell so an interrupted run only loses the cell in flight."""
    return results_dir / (cell_id.replace("|", "__").replace(".", "p") + ".json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--n-per-group", default="100,200,400,800")
    parser.add_argument("--gammas", default="0.0,0.25,0.5,1.0")
    parser.add_argument("--k-values", default="2")
    parser.add_argument("--forms", default="linear,nonlinear")
    parser.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    parser.add_argument("--out", default="selector_grid_results.json")
    parser.add_argument("--results-dir", default="grid_cells")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="skip simulation; merge existing per-cell files into --out",
    )
    parser.add_argument("--quick", action="store_true", help="tiny smoke grid")
    args = parser.parse_args()

    if args.quick:
        args.reps, args.n_per_group, args.gammas = 24, "200", "0.5"
    n_per_groups = [int(v) for v in args.n_per_group.split(",")]
    gammas = [float(v) for v in args.gammas.split(",")]
    k_values = [int(v) for v in args.k_values.split(",")]
    forms = tuple(args.forms.split(","))
    args.workers = args.workers or os.cpu_count() or 4
    cells = build_cells(n_per_groups, gammas, forms, k_values)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge(cells, results_dir, Path(args.out))
        return

    mine = shard_cells(cells, args.shard_index, args.shard_count)
    pending = [c for c in mine if not checkpoint_path(results_dir, c["id"]).exists()]
    print(
        f"cells={len(cells)} pending={len(pending)} reps={args.reps} "
        f"rules={len(RULES)} workers={args.workers}",
        flush=True,
    )
    started = time.time()
    # Cell-at-a-time keeps the checkpoint granularity coarse enough to be cheap
    # and fine enough that a crash costs a few minutes, not the whole grid.
    for index, cell in enumerate(pending, start=1):
        cell_started = time.time()
        records = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_replicate, (cell, rep)) for rep in range(args.reps)]
            for future in as_completed(futures):
                _, record = future.result()
                if record:
                    records.append(record)
        payload = {"cell": cell, "reps": args.reps, "rules": summarise(records)}
        checkpoint_path(results_dir, cell["id"]).write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )
        elapsed = (time.time() - started) / 60
        print(
            f"[{index}/{len(pending)}] {cell['id']} "
            f"{(time.time() - cell_started):.0f}s  elapsed {elapsed:.1f}m  "
            f"eta {elapsed / index * (len(pending) - index):.1f}m",
            flush=True,
        )

    print(f"\nshard {args.shard_index} finished in {(time.time() - started) / 60:.1f} min")
    merge(mine, results_dir, Path(args.out))


def merge(cells, results_dir: Path, out: Path):
    """Collect whatever per-cell files exist into one summary and render it."""
    summary = {}
    missing = []
    for cell in cells:
        path = checkpoint_path(results_dir, cell["id"])
        if path.exists():
            summary[cell["id"]] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(cell["id"])
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(summary)}/{len(cells)} cells)")
    if missing:
        print(f"MISSING {len(missing)} cells: {missing}")
    render(summary)


def render(summary):
    for cell_id, block in summary.items():
        print(f"\n== {cell_id}")
        print(
            f"   {'rule':13} {'bias':>8} {'cover':>7} {'rmse':>7} "
            f"{'mSE':>7} {'flex%':>6} {'ref':>4}"
        )
        for rule in RULES:
            row = block["rules"][rule]
            if not row.get("n"):
                print(f"   {rule:13} {'--- no usable replicates ---':>40}")
                continue
            print(
                f"   {rule:13} {row['bias']:>8.4f} {row['coverage']:>7.3f} "
                f"{row['rmse']:>7.4f} {row['mean_se']:>7.4f} "
                f"{row['flex_share'] * 100:>5.0f}% {row['refused']:>4d}"
            )


if __name__ == "__main__":
    sys.exit(main())
