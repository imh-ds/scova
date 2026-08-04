"""Report-only, post-candidate density diagnostic for qualification evidence."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scova.cf import CFSupportProfile, CFValidationProtocol, canonical_checksum
from scripts.calibrate_cf_support import _passes, arm_density, expected_smallest_arm, _screening_cell_gate

MAXIMUM_LOG10_INTERVAL_WIDTH = math.log10(2.0)


def boundary_diagnostic(
    protocol: CFValidationProtocol, candidate: CFSupportProfile | None, calibration: dict[str, Any],
) -> dict[str, Any]:
    """Estimate a density boundary after candidate selection, never adopt it."""
    base = {
        "artifact_type": "scova-cf-boundary-diagnostic", "schema_version": 1,
        "program_type": "qualification", "verification_lane": "boundary",
        "protocol_checksum": protocol.checksum,
        "design_checksum": str((protocol.design_selection or {}).get("design_checksum", "")),
        "calibration_evidence_checksum": calibration.get("evidence_checksum"),
        "candidate_profile_checksum": None if candidate is None else candidate.checksum,
        "maximum_95_log10_interval_width": MAXIMUM_LOG10_INTERVAL_WIDTH,
        "promotion_required": False, "scope_effect": "none", "promotion_effect": "none",
    }
    if candidate is None or candidate.state != "candidate":
        result = {**base, "status": "unavailable/no-candidate-profile", "informative": False}
        return {**result, "artifact_checksum": canonical_checksum(result)}
    if calibration.get("protocol_checksum") != protocol.checksum:
        raise ValueError("Boundary calibration evidence uses a different protocol")
    rows: list[tuple[int, float, bool]] = []
    for index, cell in enumerate(protocol.retained_cells):
        records = [r for r in calibration["records"] if int(r["cell_index"]) == index and not r["refused"]]
        supported = [r for r in records if _passes(r, dict(candidate.thresholds))]
        if not supported:
            continue
        density = arm_density(dict(cell), "simulated")
        if density <= 0 or expected_smallest_arm(dict(cell)) is None:
            continue
        passed, _audit = _screening_cell_gate(supported, protocol.calibration_gate_metrics)
        rows.append((int(cell["n_groups"]), math.log10(density), passed))
    strata = (2, 3)
    design = {stratum: sorted({row[1] for row in rows if row[0] == stratum}) for stratum in strata}
    design_adequate = all(len(design[stratum]) >= 3 for stratum in strata) and len(rows) >= 6
    realized_adequate = design_adequate and all(
        {row[2] for row in rows if row[0] == stratum} == {False, True} for stratum in strata
    )
    result: dict[str, Any] = {**base, "design_adequacy": {str(k): len(v) for k, v in design.items()}, "realized_fit_adequate": realized_adequate}
    if not realized_adequate:
        result.update({"status": "unavailable/realized-fit-insufficient", "informative": False})
        return {**result, "artifact_checksum": canonical_checksum(result)}
    from sklearn.linear_model import LogisticRegression
    x = np.array([[density, groups == 2, groups == 3] for groups, density, _ in rows], dtype=float)
    y = np.array([passed for _, _, passed in rows], dtype=int)
    model = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000).fit(x, y)
    coefficient = model.coef_.ravel()
    if coefficient[0] <= 0:
        result.update({"status": "unavailable/non-positive-density-effect", "informative": False})
        return {**result, "artifact_checksum": canonical_checksum(result)}
    target = math.log(float(protocol.metrics["minimum_strong_cell_pass_fraction"]) / (1 - float(protocol.metrics["minimum_strong_cell_pass_fraction"])))
    points = {stratum: float((target - coefficient[1 + (stratum == 3)]) / coefficient[0]) for stratum in strata}
    rng = np.random.default_rng(20260803)
    draws = {stratum: [] for stratum in strata}
    for _ in range(1000):
        sample = rng.integers(0, len(rows), len(rows))
        if len(set(y[sample])) < 2:
            continue
        fitted = LogisticRegression(penalty=None, fit_intercept=False, max_iter=5000).fit(x[sample], y[sample]).coef_.ravel()
        if fitted[0] > 0:
            for stratum in strata:
                draws[stratum].append(float((target - fitted[1 + (stratum == 3)]) / fitted[0]))
    intervals = {stratum: [float(np.quantile(draws[stratum], q)) for q in (0.025, 0.975)] for stratum in strata if draws[stratum]}
    informative = len(intervals) == 2 and all(interval[1] - interval[0] <= MAXIMUM_LOG10_INTERVAL_WIDTH for interval in intervals.values())
    result.update({"status": "complete/informative" if informative else "unavailable/imprecise-interval", "informative": informative, "point_log10_density": points, "intervals_log10_density": intervals})
    return {**result, "artifact_checksum": canonical_checksum(result)}
