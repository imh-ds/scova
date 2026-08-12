"""Render a concise Markdown report for a comparative methods artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.cf_comparative_methods import cell_level_summaries

_INTERVAL_BASIS: dict[str, tuple[str, bool]] = {
    "scova-cf": ("influence-function", True),
    "linear-ancova": ("model-based", True),
    "independent-aipw": ("influence-function", True),
    "psm-att": ("not evaluated", False),
    "econml-drlearner": ("not evaluated", False),
    "econml-drlearner-conservative": ("not evaluated", False),
}


def _warning_rates(records: list[dict[str, Any]], *, estimand: str) -> dict[str, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["estimand"] == estimand:
            grouped.setdefault(str(record["method"]), []).append(record)
    return {
        method: (
            sum(row["status"] != "ok" for row in rows if row["estimate"] is not None)
            / sum(row["estimate"] is not None for row in rows)
            if any(row["estimate"] is not None for row in rows)
            else None
        )
        for method, rows in grouped.items()
    }


def _table(
    summaries: dict[str, Any], *, estimand: str, records: list[dict[str, Any]]
) -> list[str]:
    warning_rates = _warning_rates(records, estimand=estimand)
    lines = [
        (
            "| Method | Interval basis | Bias | RMSE | Coverage | Failure rate | Warning rate | "
            "Retention |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in sorted(summaries.items()):
        interval_basis, coverage_evaluable = _INTERVAL_BASIS.get(name, ("not evaluated", False))
        lines.append(
            (
                "| {name} | {interval_basis} | {bias} | {rmse} | {coverage} | {failure} | "
                "{warning} | {retention} |"
            ).format(
                name=name,
                interval_basis=interval_basis,
                bias=_format(summary["bias"]),
                rmse=_format(summary["rmse"]),
                coverage=_format(summary["coverage"]) if coverage_evaluable else "â€”",
                failure=_format(summary["failure_rate"]),
                warning=_format(summary.get("warning_rate", warning_rates.get(name))),
                retention=_format(summary["treated_retained_fraction"]),
            )
        )
    return lines


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _cell_table(summaries: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Cell | Method | Bias | RMSE | Median | 95th pct. | Maximum |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell_id, methods in sorted(summaries.items()):
        for method, summary in sorted(methods.items()):
            lines.append(
                "| {cell} | {method} | {bias} | {rmse} | {median} | {p95} | {maximum} |".format(
                    cell=cell_id,
                    method=method,
                    bias=_format(summary["bias"]),
                    rmse=_format(summary["rmse"]),
                    median=_format(summary["median_absolute_error"]),
                    p95=_format(summary["absolute_error_p95"]),
                    maximum=_format(summary["maximum_absolute_error"]),
                )
            )
    return lines


def render(artifact: dict[str, Any]) -> str:
    """Render a methods-only report without qualification language."""
    if artifact.get("program_type") != "methods":
        raise ValueError("comparative report requires a methods artifact")
    cell_summaries = cell_level_summaries(artifact["records"])
    cell_ate_summaries = artifact.get("cell_ate_summaries", cell_summaries["ate"])
    cell_att_summaries = artifact.get("cell_att_summaries", cell_summaries["att"])
    lines = [
        "# SCOVA-CF two-group comparative methods study",
        "",
        artifact["interpretation"],
        "",
        f"- Protocol checksum: `{artifact['protocol_checksum']}`",
        f"- Dependency-lock checksum: `{artifact['dependency_lock_checksum']}`",
        f"- Frozen commit: `{artifact['git_commit']}`",
        f"- Execution completeness: `{artifact['complete']}`",
        "",
        "## Study-population ATE estimators",
        "",
        (
            "Failure rate counts records with no numerical estimate or standard error. "
            "Warning rate counts retained estimates whose status is not `ok`; it is a "
            "diagnostic, not a numerical failure."
        ),
        (
            "Coverage is reported only when the adapter's standard error is an ATE/ATT "
            "sampling-uncertainty estimate. DRLearner CATE-spread and matched-pair proxy "
            "intervals are retained in the raw artifact but are not evaluated as coverage."
        ),
        "",
        *_table(artifact["ate_summaries"], estimand="ate", records=artifact["records"]),
        "",
        "## Matched-treated ATT estimator",
        "",
        (
            "PSM estimates the ATT among retained matched treated units; it is not ranked "
            "against the ATE estimators."
        ),
        "",
        *_table(artifact["att_summaries"], estimand="att", records=artifact["records"]),
        "",
        "## Cell-level ATE diagnostics",
        "",
        "Tail-error columns are descriptive diagnostics, not pass/fail criteria.",
        "",
        *_cell_table(cell_ate_summaries),
        "",
        "## Cell-level ATT diagnostics",
        "",
        *_cell_table(cell_att_summaries),
        "",
        "Incomplete smoke output is incomplete methods evidence, not a narrowed conclusion.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    args.output.write_text(render(artifact), encoding="utf-8")


if __name__ == "__main__":
    main()
