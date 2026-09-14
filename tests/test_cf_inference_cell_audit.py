from __future__ import annotations

from pathlib import Path

from scova.cf import CFValidationProtocol, canonical_checksum
from scripts.audit_cf_inference_cell import summarize_audit

V9_SPEC = Path("benchmarks/specs/cf_reference_v9.json")


def _record(
    seed: int,
    *,
    family_covered: bool,
    first_covered: bool = True,
    second_covered: bool = True,
    critical_value: float = 2.4,
) -> dict:
    return {
        "seed": seed,
        "refused": False,
        "contrasts": [
            {
                "estimate": 0.0 if first_covered else 1.0,
                "truth": 0.0,
                "standard_error": 1.0,
                "covered": first_covered,
                "null": True,
            },
            {
                "estimate": 0.0 if second_covered else 1.0,
                "truth": 0.0,
                "standard_error": 1.0,
                "covered": second_covered,
                "null": True,
            },
        ],
        "simultaneous": {
            "critical_value": critical_value,
            "covered_family": family_covered,
            "any_null_rejected": not family_covered,
            "max_t_p_value": 0.05 if not family_covered else 0.5,
        },
    }


def test_summarize_audit_persists_critical_values_and_coverage_gate() -> None:
    protocol = CFValidationProtocol.load(V9_SPEC)
    records = [
        _record(4_500_000_000, family_covered=True, critical_value=2.1),
        _record(4_500_000_001, family_covered=True, critical_value=2.2),
        _record(
            4_500_000_002,
            family_covered=False,
            second_covered=False,
            critical_value=2.3,
        ),
        _record(
            4_500_000_003,
            family_covered=False,
            first_covered=False,
            critical_value=2.4,
        ),
    ]

    report = summarize_audit(
        protocol,
        cell_index=4,
        cell=protocol.inference_cells[4]["cell"],
        seed_start=4_500_000_000,
        expected_replications=4,
        bootstrap_replications=999,
        records=records,
    )

    assert report["completed_replications"] == 4
    assert report["refusal_count"] == 0
    assert report["simultaneous_coverage"] == 0.5
    assert report["pointwise_coverage"] == [0.75, 0.75]
    assert report["critical_value_summary"] == {
        "minimum": 2.1,
        "mean": 2.25,
        "median": 2.25,
        "p95": 2.4,
        "maximum": 2.4,
    }
    assert report["coverage_gate"]["target"] == 0.95
    assert report["coverage_gate"]["lower_bound"] > 0.5
    assert report["coverage_gate"]["passed"] is False
    assert report["records"][2]["simultaneous"]["critical_value"] == 2.3
    supplied_checksum = report.pop("evidence_checksum")
    assert supplied_checksum == canonical_checksum(report)


def test_summarize_audit_reports_refusals_without_claiming_completion() -> None:
    protocol = CFValidationProtocol.load(V9_SPEC)
    report = summarize_audit(
        protocol,
        cell_index=4,
        cell=protocol.inference_cells[4]["cell"],
        seed_start=4_500_000_000,
        expected_replications=2,
        bootstrap_replications=999,
        records=[
            {"seed": 4_500_000_000, "refused": True, "status_code": "refused/test"},
            _record(4_500_000_001, family_covered=True),
        ],
    )

    assert report["completed_replications"] == 1
    assert report["refusal_count"] == 1
    assert report["all_replications_completed"] is False
    assert report["simultaneous_coverage"] == 1.0


def test_v9_workflow_exposes_independent_cell_audit_tier() -> None:
    workflow = Path(".github/workflows/cf-reference-v9-validation.yml").read_text(
        encoding="utf-8"
    )

    assert "- inference_cell_audit" in workflow
    assert "audit_seed_start" in workflow
    assert "audit_replications" in workflow
    audit_job = workflow.split("  inference_cell_audit:", 1)[1].split(
        "  inference_reaggregate:", 1
    )[0]
    assert "needs: campaign_freeze" in audit_job
    assert "python -m scripts.audit_cf_inference_cell" in audit_job
    assert "cf-inference-cell-audit" in audit_job
