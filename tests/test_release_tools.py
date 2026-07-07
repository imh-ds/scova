import json
from pathlib import Path

from benchmarks.stage3_campaign import _failed_fit
from scripts.check_critical_coverage import CRITICAL_SUFFIXES, coverage_failures
from scripts.check_stage3_release import blocking_reasons
from scripts.generate_stage3_evidence import _artifact_record


def _coverage_report(fraction: float = 1.0) -> dict:
    summary = {
        "num_statements": 50,
        "covered_lines": round(50 * fraction),
        "num_branches": 10,
        "covered_branches": round(10 * fraction),
    }
    return {
        "files": {f"src/{suffix}": {"summary": summary} for suffix in CRITICAL_SUFFIXES},
        "totals": summary,
    }


def test_coverage_gate_accepts_and_rejects_reports() -> None:
    assert coverage_failures(_coverage_report()) == []
    failures = coverage_failures(_coverage_report(0.8))
    assert len(failures) == len(CRITICAL_SUFFIXES) + 1
    missing = _coverage_report()
    missing["files"].pop(next(iter(missing["files"])))
    assert "expected exactly one" in coverage_failures(missing)[0]


def test_promotion_checker_preserves_external_blockers() -> None:
    manifest = json.loads(Path("release/stage3_promotion.json").read_text(encoding="utf-8"))
    reasons = blocking_reasons(manifest)
    assert "theory.review_1_approved is false" in reasons
    assert not any("coverage is below" in reason for reason in reasons)
    assert "quality.build_passed is false" not in reasons


def test_evidence_and_campaign_failures_are_explicit(tmp_path: Path) -> None:
    missing = _artifact_record("simulation_summary", None)
    assert missing == {
        "role": "simulation_summary",
        "path": None,
        "present": False,
        "sha256": None,
    }
    present_path = tmp_path / "artifact.json"
    present_path.write_text("{}", encoding="utf-8")
    present = _artifact_record("artifact", present_path)
    assert present["present"] is True
    assert present["sha256"]

    failure = _failed_fit(RuntimeError("deliberate"))
    assert failure["refused"] is True
    assert failure["execution_error"] == {
        "type": "RuntimeError",
        "message": "deliberate",
    }
