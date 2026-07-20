import json
from hashlib import sha256
from pathlib import Path

from benchmarks.stage3_campaign import _failed_fit
from scripts import promote_cf_reference
from scripts.check_critical_coverage import CRITICAL_SUFFIXES, coverage_failures
from scripts.check_stage3_release import blocking_reasons
from scripts.check_stage5b_promotion_audit import blocking_reasons as stage5b_blocking_reasons
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


def test_promotion_checker_uses_artifacts_without_review_blockers() -> None:
    manifest = json.loads(Path("release/stage3_promotion.json").read_text(encoding="utf-8"))
    reasons = blocking_reasons(manifest)
    assert not any("review" in reason for reason in reasons)
    assert any("missing thresholds artifact" in reason for reason in reasons)
    assert any("missing validation_summary artifact" in reason for reason in reasons)


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


def _write_hashed(
    path: Path,
    values: dict,
    field: str,
    *,
    indent: int | None = None,
    allow_nan: bool = True,
) -> dict:
    payload = dict(values)
    encoded = json.dumps(payload, indent=indent, sort_keys=True, allow_nan=allow_nan).encode()
    payload[field] = sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_artifact_backed_promotion_can_pass(tmp_path: Path) -> None:
    manifest = json.loads(Path("release/stage3_promotion.json").read_text(encoding="utf-8"))
    specification_path = tmp_path / manifest["protocol"]["specification"]
    specification_path.parent.mkdir(parents=True)
    specification_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "validation_level": "directional",
                "tiers": {
                    "calibration": {"cells": 1, "repetitions": 2},
                    "directional_validation": {"cells": 1, "repetitions": 2},
                    "directional_robustness": {"cells": 1, "repetitions": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_path = tmp_path / manifest["protocol"]["threshold_candidates"]
    candidate_path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    artifact_paths = {name: tmp_path / value for name, value in manifest["artifacts"].items()}
    threshold = _write_hashed(
        artifact_paths["thresholds"],
        {"calibrated": True, "validation_level": "directional"},
        "sha256",
    )
    artifact_paths["packaged_thresholds"].parent.mkdir(parents=True)
    artifact_paths["packaged_thresholds"].write_text(json.dumps(threshold), encoding="utf-8")
    specification_hash = sha256(specification_path.read_bytes()).hexdigest()
    for role, tier in (
        ("calibration_shards", "calibration"),
        ("validation_shards", "directional_validation"),
        ("robustness_shards", "directional_robustness"),
    ):
        _write_hashed(
            artifact_paths[role],
            {
                "tier": tier,
                "git_commit": "abc123",
                "specification_sha256": specification_hash,
                "threshold_artifact_sha256": (
                    None if role == "calibration_shards" else threshold["sha256"]
                ),
                "record_count": 2,
            },
            "sha256",
        )
    for role, tier in (
        ("validation_summary", "directional_validation"),
        ("robustness_summary", "directional_robustness"),
    ):
        _write_hashed(
            artifact_paths[role],
            {
                "all_cells_passed": True,
                "tier_passed": True,
                "tier": tier,
                "specification_sha256": specification_hash,
                "threshold_artifact_sha256": threshold["sha256"],
            },
            "summary_sha256",
            allow_nan=False,
        )
    for role, version in (
        ("jax_minimum", "0.4.38"),
        ("jax_maximum", "0.10.2"),
    ):
        _write_hashed(
            artifact_paths[role],
            {"jax_version": version, "cases": 2000, "failures": []},
            "sha256",
            indent=2,
        )
    _write_hashed(
        artifact_paths["memory"],
        {
            "peak_below_unbatched_multiplier_cube": True,
            "nuisance_refit_possible_from_result": False,
        },
        "sha256",
    )
    _write_hashed(artifact_paths["build"], {"passed": True}, "sha256")
    artifact_paths["coverage"].parent.mkdir(parents=True, exist_ok=True)
    artifact_paths["coverage"].write_text(json.dumps(_coverage_report()), encoding="utf-8")
    _write_hashed(
        artifact_paths["evidence"],
        {"complete": True, "validation_level": "directional"},
        "report_sha256",
        indent=2,
    )
    assert blocking_reasons(manifest, tmp_path) == []


def test_stage5b_promotion_audit_requires_experimental_linked_evidence(tmp_path: Path) -> None:
    specification = {
        "protocol": "stage5b-promotion-audit-v1",
        "repetitions_per_arm": 1000,
        "required_criteria": ["coverage", "experimental"],
    }
    spec_path = tmp_path / "benchmarks/specs/stage5b_promotion_audit.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps(specification), encoding="utf-8")
    audit_path = tmp_path / "release/stage5b_promotion_audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps({"protocol": specification["protocol"], "public_verdict": "experimental"}),
        encoding="utf-8",
    )
    evidence = {
        "protocol": specification["protocol"],
        "status": "pass",
        "audit_manifest_sha256": sha256(audit_path.read_bytes()).hexdigest(),
        "criteria": {"coverage": True, "experimental": True},
        "metrics": {"repetitions_per_arm": 1000},
    }
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["sha256"] = sha256(encoded).hexdigest()
    evidence_path = tmp_path / "release/artifacts/stage5b-promotion-evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert stage5b_blocking_reasons(tmp_path) == []
    evidence["audit_manifest_sha256"] = "tampered"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert stage5b_blocking_reasons(tmp_path)


def test_stage5b_workflows_keep_pr_and_full_audit_campaigns_separate() -> None:
    pr_workflow = Path(".github/workflows/stage5b-lipschitz-anchor.yml").read_text(encoding="utf-8")
    audit_workflow = Path(".github/workflows/stage5b-promotion-audit.yml").read_text(
        encoding="utf-8"
    )
    assert "pull_request:" in pr_workflow
    assert "stage5b_promotion_campaign.py" not in pr_workflow
    assert "schedule:" in audit_workflow
    assert "workflow_dispatch:" in audit_workflow
    assert "stage5b_promotion_campaign.py" in audit_workflow
    assert "pull_request:" not in audit_workflow


def test_cf_priority3_workflow_is_ordered_and_fail_closed() -> None:
    workflow = Path(".github/workflows/cf-reference-validation.yml").read_text(
        encoding="utf-8"
    )
    assert 'counts = {"pilot": 16, "simultaneous_inference": 64}' in workflow
    assert "--shard-count 16 --resume" in workflow
    assert "--stage external" in workflow
    assert "--stage inference" in workflow
    assert "--stage validation" in workflow
    assert "scova-cf-reference-v3-freeze-r4" in workflow
    assert "- freeze_check" in workflow
    prepare_job = workflow.split("  prepare:", 1)[1].split("  campaign_freeze:", 1)[0]
    freeze_job = workflow.split("  campaign_freeze:", 1)[1].split(
        "  prerequisite_lock:", 1
    )[0]
    assert "freeze_check" not in prepare_job
    assert "inputs.tier == 'freeze_check'" in freeze_job
    assert "python -m scripts.write_cf_freeze_manifest" in workflow
    assert "python scripts/write_cf_freeze_manifest.py" not in workflow
    for module in (
        "cf_reference_campaign",
        "aggregate_cf_campaign",
        "cf_external_agreement",
        "cf_inference_campaign",
    ):
        assert f"python -m benchmarks.{module}" in workflow
    assert "python benchmarks/" not in workflow
    aggregate_job = workflow.split("  campaign_aggregate:", 1)[1].split(
        "  calibrate_support:", 1
    )[0]
    assert "always()" in aggregate_job
    assert "needs.campaign_shard.result == 'success'" in aggregate_job
    calibrate_job = workflow.split("  calibrate_support:", 1)[1].split(
        "  external_agreement:", 1
    )[0]
    assert "--require-candidate" in calibrate_job
    assert "if: always()" in calibrate_job
    assert "actions/attest-build-provenance@v3" in workflow
    assert "SCOVA_RELEASE_GPG_PRIVATE_KEY" in workflow
    assert "gh release create v0.5.0" in workflow
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert '"numpy==2.2.6" "pandas==2.2.3" "scipy==1.15.3"' in ci
    assert '"scikit-learn==1.6.1"' in ci
    assert "benchmarks/specs/cf_reference_v3.json" in ci
    assert "benchmarks/specs/cf_reference_v1.json" not in ci
    assert "python benchmarks/" not in ci
    assert "--shard-count 4" in ci
    assert "--replications 1 --max-cells 4 --skip-stability" in ci


def test_cf_v4_workflow_is_isolated_from_v3_campaign_identity() -> None:
    workflow = Path(".github/workflows/cf-reference-v4-validation.yml").read_text(
        encoding="utf-8"
    )
    assert "SCOVA-CF v4 frozen reference validation" in workflow
    assert "benchmarks/specs/cf_reference_v4.json" in workflow
    assert "scova-cf-reference-v4-freeze-r1" in workflow
    assert "cf-reference-v3-freeze" not in workflow


def test_cf_v6_workflow_limits_execution_to_the_inference_amendment() -> None:
    workflow = Path(".github/workflows/cf-reference-v6-validation.yml").read_text(
        encoding="utf-8"
    )
    assert "SCOVA-CF v6 inference-profile amendment" in workflow
    assert "benchmarks/specs/cf_reference_v6.json" in workflow
    assert "scova-cf-reference-v6-freeze-r4" in workflow
    assert "- calibration\n" not in workflow
    assert "calibration_source" in workflow
    assert "- simultaneous_inference" in workflow
    assert "- external_smoke" not in workflow
    assert "- inference_reaggregate" in workflow


def test_cf_promotion_applies_only_exact_evidence_and_release_text(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    profile = {
        "profile_id": "cf-randomized-continuous-aipw-unnormalized-v3-promoted",
        "profile_checksum": "c" * 64,
    }
    proposed = {"schema_version": 1, "profiles": [profile]}
    (evidence / "cf-reference-support-profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    (evidence / "proposed-support-profiles.json").write_text(
        json.dumps(proposed), encoding="utf-8"
    )
    manifest = tmp_path / "support_profiles.json"
    manifest.write_text('{"schema_version":1,"profiles":[]}', encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.3.0.dev0"\n', encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("The `0.3.0.dev0` source tree is provisional.\n", encoding="utf-8")
    documentation = tmp_path / "scova_cf.md"
    documentation.write_text(
        f"{promote_cf_reference.STATUS_START}\ncandidate\n"
        f"{promote_cf_reference.STATUS_END}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(promote_cf_reference, "blocking_reasons", lambda *_: [])
    promote_cf_reference.promote(
        evidence_root=evidence,
        spec=tmp_path / "spec.json",
        packaged_manifest=manifest,
        pyproject=pyproject,
        readme=readme,
        documentation=documentation,
    )
    assert json.loads(manifest.read_text(encoding="utf-8")) == proposed
    assert 'version = "0.5.0"' in pyproject.read_text(encoding="utf-8")
    assert "The `0.5.0` source tree" in readme.read_text(encoding="utf-8")
    assert profile["profile_id"] in documentation.read_text(encoding="utf-8")
