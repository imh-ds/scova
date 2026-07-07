"""Select and lock monotone thresholds using calibration seeds only."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from scipy.stats import beta

MINIMUM_METRICS = ("min_group_ess", "target_ess_ratio", "min_propensity_q01")


def _accepted(metrics: dict[str, float], profile: dict) -> bool:
    for name in MINIMUM_METRICS:
        if metrics[name] < profile[name]:
            return False
    for name in set(profile).difference(MINIMUM_METRICS, {"name"}):
        if metrics[name] > profile[name]:
            return False
    return True


def _lower(successes: int, total: int) -> float:
    return 0.0 if successes == 0 else float(beta.ppf(0.05, successes, total - successes + 1))


def _upper(successes: int, total: int) -> float:
    return 1.0 if successes == total else float(beta.ppf(0.95, successes + 1, total - successes))


def calibrate(campaigns: list[dict], candidates: dict, seed_namespace: int) -> dict:
    if any(campaign.get("seed_namespace") != seed_namespace for campaign in campaigns):
        raise ValueError("threshold calibration received non-calibration seeds")
    records = [record for campaign in campaigns for record in campaign["records"]]
    selected = None
    audit = []
    for profile in candidates["profiles"]:
        accepted = [
            record
            for record in records
            if not record["alternative"]["refused"]
            and _accepted(record["alternative"]["gate_metrics"], profile)
        ]
        null_accepted = [
            record
            for record in records
            if not record["null"]["refused"]
            and _accepted(record["null"]["gate_metrics"], profile)
        ]
        covered = sum(record["alternative"]["uniform_coverage"] for record in accepted)
        false_signs = sum(record["null"]["false_sign_certificate"] for record in null_accepted)
        coverage_lower = _lower(covered, len(accepted)) if accepted else 0.0
        fwer_upper = _upper(false_signs, len(null_accepted)) if null_accepted else 1.0
        passed = coverage_lower >= 0.925 and fwer_upper <= 0.065
        audit.append(
            {
                "profile": profile["name"],
                "accepted": len(accepted),
                "coverage_lower_95": coverage_lower,
                "fwer_upper_95": fwer_upper,
                "passed": passed,
            }
        )
        if passed and selected is None:
            selected = profile
    if selected is None:
        raise RuntimeError("no candidate threshold profile satisfies calibration criteria")
    selected_index = candidates["profiles"].index(selected)
    warning_profile = candidates["profiles"][max(0, selected_index - 1)]
    artifact = {
        "schema_version": 1,
        "version": "stage3-calibrated-v1",
        "calibrated": True,
        "pass_profile": selected,
        "warning_floor_profile": warning_profile,
        "audit": audit,
    }
    encoded = json.dumps(artifact, sort_keys=True).encode()
    artifact["sha256"] = sha256(encoded).hexdigest()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("benchmarks/specs/stage3_threshold_candidates.json"),
    )
    parser.add_argument(
        "--release-spec",
        type=Path,
        default=Path("benchmarks/specs/stage3_release.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaigns = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    release_spec = json.loads(args.release_spec.read_text(encoding="utf-8"))
    artifact = calibrate(campaigns, candidates, release_spec["calibration_seed_namespace"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
