"""Apply a validated SCOVA-CF profile to a release checkout, and nothing else."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from scripts.check_cf_reference_release import blocking_reasons
except ModuleNotFoundError:  # direct ``python scripts/promote_cf_reference.py``
    from check_cf_reference_release import blocking_reasons

STATUS_START = "<!-- CF_REFERENCE_PROFILE_STATUS_START -->"
STATUS_END = "<!-- CF_REFERENCE_PROFILE_STATUS_END -->"


def _replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(
        pattern, replacement, text, count=1, flags=re.MULTILINE | re.DOTALL
    )
    if count != 1:
        raise ValueError(f"Could not update exactly one {label}")
    return updated


def promote(
    *,
    evidence_root: Path,
    spec: Path,
    packaged_manifest: Path,
    pyproject: Path,
    readme: Path,
    documentation: Path,
) -> None:
    reasons = blocking_reasons(evidence_root, spec)
    if reasons:
        raise ValueError("Promotion evidence failed:\n- " + "\n- ".join(reasons))
    proposed = json.loads(
        (evidence_root / "proposed-support-profiles.json").read_text(encoding="utf-8")
    )
    profile = json.loads(
        (evidence_root / "cf-reference-support-profile.json").read_text(encoding="utf-8")
    )
    if proposed != {"schema_version": 1, "profiles": [profile]}:
        raise ValueError("Proposed manifest is not the exact promoted profile")
    packaged_manifest.write_text(
        json.dumps(proposed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pyproject.write_text(
        _replace_once(
            pyproject.read_text(encoding="utf-8"),
            r'^version = "[^"]+"$',
            'version = "0.5.0"',
            "project version",
        ),
        encoding="utf-8",
    )
    readme.write_text(
        _replace_once(
            readme.read_text(encoding="utf-8"),
            r"The `[^`]+` source tree",
            "The `0.5.0` source tree",
            "README version",
        ),
        encoding="utf-8",
    )
    status = (
        f"{STATUS_START}\n"
        f"The randomized continuous unnormalized-AIPW profile `{profile['profile_id']}` "
        f"is promoted. Its packaged profile checksum is `{profile['profile_checksum']}` and "
        "confirmatory support is available only when this profile is explicitly selected.\n"
        f"{STATUS_END}"
    )
    documentation.write_text(
        _replace_once(
            documentation.read_text(encoding="utf-8"),
            rf"{re.escape(STATUS_START)}.*?{re.escape(STATUS_END)}",
            status,
            "SCOVA-CF profile status block",
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--packaged-manifest",
        type=Path,
        default=Path("src/scova/cf/data/support_profiles.json"),
    )
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--documentation", type=Path, default=Path("docs/scova_cf.md"))
    args = parser.parse_args()
    promote(
        evidence_root=args.evidence_root,
        spec=args.spec,
        packaged_manifest=args.packaged_manifest,
        pyproject=args.pyproject,
        readme=args.readme,
        documentation=args.documentation,
    )


if __name__ == "__main__":
    main()
