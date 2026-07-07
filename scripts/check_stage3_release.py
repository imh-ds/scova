"""Fail unless every Stage-3 stable-promotion gate is satisfied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def blocking_reasons(manifest: dict) -> list[str]:
    reasons: list[str] = []
    theory = manifest["theory"]
    for key in ("review_1_approved", "review_2_approved"):
        if not theory[key]:
            reasons.append(f"theory.{key} is false")
    if theory["critical_comments_open"] or theory["major_comments_open"]:
        reasons.append("theory has unresolved critical or major comments")
    for section in ("simulation", "gates", "jax", "evidence"):
        for key, value in manifest[section].items():
            if not value:
                reasons.append(f"{section}.{key} is false")
    quality = manifest["quality"]
    if quality["critical_modules_branch_coverage"] < 0.95:
        reasons.append("critical module branch coverage is below 0.95")
    if quality["package_branch_coverage"] < 0.92:
        reasons.append("package branch coverage is below 0.92")
    if not quality["build_passed"]:
        reasons.append("quality.build_passed is false")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("release/stage3_promotion.json"),
    )
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reasons = blocking_reasons(manifest)
    if reasons:
        print("Stage 3 is not eligible for stable promotion:")
        for reason in reasons:
            print(f"- {reason}")
        if not args.allow_pending:
            raise SystemExit(1)
    else:
        print("All Stage-3 promotion gates are satisfied.")


if __name__ == "__main__":
    main()
