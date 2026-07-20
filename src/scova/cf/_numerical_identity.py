"""Commit-level identities for frozen SCOVA-CF numerical evidence."""

from __future__ import annotations

import ast
import subprocess
from hashlib import sha256
from typing import Literal

EvidenceKind = Literal["external", "inference"]

# Whole-file entries are restricted to modules whose contents determine fitted
# values, uncertainty, or evidence gates.  Mixed orchestration modules are
# handled separately at symbol granularity below.
_CORE_NUMERICAL_PATHS = (
    "src/scova/_aipw.py",
    "src/scova/declaration.py",
    "src/scova/diagnostics.py",
    "src/scova/estimator.py",
    "src/scova/inference.py",
    "src/scova/result.py",
    "src/scova/cf/benchmarks.py",
    "src/scova/cf/declaration.py",
    "src/scova/cf/estimator.py",
    "src/scova/cf/result.py",
    "src/scova/cf/status.py",
    "src/scova/cf/support.py",
)

_CORE_NUMERICAL_DATA_PATHS = (
    "src/scova/cf/data/support_profiles.json",
)

_EXTERNAL_NUMERICAL_PATHS = (
    "benchmarks/cf_external_agreement.py",
    "benchmarks/cf_external_validation.py",
)

_CAMPAIGN_NUMERICAL_SYMBOLS = (
    "STABILITY_SEEDS",
    "CampaignData",
    "_probabilities",
    "_conditional_means",
    "_errors",
    "simulate_reference_cell",
    "_declaration",
    "_contrast_summary",
    "_support_features",
    "fit_campaign_record",
)

_INFERENCE_NUMERICAL_SYMBOLS = (
    "N_BOOTSTRAP",
    "_NUMERICAL_ENVIRONMENT_FIELDS",
    "_version",
    "_commit",
    "_numerical_environment_identity",
    "_familywise_error_gate",
    "run_shard",
    "aggregate",
)

_MIXED_NUMERICAL_SOURCES: dict[EvidenceKind, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "external": (
        ("benchmarks/cf_reference_campaign.py", _CAMPAIGN_NUMERICAL_SYMBOLS[:7]),
    ),
    "inference": (
        ("benchmarks/cf_reference_campaign.py", _CAMPAIGN_NUMERICAL_SYMBOLS),
        ("benchmarks/cf_inference_campaign.py", _INFERENCE_NUMERICAL_SYMBOLS),
    ),
}


class _NumericalAstNormalizer(ast.NodeTransformer):
    """Canonicalize identity-plumbing refactors that do not change evidence."""

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        visited = self.generic_visit(node)
        if not isinstance(visited, ast.Call):
            return visited
        node = visited
        if isinstance(node.func, ast.Name) and node.func.id in {
            "_cf_numerical_fingerprint",
            "cf_numerical_fingerprint",
        }:
            node.func.id = "cf_numerical_fingerprint"
            if (
                len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "inference"
            ):
                node.args = node.args[:1]
        return node


def _top_level_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def _symbol_payload(source: bytes, *, path: str, symbols: tuple[str, ...]) -> bytes:
    """Return formatting-insensitive AST payload for selected top-level symbols."""
    tree = ast.parse(source, filename=path)
    selected = {_top_level_name(node): node for node in tree.body}
    return "\n".join(
        (
            f"Missing(name={name!r})"
            if name not in selected
            else ast.dump(
                _NumericalAstNormalizer().visit(selected[name]),
                annotate_fields=True,
                include_attributes=False,
            )
        )
        for name in symbols
    ).encode("utf-8")


def _committed_file(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def cf_numerical_fingerprint(commit: str, kind: EvidenceKind) -> str:
    """Hash code that can change the specified evidence's numerical meaning."""
    digest = sha256()
    whole_paths = _CORE_NUMERICAL_PATHS + (
        _EXTERNAL_NUMERICAL_PATHS if kind == "external" else ()
    )
    for path in whole_paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            ast.dump(
                ast.parse(_committed_file(commit, path), filename=path),
                annotate_fields=True,
                include_attributes=False,
            ).encode("utf-8")
        )
        digest.update(b"\0")
    for path in _CORE_NUMERICAL_DATA_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_committed_file(commit, path))
        digest.update(b"\0")
    for path, symbols in _MIXED_NUMERICAL_SOURCES[kind]:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_symbol_payload(_committed_file(commit, path), path=path, symbols=symbols))
        digest.update(b"\0")
    return digest.hexdigest()


def same_cf_numerical_implementation(
    left: str, right: str, kind: EvidenceKind
) -> bool:
    """Fail closed if either commit or any required numerical source is unavailable."""
    try:
        return cf_numerical_fingerprint(left, kind) == cf_numerical_fingerprint(right, kind)
    except (OSError, subprocess.SubprocessError, SyntaxError, ValueError):
        return False
