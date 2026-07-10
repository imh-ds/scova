import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_smoke_verifier_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_stage4_smoke.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_stage4_aggregator_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "benchmarks/summarize_stage4.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
