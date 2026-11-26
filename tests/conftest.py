"""
tests/conftest.py — shared test setup.

Every test that runs the pipeline would normally write to the project's
real memory.json. The autouse fixture below points the memory file at a
temporary folder instead, so tests never touch your real data — and
never leave files behind.
"""

import os
import sys

import pytest

# Make sure `import agents...` works when pytest runs from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def temp_memory_file(tmp_path, monkeypatch):
    """Redirect memory.json and results.json to a temp folder per test."""
    import agents.orchestrator as orchestrator
    import api

    monkeypatch.setattr(
        orchestrator, "MEMORY_FILE", str(tmp_path / "memory.json")
    )
    monkeypatch.setattr(api, "RESULTS_FILE", str(tmp_path / "results.json"))
    api._RESULTS.clear()
    yield
