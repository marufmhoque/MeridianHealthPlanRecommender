import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(ROOT / "src"))

# Deterministic, offline test runs: fixed review date, no LLM calls.
os.environ["MERIDIAN_REVIEW_DATE"] = "2026-11-06"
os.environ["MERIDIAN_DISABLE_LLM"] = "1"
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture()
def reviewer(tmp_path):
    from langgraph.checkpoint.memory import MemorySaver

    from meridian_pa.graph import Reviewer
    from meridian_pa.memory import MemoryStore

    return Reviewer(memory=MemoryStore(tmp_path / "memory.db"), checkpointer=MemorySaver())


@pytest.fixture()
def requests():
    """Synthetic test requests (tests/fixtures/requests), keyed by request id."""
    from meridian_pa.data_loader import load_requests

    return {r.request_id: r for r in load_requests([FIXTURES / "requests"])}


@pytest.fixture()
def expected():
    data = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


@pytest.fixture()
def run_all(reviewer, requests):
    return {rid: reviewer.review(req) for rid, req in requests.items()}
