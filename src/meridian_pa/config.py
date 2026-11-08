"""Runtime configuration (paths, model ids, feature switches)."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MERIDIAN_DATA_DIR", ROOT / "data"))
POLICY_DIR = DATA_DIR / "policies"


def _writable_dir(preferred: Path, name: str) -> Path:
    """Use ``preferred`` if it can be created and written, else a temp dir (read-only hosts)."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return preferred
    except OSError:
        import tempfile

        fallback = Path(tempfile.gettempdir()) / "meridian_pa" / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


STATE_DIR = _writable_dir(Path(os.getenv("MERIDIAN_STATE_DIR", ROOT / ".state")), "state")
OUTPUT_DIR = _writable_dir(Path(os.getenv("MERIDIAN_OUTPUT_DIR", ROOT / "outputs")), "outputs")

# Claude is used only when a key is present AND LLM mode is not disabled.
# Without it the pipeline runs fully deterministically (rules + local embeddings).
CLAUDE_MODEL = os.getenv("MERIDIAN_CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("MERIDIAN_CLAUDE_EFFORT", "low")
OPENAI_EMBEDDING_MODEL = os.getenv("MERIDIAN_EMBEDDING_MODEL", "text-embedding-3-small")


def llm_enabled() -> bool:
    if os.getenv("MERIDIAN_DISABLE_LLM", "").lower() in {"1", "true", "yes"}:
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def review_date() -> date:
    """Date the review is performed; drives the 30-day note-recency check."""
    override = os.getenv("MERIDIAN_REVIEW_DATE")
    return date.fromisoformat(override) if override else date.today()


def ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
