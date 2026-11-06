"""Long-term memory: a small SQLite store that outlives any one review.

* ``packets``  - every draft the assistant produced (member, service, facts).
  Used to spot duplicate requests and contradictions across a member's notes.
* ``reviews``  - the nurse's final decision, edited letter, and comments.
  Finalized approvals feed MP-204's "repeat imaging within 12 months" check.

Short-term (per-request) memory is the LangGraph checkpointer in graph.py,
keyed by thread_id = request_id, so a review can be resumed after a restart.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS packets (
    request_id TEXT PRIMARY KEY,
    member_id TEXT,
    service_category TEXT,
    cpt_codes TEXT,
    recommendation TEXT,
    facts_json TEXT,
    packet_json TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    nurse_decision TEXT,
    reviewer TEXT,
    final_letter TEXT,
    comments TEXT,
    ai_recommendation TEXT,
    overridden INTEGER,
    created_at TEXT
);
"""


class MemoryStore:
    def __init__(self, path: Path | None = None):
        config.ensure_dirs()
        self.path = path or (config.STATE_DIR / "memory.db")
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def save_packet(self, request_id: str, member_id: str | None, service_category: str,
                    cpt_codes: list[str], recommendation: str, facts: dict, packet: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO packets VALUES (?,?,?,?,?,?,?,?)",
            (request_id, member_id, service_category, ",".join(cpt_codes), recommendation,
             json.dumps(facts, default=str), json.dumps(packet, default=str), _now()),
        )
        self.conn.commit()

    def save_review(self, request_id: str, nurse_decision: str, reviewer: str, final_letter: str,
                    comments: str, ai_recommendation: str) -> None:
        self.conn.execute(
            "INSERT INTO reviews (request_id, nurse_decision, reviewer, final_letter, comments, "
            "ai_recommendation, overridden, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (request_id, nurse_decision, reviewer, final_letter, comments, ai_recommendation,
             int(nurse_decision != ai_recommendation), _now()),
        )
        self.conn.commit()

    def latest_review(self, request_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM reviews WHERE request_id=? ORDER BY id DESC LIMIT 1", (request_id,)
        ).fetchone()
        return dict(row) if row else None

    def member_history(self, member_id: str | None, exclude_request_id: str | None = None) -> list[dict]:
        if not member_id:
            return []
        rows = self.conn.execute(
            "SELECT * FROM packets WHERE member_id=? AND request_id != ? ORDER BY created_at",
            (member_id, exclude_request_id or ""),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["facts"] = json.loads(d.pop("facts_json") or "{}")
            d.pop("packet_json", None)
            review = self.latest_review(d["request_id"])
            if review:
                d["nurse_decision"] = review["nurse_decision"]
            out.append(d)
        return out

    def override_rate(self) -> dict:
        row = self.conn.execute("SELECT COUNT(*) n, COALESCE(SUM(overridden),0) o FROM reviews").fetchone()
        return {"reviews": row["n"], "overrides": row["o"]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
