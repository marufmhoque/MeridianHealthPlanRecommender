"""Loads prior-auth requests, members, plans, fee schedule and policies from disk."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from . import config
from .schemas import PriorAuthRequest

_HEADER = re.compile(
    r"Member\s+(?P<id>M\d+)\s*(?:\((?P<name>[^)]+)\))?(?:,\s*DOB\s+(?P<dob>\d{4}-\d{2}-\d{2}))?",
    re.I,
)
_CPT = re.compile(r"CPT\s+([\d,\s]+)", re.I)
_NOTE_DATE = re.compile(r"^(?:[A-Za-z ]*note)\s+(\d{4}-\d{2}-\d{2})\s*:", re.I | re.M)


def request_id_from_name(name: str) -> str:
    """req_001.md -> REQ-001; any other file name -> its upper-cased stem."""
    stem = Path(name).stem
    m = re.fullmatch(r"req[_-]?(\d+)", stem, re.I)
    return f"REQ-{int(m.group(1)):03d}" if m else stem.upper()


def parse_request(text: str, request_id: str, source_file: str | None = None) -> PriorAuthRequest:
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    header = lines[0] if lines else ""
    req = PriorAuthRequest(request_id=request_id, raw_text=text, source_file=source_file)

    if m := _HEADER.search(header):
        req.member_id = m.group("id")
        req.member_name = (m.group("name") or "").strip() or None
        req.dob = m.group("dob")

    note_lines: list[str] = []
    in_note = False
    for ln in lines[1:]:
        low = ln.lower()
        if low.startswith("requested:"):
            req.requested_service = ln.split(":", 1)[1].strip()
            if cm := _CPT.search(ln):
                req.cpt_codes = [c.strip() for c in cm.group(1).split(",") if c.strip()]
        elif low.startswith("ordering:"):
            req.ordering_provider = ln.split(":", 1)[1].strip()
        elif dm := _NOTE_DATE.match(ln):
            req.note_date = dm.group(1)
            in_note = True
            rest = ln[dm.end():].strip()
            if rest:
                note_lines.append(rest)
        elif low.startswith("note:"):
            in_note = True
            note_lines.append(ln.split(":", 1)[1].strip())
        elif in_note or ln.strip():
            in_note = True
            note_lines.append(ln)
    req.note_text = "\n".join(note_lines).strip()
    return req


def load_request_file(path: Path) -> PriorAuthRequest:
    return parse_request(path.read_text(encoding="utf-8"), request_id_from_name(path.name), path.name)


def load_requests(paths: list[Path]) -> list[PriorAuthRequest]:
    """Load request files from any mix of files and directories (*.md, *.txt)."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in {".md", ".txt"}) if p.is_dir() else [p]
    return [load_request_file(f) for f in files]


@lru_cache(maxsize=1)
def load_members() -> dict[str, dict]:
    data = json.loads((config.DATA_DIR / "members.json").read_text(encoding="utf-8"))
    return {m["member_id"]: m for m in data}


@lru_cache(maxsize=1)
def load_plans() -> dict[str, dict]:
    data = json.loads((config.DATA_DIR / "plans.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def load_fee_schedule() -> dict[str, dict]:
    data = json.loads((config.DATA_DIR / "fee_schedule.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_gold_labels(path: Path | None = None) -> dict[str, dict]:
    """Optional expected outcomes {request_id: {"recommendation": ...}} for scoring a run."""
    if path is None or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_policy_documents() -> dict[str, str]:
    """Returns {policy_id: markdown}. Policy id is read from the document header."""
    docs: dict[str, str] = {}
    for p in sorted(config.POLICY_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"Medical Policy\s+(MP-\d+)", text)
        docs[m.group(1) if m else p.stem] = text
    return docs
