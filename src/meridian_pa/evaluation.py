"""Offline evaluation harness (Ragas-style metrics, computed locally).

* decision_accuracy  - recommendation vs. expected label, only when the caller
                       supplies expected outcomes (optional; reference-free otherwise)
* faithfulness       - share of factual claims in the letter (numbers, dates,
                       quoted policy text) that are found in the source note,
                       request, policy, or cost tool output: catches invented facts
* answer_relevance   - letter addresses this request: id, member, service,
                       governing policy, and an outcome consistent with the packet
* context_recall     - policy chunks needed for the packet's decision basis are
                       among the retrieved chunks
* guardrail_pass     - output validation passed
* phi_redaction      - no member name / DOB / member ID in the redacted note

Ragas or TruLens can be plugged in for LLM-judged versions of these metrics
when an LLM key is available. These deterministic versions run in CI.
"""
from __future__ import annotations

import re

from .data_loader import load_policy_documents

# Policy chunks a reviewer needs in front of them to check each decision basis.
REQUIRED_CHUNKS = {
    "MP-204-C1": ["MP-204-02"],
    "MP-204-C2": ["MP-204-03"],
    "MP-204-C3": ["MP-204-04"],
    "MP-204-NMN1": ["MP-204-05"],
    "MP-204-DOC": ["MP-204-08", "MP-204-10"],
    "MP-310-AUTH": ["MP-310-02"],
    "MP-310-C1-3": ["MP-310-03", "MP-310-04", "MP-310-05"],
    "MP-310-C3": ["MP-310-05"],
    "MP-310-MAX": ["MP-310-06"],
    "MP-310-DOC": ["MP-310-03", "MP-310-04", "MP-310-05"],
}

_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_QUOTE = re.compile(r"\"([^\"]{12,})\"")
_OUTCOME_WORDS = {"APPROVE": "approved", "DENY": "does not meet medical necessity", "PEND": "pended"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("*", "")).strip().lower()


def faithfulness(state: dict, policy_docs: dict[str, str]) -> dict:
    req, packet = state["request"], state["packet"]
    sources = " ".join([
        req.get("raw_text", ""), req.get("requested_service", ""),
        " ".join(policy_docs.values()), str(state.get("financial", {})),
        " ".join(e for c in packet.get("criteria", []) for e in c.get("evidence", [])),
    ])
    src_norm = _norm(sources)
    src_nums = {n.replace("$", "").replace(",", "") for n in _NUM.findall(sources)}
    body = packet["letter"].split("\n", 2)[-1]  # skip banner
    body = re.sub(r"^Date: .*$", "", body, flags=re.M)  # review date is metadata, not a claim
    claims, unsupported = 0, []
    for q in _QUOTE.findall(body):
        claims += 1
        if _norm(q) not in src_norm:
            unsupported.append(f"quote: {q[:60]}")
    for n in _NUM.findall(body):
        key = n.replace("$", "").replace(",", "")
        claims += 1
        if key not in src_nums and not _derived_number(key, req, packet):
            unsupported.append(f"number: {n}")
    score = 1.0 if claims == 0 else round((claims - len(unsupported)) / claims, 3)
    return {"score": score, "claims": claims, "unsupported": unsupported}


def _derived_number(key: str, req: dict, packet: dict) -> bool:
    """Numbers legitimately computed by the pipeline (visit ranges, year, list indices)."""
    try:
        v = float(key)
    except ValueError:
        return False
    if v.is_integer() and 1 <= v <= 30:  # visit numbers / list indices / criterion numbers
        return True
    if v.is_integer() and 2000 <= v <= 2100:  # calendar year
        return True
    return False


def answer_relevance(state: dict) -> dict:
    req, packet, letter = state["request"], state["packet"], state["packet"]["letter"]
    checks = {
        "request_id": req["request_id"] in letter,
        "member": bool(req.get("member_name")) and req["member_name"] in letter,
        "service": all(c in letter for c in req.get("cpt_codes", [])) and bool(req.get("cpt_codes")),
        "policy": packet["policy_id"] in letter,
        "outcome": _OUTCOME_WORDS[packet["recommendation"]] in letter,
    }
    return {"score": round(sum(checks.values()) / len(checks), 3), "checks": checks}


def context_recall(state: dict) -> dict:
    need = REQUIRED_CHUNKS.get(state["packet"]["primary_basis"], [])
    got = {c["chunk_id"] for c in state.get("retrieved", [])}
    if not need:
        return {"score": None, "missing": []}
    missing = [n for n in need if n not in got]
    return {"score": round((len(need) - len(missing)) / len(need), 3), "missing": missing}


def phi_redaction(state: dict) -> dict:
    req, red = state["request"], state.get("redacted_note", "")
    leaks = [x for x in (req.get("member_name"), req.get("dob"), req.get("member_id")) if x and x in red]
    for part in (req.get("member_name") or "").split():
        if len(part) > 2 and re.search(rf"\b{re.escape(part)}\b", red):
            leaks.append(part)
    return {"score": 0.0 if leaks else 1.0, "leaks": leaks}


def evaluate_state(state: dict, gold: dict | None, policy_docs: dict[str, str]) -> dict:
    rec = state["packet"]["recommendation"]
    return {
        "request_id": state["request"]["request_id"],
        "recommendation": rec,
        "expected": gold["recommendation"] if gold else None,
        "correct": (rec == gold["recommendation"]) if gold else None,
        "faithfulness": faithfulness(state, policy_docs),
        "answer_relevance": answer_relevance(state),
        "context_recall": context_recall(state),
        "guardrail_pass": bool(state.get("output_validation", {}).get("ok")),
        "phi_redaction": phi_redaction(state),
    }


def evaluate_run(states: dict[str, dict], expected: dict[str, dict] | None = None) -> dict:
    """Score a run. ``expected`` ({request_id: {"recommendation": ...}}) is optional."""
    gold = expected or {}
    docs = load_policy_documents()
    per = [evaluate_state(s, gold.get(rid), docs) for rid, s in states.items()]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    agg = {
        "requests": len(per),
        "decision_accuracy": mean([1.0 if p["correct"] else 0.0 for p in per if p["correct"] is not None]),
        "faithfulness": mean([p["faithfulness"]["score"] for p in per]),
        "answer_relevance": mean([p["answer_relevance"]["score"] for p in per]),
        "context_recall": mean([p["context_recall"]["score"] for p in per]),
        "guardrail_pass_rate": mean([1.0 if p["guardrail_pass"] else 0.0 for p in per]),
        "phi_redaction": mean([p["phi_redaction"]["score"] for p in per]),
    }
    return {"aggregate": agg, "per_request": per}
