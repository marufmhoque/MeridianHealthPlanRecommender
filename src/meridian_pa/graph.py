"""LangGraph orchestration of the prior-authorization review.

    guardrails_in -> clinical_extractor -> policy_retriever -> financial_calculator
                  -> decision_synthesizer -> guardrails_out -> END
    guardrails_in --(invalid input)--> reject_input -> guardrails_out

State is checkpointed per request (thread_id = request_id) so a review can be
inspected or resumed later; long-term member memory lives in memory.py.
"""
from __future__ import annotations

import operator
import sqlite3
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config
from .agents import decision, extractor, financial, policy
from .data_loader import load_members, load_policy_documents
from .guardrails import known_names, redact_phi, validate_packet, validate_request
from .memory import MemoryStore
from .schemas import (ClinicalFacts, CriterionResult, DecisionPacket, PriorAuthRequest, Recommendation,
                      ReviewerFlag)
from .tools import get_member_benefits
from .vectorstore import get_index


class ReviewState(TypedDict, total=False):
    request: dict
    validation: dict
    redacted_note: str
    facts: dict
    extraction_notes: list[str]
    history: list[dict]
    policy_id: str
    policy_title: str
    retrieved: list[dict]
    criteria: list[dict]
    financial: dict
    packet: dict
    output_validation: dict
    trace: Annotated[list[dict], operator.add]


def _t(node: str, start: float, **info: Any) -> list[dict]:
    return [{"node": node, "ms": round((time.perf_counter() - start) * 1000, 1), **info}]


def build_graph(memory: MemoryStore, checkpointer=None):
    index = get_index()
    policy_docs = load_policy_documents()

    def guardrails_in(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        req = PriorAuthRequest.model_validate(state["request"])
        v = validate_request(req, load_members())
        return {
            "validation": {"ok": v.ok, "errors": v.errors, "warnings": v.warnings},
            "redacted_note": redact_phi(req.note_text, known_names(req)),
            "trace": _t("guardrails_in", t0, ok=v.ok, errors=len(v.errors)),
        }

    def route_after_input(state: ReviewState) -> str:
        return "clinical_extractor" if state["validation"]["ok"] else "reject_input"

    def reject_input(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        req = PriorAuthRequest.model_validate(state["request"])
        errs = state["validation"]["errors"]
        letter = "\n".join([decision.DRAFT_BANNER, "", f"Subject: Request {req.request_id} could not be processed",
                            "", "The submission is incomplete:", *[f"  - {e}" for e in errs], "",
                            "Please correct and resubmit.", "", "Utilization Management, Meridian Health Plan"])
        packet = DecisionPacket(
            request_id=req.request_id, recommendation=Recommendation.PEND, policy_id="N/A",
            policy_title="Input validation", primary_basis="INPUT_VALIDATION", criteria=[],
            flags=[ReviewerFlag(severity="CRITICAL", message=e) for e in errs],
            missing_information=errs, letter=letter, confidence="LOW",
        )
        return {"packet": packet.model_dump(mode="json"), "trace": _t("reject_input", t0)}

    def clinical_extractor(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        req = PriorAuthRequest.model_validate(state["request"])
        facts, notes = extractor.extract(req)
        return {"facts": facts.model_dump(mode="json"), "extraction_notes": notes,
                "trace": _t("clinical_extractor", t0, method=facts.extraction_method)}

    def policy_retriever(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        req = PriorAuthRequest.model_validate(state["request"])
        f = ClinicalFacts.model_validate(state["facts"])
        history = memory.member_history(req.member_id, exclude_request_id=req.request_id)
        pid, title, chunks = policy.retrieve(index, req, f)
        if f.service_category.value == "MRI_LUMBAR":
            criteria = policy.evaluate_mri(index, req, f, history)
        elif f.service_category.value == "PHYSICAL_THERAPY":
            criteria = policy.evaluate_pt(index, req, f, get_member_benefits(req.member_id or ""))
        else:
            criteria = []
        return {
            "policy_id": pid, "policy_title": title, "history": history,
            "retrieved": [c.model_dump() for c in chunks],
            "criteria": [c.model_dump() for c in criteria],
            "trace": _t("policy_retriever", t0, policy=pid, retrieved=len(chunks), backend=index.backend),
        }

    def financial_calculator(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        req = PriorAuthRequest.model_validate(state["request"])
        f = ClinicalFacts.model_validate(state["facts"])
        fin = financial.run(req, f)
        return {"financial": fin, "trace": _t("financial_calculator", t0, planner=fin["planner"],
                                               tool_calls=len(fin["tool_calls"]))}

    def decision_synthesizer(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        req = PriorAuthRequest.model_validate(state["request"])
        f = ClinicalFacts.model_validate(state["facts"])
        packet = decision.run(
            req, f, state["policy_id"], state["policy_title"],
            [CriterionResult.model_validate(c) for c in state["criteria"]], state["financial"],
            state.get("history", []), state.get("extraction_notes", []), state["validation"]["warnings"],
        )
        return {"packet": packet.model_dump(mode="json"),
                "trace": _t("decision_synthesizer", t0, recommendation=packet.recommendation.value)}

    def guardrails_out(state: ReviewState) -> dict:
        t0 = time.perf_counter()
        packet = DecisionPacket.model_validate(state["packet"])
        req = PriorAuthRequest.model_validate(state["request"])
        if packet.primary_basis == "INPUT_VALIDATION":
            result = {"ok": True, "errors": [], "warnings": ["Input rejected; output checks skipped."]}
        else:
            v = validate_packet(packet, policy_docs)
            result = {"ok": v.ok, "errors": v.errors, "warnings": v.warnings}
            if not v.ok:
                packet.flags += [ReviewerFlag(severity="CRITICAL", message=f"Output guardrail: {e}") for e in v.errors]
                packet.confidence = "LOW"
        memory.save_packet(req.request_id, req.member_id, (state.get("facts") or {}).get("service_category", "UNKNOWN"),
                           req.cpt_codes, packet.recommendation.value, state.get("facts") or {},
                           packet.model_dump(mode="json"))
        return {"packet": packet.model_dump(mode="json"), "output_validation": result,
                "trace": _t("guardrails_out", t0, ok=result["ok"])}

    g = StateGraph(ReviewState)
    for name, fn in [("guardrails_in", guardrails_in), ("reject_input", reject_input),
                     ("clinical_extractor", clinical_extractor), ("policy_retriever", policy_retriever),
                     ("financial_calculator", financial_calculator),
                     ("decision_synthesizer", decision_synthesizer), ("guardrails_out", guardrails_out)]:
        g.add_node(name, fn)
    g.add_edge(START, "guardrails_in")
    g.add_conditional_edges("guardrails_in", route_after_input, ["clinical_extractor", "reject_input"])
    g.add_edge("reject_input", "guardrails_out")
    g.add_edge("clinical_extractor", "policy_retriever")
    g.add_edge("policy_retriever", "financial_calculator")
    g.add_edge("financial_calculator", "decision_synthesizer")
    g.add_edge("decision_synthesizer", "guardrails_out")
    g.add_edge("guardrails_out", END)
    return g.compile(checkpointer=checkpointer)


def default_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver

    config.ensure_dirs()
    conn = sqlite3.connect(str(config.STATE_DIR / "checkpoints.db"), check_same_thread=False)
    return SqliteSaver(conn)


class Reviewer:
    """Convenience facade used by the CLI, UI and tests."""

    def __init__(self, memory: MemoryStore | None = None, checkpointer=None, persist_checkpoints: bool = True):
        self.memory = memory or MemoryStore()
        if checkpointer is None and persist_checkpoints:
            checkpointer = default_checkpointer()
        self.graph = build_graph(self.memory, checkpointer)

    def review(self, req: PriorAuthRequest) -> ReviewState:
        cfg = {"configurable": {"thread_id": req.request_id}}
        return self.graph.invoke({"request": req.model_dump(mode="json"), "trace": []}, cfg)
