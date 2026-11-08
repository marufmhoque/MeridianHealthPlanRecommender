"""Nurse reviewer dashboard: upload prior-auth requests, review the AI decision
packets, edit the letters, record final decisions, and download the results.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meridian_pa import config  # noqa: E402
from meridian_pa.data_loader import parse_request, request_id_from_name  # noqa: E402
from meridian_pa.evaluation import evaluate_run  # noqa: E402
from meridian_pa.graph import Reviewer  # noqa: E402
from meridian_pa.report import packet_markdown  # noqa: E402

st.set_page_config(page_title="Meridian Prior Auth Assistant", layout="wide")

STATUS_STYLE = {"MET": "Met", "NOT_MET": "Not met", "INSUFFICIENT_INFO": "Insufficient info", "NOT_APPLICABLE": "N/A"}
REC_COLOR = {"APPROVE": "green", "DENY": "red", "PEND": "orange"}


@st.cache_resource
def get_reviewer() -> Reviewer:
    return Reviewer()


def results_zip(states: dict[str, dict]) -> bytes:
    """All packets, letters, a summary and the evaluation in one archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rid, s in states.items():
            final = reviewer.memory.latest_review(rid)
            z.writestr(f"{rid}/{rid}_packet.json", json.dumps(s, indent=2, default=str))
            z.writestr(f"{rid}/{rid}_draft_letter.txt", s["packet"]["letter"])
            if final:
                z.writestr(f"{rid}/{rid}_final_letter.txt", final["final_letter"])
        z.writestr("summary.md", "\n".join(packet_markdown(s) for s in states.values()))
        z.writestr("evaluation.json", json.dumps(evaluate_run(states), indent=2))
    return buf.getvalue()


reviewer = get_reviewer()
states: dict[str, dict] = st.session_state.setdefault("states", {})

# ---------------- Sidebar: upload + queue ----------------
st.sidebar.title("Prior auth requests")
st.sidebar.caption(f"Review date: {config.review_date().isoformat()} | "
                   f"LLM: {'Claude ' + config.CLAUDE_MODEL if config.llm_enabled() else 'off (deterministic mode)'}")

uploads = st.sidebar.file_uploader("Upload request files (.md / .txt)", type=["md", "txt"],
                                   accept_multiple_files=True, key=f"up{st.session_state.get('up_key', 0)}")
if st.sidebar.button("Review uploaded files", type="primary", disabled=not uploads):
    with st.spinner("Running the review agents..."):
        for f in sorted(uploads, key=lambda u: u.name):
            req = parse_request(f.getvalue().decode("utf-8", errors="replace"), request_id_from_name(f.name), f.name)
            states[req.request_id] = reviewer.review(req)
    st.session_state.up_key = st.session_state.get("up_key", 0) + 1
    st.rerun()

with st.sidebar.expander("Or paste a request"):
    txt = st.text_area("Request text", height=160,
                       placeholder="PRIOR AUTH REQUEST - Member M1001 (Name), DOB YYYY-MM-DD\nRequested: ...\n"
                                   "Ordering: ...\nOffice note YYYY-MM-DD:\n...")
    pasted_id = st.text_input("Request ID", value=f"REQ-P{len(states) + 1:02d}")
    if st.button("Review pasted request", disabled=not txt.strip()):
        states[pasted_id] = reviewer.review(parse_request(txt, pasted_id, "pasted"))
        st.rerun()

if not states:
    st.title("Prior Authorization Assistant")
    st.info("Upload one or more prior-authorization request files in the sidebar, then click "
            "**Review uploaded files**. Each request gets a decision packet: policy criteria check, member cost "
            "estimate, reviewer flags, and a draft provider letter you can edit, finalize, and download.")
    st.stop()

if st.sidebar.button("Clear queue"):
    st.session_state.states = {}
    st.rerun()
st.sidebar.download_button("Download all results (.zip)", results_zip(states), file_name="prior_auth_results.zip",
                           mime="application/zip")

labels = []
for rid, s in states.items():
    p, r = s["packet"], s["request"]
    done = reviewer.memory.latest_review(rid)
    mark = f" | final: {done['nurse_decision']}" if done else ""
    labels.append(f"{rid} - {r.get('member_name') or '?'} - AI: {p['recommendation']}{mark}")
choice = st.sidebar.radio("Review queue", labels, index=0)
rid = choice.split(" - ")[0]
state = states[rid]
req, packet = state["request"], state["packet"]

# ---------------- Header ----------------
st.title("Prior Authorization Decision Packet")
st.warning("AI recommendation only. The assistant never makes the determination: a licensed reviewer "
           "verifies the evidence, edits the letter, and records the final decision.", icon="⚠️")
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"**Request**  \n{rid}")
c2.markdown(f"**AI recommendation**  \n:{REC_COLOR[packet['recommendation']]}[**{packet['recommendation']}**]")
c3.metric("Confidence", packet["confidence"])
c4.metric("Policy", packet["policy_id"])
st.caption(f"{req.get('member_name')} ({req.get('member_id')}) | {req.get('requested_service')} | "
           f"Ordering: {req.get('ordering_provider')}")

for fl in packet["flags"]:
    {"CRITICAL": st.error, "WARNING": st.warning, "INFO": st.info}[fl["severity"]](fl["message"])

tabs = st.tabs(["Policy audit", "Clinical facts", "Financial", "Draft letter & decision", "Agent trace", "Evaluation"])

# ---------------- Policy audit ----------------
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Criteria, line by line")
        for c in packet["criteria"]:
            icon = {"MET": "✅", "NOT_MET": "❌", "INSUFFICIENT_INFO": "❔", "NOT_APPLICABLE": "➖"}[c["status"]]
            with st.expander(f"{icon} {c['criterion_id']} - {c['description']} - {STATUS_STYLE[c['status']]}",
                             expanded=c["criterion_id"] == packet["primary_basis"]):
                st.markdown("**Evidence from the note**")
                for e in c["evidence"]:
                    st.markdown(f"- {e}")
                if c["policy_text"]:
                    st.markdown("**Policy text**")
                    st.info(c["policy_text"])
        if packet["missing_information"]:
            st.subheader("Information to request")
            for m in packet["missing_information"]:
                st.markdown(f"- {m}")
    with right:
        st.subheader("Source note")
        st.code(req["raw_text"], language=None)
        st.subheader("Retrieved policy context")
        st.caption("Top chunks from the vector index for this request")
        for ch in state.get("retrieved", []):
            st.markdown(f"`{ch['chunk_id']}` score {ch['score']:.2f}: {ch['text']}")

# ---------------- Clinical facts ----------------
with tabs[1]:
    facts = state.get("facts") or {}
    st.caption(f"Extraction method: {facts.get('extraction_method', 'n/a')}")
    rows = [{"Field": k, "Value": json.dumps(v) if isinstance(v, (list, dict)) else str(v)}
            for k, v in facts.items() if v not in (None, [], "")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("PHI-redacted note (what leaves the system)")
    st.code(state.get("redacted_note", ""), language=None)

# ---------------- Financial ----------------
with tabs[2]:
    fin = state.get("financial") or {}
    b = fin.get("benefits") or {}
    cost = packet.get("cost_estimate")
    st.markdown(f"**{packet['cost_summary']}**")
    if b and "error" not in b:
        a, bb, c = st.columns(3)
        a.metric("Deductible met", f"${b['deductible_met']:,} / ${b['deductible']:,}")
        bb.metric("OOP met", f"${b['oop_met']:,} / ${b['oop_max']:,}")
        c.metric("PT visits YTD", f"{b['pt_visits_ytd']} / 30")
    if cost:
        st.subheader("Calculation")
        for line in cost["breakdown"]:
            st.markdown(f"- {line}")
        for n in cost["notes"]:
            st.caption(n)
        chart = pd.DataFrame({"Amount": [cost["member_responsibility"], cost["plan_pays"]]},
                             index=["Member pays", "Plan pays"])
        st.bar_chart(chart, horizontal=True)
    st.caption(f"Tool planner: {fin.get('planner')}. Allowed amounts come from a fictional fee schedule.")
    with st.expander("Tool calls"):
        st.json(fin.get("tool_calls", []))

# ---------------- Letter & decision ----------------
with tabs[3]:
    prior = reviewer.memory.latest_review(rid)
    letter = st.text_area("Provider letter (editable)", value=prior["final_letter"] if prior else packet["letter"],
                          height=480)
    col_a, col_b = st.columns(2)
    options = ["APPROVE", "DENY", "PEND"]
    decision = col_a.radio("Final decision", options, horizontal=True,
                           index=options.index(prior["nurse_decision"] if prior else packet["recommendation"]))
    reviewer_name = col_b.text_input("Reviewer", value=prior["reviewer"] if prior else "")
    comments = st.text_area("Reviewer comments / rationale for any override", value=prior["comments"] if prior else "")
    if decision != packet["recommendation"]:
        st.warning(f"You are overriding the AI recommendation ({packet['recommendation']}). Document the rationale.")
    if st.button("Record final decision", type="primary", disabled=not reviewer_name.strip()):
        final = letter.replace(
            letter.splitlines()[0], f"FINALIZED BY {reviewer_name.strip()}") if letter.startswith("***") else letter
        reviewer.memory.save_review(rid, decision, reviewer_name.strip(), final, comments, packet["recommendation"])
        st.success(f"Decision recorded: {decision} by {reviewer_name}. Saved to reviewer memory.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download letter (.txt)", letter, file_name=f"{rid}_letter.txt")
    d2.download_button("Download packet (.json)", json.dumps(state, indent=2, default=str),
                       file_name=f"{rid}_packet.json", mime="application/json")
    d3.download_button("Download packet summary (.md)", packet_markdown(state), file_name=f"{rid}_summary.md")

# ---------------- Trace ----------------
with tabs[4]:
    st.subheader("LangGraph execution")
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style=rounded];
      guardrails_in -> clinical_extractor -> policy_retriever -> financial_calculator -> decision_synthesizer -> guardrails_out;
      guardrails_in -> reject_input [style=dashed, label="invalid"]; reject_input -> guardrails_out; }""")
    st.dataframe(pd.DataFrame(state.get("trace", [])), hide_index=True, width="stretch")
    st.subheader("Output guardrails")
    st.json(state.get("output_validation", {}))

# ---------------- Evaluation ----------------
with tabs[5]:
    report = evaluate_run(states)
    st.subheader("Run-level metrics")
    agg = report["aggregate"]
    items = [(k, v) for k, v in agg.items() if k != "requests"]
    for row in (items[:3], items[3:]):
        for col, (k, v) in zip(st.columns(3), row):
            col.metric(k.replace("_", " ").capitalize(), "n/a" if v is None else f"{v:.0%}")
    st.dataframe(pd.DataFrame([{
        "request": p["request_id"], "AI": p["recommendation"], "expected": p["expected"], "correct": p["correct"],
        "faithfulness": p["faithfulness"]["score"], "relevance": p["answer_relevance"]["score"],
        "context recall": p["context_recall"]["score"], "guardrails": p["guardrail_pass"],
    } for p in report["per_request"]]), hide_index=True, width="stretch")
    ov = reviewer.memory.override_rate()
    st.caption(f"Nurse reviews recorded: {ov['reviews']}, overrides of AI recommendation: {ov['overrides']}")
