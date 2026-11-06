"""Markdown rendering of a decision packet (used by the CLI summary)."""
from __future__ import annotations

_ICON = {"MET": "MET", "NOT_MET": "NOT MET", "INSUFFICIENT_INFO": "INSUFFICIENT INFO", "NOT_APPLICABLE": "N/A"}


def packet_markdown(state: dict) -> str:
    req, p = state["request"], state["packet"]
    out = [
        f"## {req['request_id']}: {req.get('member_name')} ({req.get('member_id')}): {req.get('requested_service')}",
        "",
        f"**Recommendation:** {p['recommendation']} (basis `{p['primary_basis']}`, confidence {p['confidence']})  ",
        f"**Policy:** {p['policy_id']} {p['policy_title']}  ",
        f"**Ordering:** {req.get('ordering_provider')}  ",
        f"**Cost:** {p['cost_summary']}",
        "",
        "| Criterion | Status | Evidence |",
        "|---|---|---|",
    ]
    for c in p["criteria"]:
        ev = "; ".join(c["evidence"]).replace("|", "/")
        out.append(f"| {c['criterion_id']} {c['description']} | {_ICON[c['status']]} | {ev} |")
    if p["flags"]:
        out += ["", "**Reviewer flags:**"]
        out += [f"- [{f['severity']}] {f['message']}" for f in p["flags"]]
    if p["missing_information"]:
        out += ["", "**Information requested:**"]
        out += [f"- {m}" for m in p["missing_information"]]
    out += ["", "<details><summary>Draft provider letter</summary>", "", "```text", p["letter"], "```", "</details>", ""]
    return "\n".join(out)
