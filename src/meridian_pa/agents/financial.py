"""Financial Cost Calculator agent.

With Claude available, the model plans the tool calls (get_member_benefits,
then calculate_patient_cost) in a tool-use loop. Without it, a deterministic
planner makes the same calls. Either way the dollar figures come from the
tool output, never from model-generated text.
"""
from __future__ import annotations

from ..llm import tool_loop
from ..schemas import ClinicalFacts, PriorAuthRequest, ServiceCategory
from ..tools import ANTHROPIC_TOOLS, TOOL_REGISTRY, calculate_patient_cost, get_member_benefits

_SYSTEM = (
    "You are the financial agent in a prior-authorization review. Use the tools to estimate the member's "
    "out-of-pocket cost if the requested service is approved and rendered in-network. Call "
    "get_member_benefits first, then calculate_patient_cost exactly once with the request's CPT codes. "
    "Units are 1 for imaging, or the number of therapy visits requested. Reply with one short sentence; "
    "do not do any arithmetic yourself."
)


def _units(f: ClinicalFacts) -> int:
    if f.service_category == ServiceCategory.PHYSICAL_THERAPY:
        return f.visits_requested or 1
    return 1


def run(req: PriorAuthRequest, f: ClinicalFacts) -> dict:
    units = _units(f)
    if req.member_id and req.cpt_codes:
        user = (f"Member {req.member_id}; CPT codes {req.cpt_codes}; units {units}; network in_network.")
        result = tool_loop(_SYSTEM, user, ANTHROPIC_TOOLS, TOOL_REGISTRY)
        if result:
            _, calls = result
            benefits = next((c["output"] for c in calls if c["tool"] == "get_member_benefits"), None)
            cost = next((c["output"] for c in reversed(calls) if c["tool"] == "calculate_patient_cost"), None)
            if benefits and cost and "error" not in cost:
                return {"benefits": benefits, "cost": cost, "tool_calls": calls, "planner": "claude"}

    # Deterministic planner: same tools, same order.
    calls = []
    benefits = get_member_benefits(req.member_id or "")
    calls.append({"tool": "get_member_benefits", "input": {"member_id": req.member_id}, "output": benefits})
    cost = None
    if "error" not in benefits and req.cpt_codes:
        args = {"member_id": req.member_id, "cpt_codes": req.cpt_codes, "units": units, "network": "in_network"}
        cost = calculate_patient_cost(**args)
        calls.append({"tool": "calculate_patient_cost", "input": args, "output": cost})
        if "error" in cost:
            cost = None
    return {"benefits": benefits, "cost": cost, "tool_calls": calls, "planner": "deterministic"}
