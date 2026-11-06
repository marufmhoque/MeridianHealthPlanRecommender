"""Tools exposed to the Financial agent (and to Claude via tool calling).

All arithmetic happens here, in plain Python, never in the model. When Claude
drives the Financial agent it chooses *which* tool to call with *which*
arguments; the numbers it reports back are taken from these tool results.
"""
from __future__ import annotations

from typing import Any

from .data_loader import load_fee_schedule, load_members, load_plans
from .schemas import CostEstimate


def get_member_benefits(member_id: str) -> dict[str, Any]:
    """Eligibility + accumulators + plan design for one member."""
    members = load_members()
    if member_id not in members:
        return {"error": f"member {member_id} not found"}
    m = members[member_id]
    plan = load_plans().get(m["plan"])
    if plan is None:
        return {"error": f"plan {m['plan']} not configured"}
    ded = plan["deductible"]["in_network"]
    oop = plan["oop_max"]["in_network"]
    return {
        "member_id": member_id,
        "plan": m["plan"],
        "plan_name": plan["plan_name"],
        "deductible": ded,
        "deductible_met": m["deductible_met"],
        "deductible_remaining": max(0, ded - m["deductible_met"]),
        "oop_max": oop,
        "oop_met": m["oop_met"],
        "oop_remaining": max(0, oop - m["oop_met"]),
        "pt_visits_ytd": m.get("pt_visits_ytd", 0),
        "services": plan["services"],
        "no_auth_rule": plan["no_auth_rule"],
    }


def lookup_allowed_amount(cpt_codes: list[str]) -> dict[str, Any]:
    fees = load_fee_schedule()
    found = {c: fees[c] for c in cpt_codes if c in fees}
    missing = [c for c in cpt_codes if c not in fees]
    service_types = {v["service_type"] for v in found.values()}
    return {
        "codes": found,
        "missing": missing,
        "service_type": service_types.pop() if len(service_types) == 1 else None,
        "allowed_per_unit": round(sum(v["allowed_amount"] for v in found.values()), 2),
    }


def calculate_patient_cost(
    member_id: str,
    cpt_codes: list[str],
    units: int = 1,
    network: str = "in_network",
) -> dict[str, Any]:
    """Member out-of-pocket estimate for ``units`` services (MRI=1, PT=visits)."""
    benefits = get_member_benefits(member_id)
    if "error" in benefits:
        return benefits
    fee = lookup_allowed_amount(cpt_codes)
    stype = fee["service_type"]
    if not stype or stype not in benefits["services"]:
        return {"error": f"no benefit configured for CPT {cpt_codes}"}
    rule = benefits["services"][stype][network]
    allowed_unit = fee["allowed_per_unit"]
    ded_rem = float(benefits["deductible_remaining"])
    oop_rem = float(benefits["oop_remaining"])
    est = CostEstimate(service_type=stype, network=network, units=units)
    notes, steps = est.notes, est.breakdown

    if rule["type"] == "copay_per_visit":
        limit = rule.get("annual_visit_limit")
        ytd = benefits["pt_visits_ytd"]
        covered = units
        if limit is not None:
            covered = max(0, min(units, limit - ytd))
            est.visits_remaining_after = max(0, limit - ytd - covered)
            if covered < units:
                notes.append(f"Only {covered} of {units} visits fit under the {limit}-visit annual limit.")
        per_visit = min(rule["copay"], allowed_unit)
        est.allowed_amount_total = round(allowed_unit * covered, 2)
        est.copay_applied = round(per_visit * covered, 2)
        steps.append(f"{covered} visit(s) x ${per_visit:,.2f} copay = ${est.copay_applied:,.2f}")
        steps.append("Copays are not subject to the deductible on this plan design.")
        member = est.copay_applied
    else:  # coinsurance after deductible
        rate = rule["coinsurance"]
        allowed = round(allowed_unit * units, 2)
        est.allowed_amount_total = allowed
        est.deductible_applied = round(min(allowed, ded_rem), 2)
        est.coinsurance_applied = round((allowed - est.deductible_applied) * rate, 2)
        steps.append(f"Estimated allowed amount: ${allowed:,.2f} (fictional fee schedule)")
        steps.append(
            f"Deductible: ${est.deductible_applied:,.2f} applied "
            f"(${ded_rem:,.2f} remaining of ${benefits['deductible']:,})"
        )
        steps.append(
            f"Coinsurance: {int(rate * 100)}% x ${allowed - est.deductible_applied:,.2f} "
            f"= ${est.coinsurance_applied:,.2f}"
        )
        member = est.deductible_applied + est.coinsurance_applied

    if member > oop_rem:
        notes.append(f"Capped at out-of-pocket maximum (${oop_rem:,.2f} remaining).")
        steps.append(f"OOP max cap: ${member:,.2f} -> ${oop_rem:,.2f}")
        member = oop_rem
    est.member_responsibility = round(member, 2)
    est.plan_pays = round(max(0.0, est.allowed_amount_total - est.member_responsibility), 2)
    est.deductible_remaining_after = round(ded_rem - est.deductible_applied, 2)
    est.oop_remaining_after = round(oop_rem - est.member_responsibility, 2)
    steps.append(f"Estimated member responsibility: ${est.member_responsibility:,.2f}")
    if fee["missing"]:
        notes.append(f"No fee schedule entry for {fee['missing']}; excluded from estimate.")
    return est.model_dump()


TOOL_REGISTRY = {
    "get_member_benefits": get_member_benefits,
    "calculate_patient_cost": calculate_patient_cost,
}

# Anthropic tool definitions (strict -> inputs always match the schema).
ANTHROPIC_TOOLS = [
    {
        "name": "get_member_benefits",
        "description": (
            "Return the member's plan design and year-to-date accumulators: deductible and "
            "out-of-pocket maximum (amount met and remaining), PT visits used this year, and the "
            "cost-sharing rule for each service category. Call this first for any cost question."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"member_id": {"type": "string", "description": "Member ID, e.g. M1001"}},
            "required": ["member_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate_patient_cost",
        "description": (
            "Compute the member's estimated out-of-pocket cost for a service, applying deductible, "
            "coinsurance or copay, annual visit limits and the out-of-pocket maximum. Use units=1 for "
            "imaging and units=<number of visits requested> for therapy."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "member_id": {"type": "string"},
                "cpt_codes": {"type": "array", "items": {"type": "string"}},
                "units": {"type": "integer", "description": "Number of services or visits"},
                "network": {"type": "string", "enum": ["in_network", "out_of_network"]},
            },
            "required": ["member_id", "cpt_codes", "units", "network"],
            "additionalProperties": False,
        },
    },
]
