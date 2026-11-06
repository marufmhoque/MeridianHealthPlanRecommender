"""Decision & Letter Synthesis agent.

Combines criterion results, cost estimate, and member history into a
DecisionPacket: a recommendation (APPROVE / DENY / PEND), reviewer flags,
a cost summary, and a provider letter drafted from fixed templates so every
statement traces back to the note, the policy, or the tool output.
"""
from __future__ import annotations

from ..config import review_date
from ..schemas import (ClinicalFacts, CostEstimate, CriterionResult, DecisionPacket, PriorAuthRequest,
                       Recommendation, ReviewerFlag, ServiceCategory)

DRAFT_BANNER = (
    "*** DRAFT - PREPARED BY THE AI ASSISTANT FOR NURSE REVIEW. "
    "NOT A COVERAGE DETERMINATION UNTIL REVIEWED AND SIGNED BY A LICENSED CLINICIAN. ***"
)

MISSING_INFO_TEXT = {
    "MP-204-DOC1": "A complete office evaluation note dated within 30 days of the request",
    "MP-204-DOC2": "Onset date and duration of the current back pain",
    "MP-204-DOC3": "Conservative therapy tried: type, dates, and response (PT or supervised home exercise program; NSAID/analgesic trial)",
    "MP-204-DOC4": "Neurological examination findings (strength, reflexes, sensation, straight leg raise)",
    "MP-204-C1": "Cancer history details (type, treatment dates, current status) and whether this back pain is new",
    "MP-204-NMN2": "Date and result of any prior lumbar MRI, and the change in clinical status since",
    "MP-310-C1": "Specific, measurable functional goals",
    "MP-310-C2": "Objective measures of progress since start of care (ROM, strength, LEFS/ODI)",
    "MP-310-C3": "Statement of expected continued functional improvement",
}


_SERVICE_LABEL = {ServiceCategory.MRI_LUMBAR: "lumbar spine MRI", ServiceCategory.PHYSICAL_THERAPY: "physical therapy"}


def _by_id(criteria: list[CriterionResult]) -> dict[str, CriterionResult]:
    return {c.criterion_id: c for c in criteria}


def build_flags(req: PriorAuthRequest, f: ClinicalFacts, history: list[dict], benefits: dict,
                extraction_notes: list[str], validation_warnings: list[str]) -> list[ReviewerFlag]:
    flags = [ReviewerFlag(severity="WARNING", message=m) for m in extraction_notes]
    flags += [ReviewerFlag(severity="INFO", message=m) for m in validation_warnings]

    if f.pt_weeks_stated and f.pt_weeks_by_dates and f.pt_weeks_by_dates < f.pt_weeks_stated - 0.25:
        flags.append(ReviewerFlag(severity="WARNING", message=(
            f"Note states {f.pt_weeks_stated:g} weeks of PT, but the dates given ({f.pt_start} to {f.pt_end}) span "
            f"{f.pt_weeks_by_dates:g} weeks. MP-204 requires at least 6 weeks; confirm PT course length or "
            f"accept the provider's attestation of 6 weeks.")))

    same_service = [h for h in history if h.get("service_category") == f.service_category.value]
    for h in same_service:
        status = h.get("nurse_decision") or f"AI draft: {h.get('recommendation')}"
        flags.append(ReviewerFlag(severity="WARNING", message=(
            f"Member has another {_SERVICE_LABEL.get(f.service_category, 'same-service')} request on file "
            f"({h['request_id']}, CPT {h.get('cpt_codes')}, {status}). Check for a duplicate submission.")))
    for h in history:
        prior = h.get("facts", {})
        if prior.get("cancer_history") is not None and f.cancer_history is not None \
                and prior["cancer_history"] != f.cancer_history:
            this_says = "documents a cancer history" if f.cancer_history else "states no cancer history"
            prior_says = "documents a cancer history" if prior["cancer_history"] else "states 'No cancer history'"
            flags.append(ReviewerFlag(severity="CRITICAL", message=(
                f"Conflicting history: this request {this_says}, but {h['request_id']} {prior_says}. "
                f"Reconcile with the ordering provider before relying on either.")))

    if f.service_category == ServiceCategory.PHYSICAL_THERAPY and f.visits_completed is not None:
        ytd = benefits.get("pt_visits_ytd")
        if ytd is not None and ytd != f.visits_completed:
            flags.append(ReviewerFlag(severity="WARNING", message=(
                f"Therapy note reports {f.visits_completed} visits completed; claims history shows {ytd}.")))
        elif ytd is not None:
            flags.append(ReviewerFlag(severity="INFO", message=(
                f"Visit count in note ({f.visits_completed}) matches claims history ({ytd}).")))
    return flags


def decide_mri(c: dict[str, CriterionResult]) -> tuple[Recommendation, str]:
    coverage = [c["MP-204-C1"], c["MP-204-C2"], c["MP-204-C3"]]
    met = [x for x in coverage if x.status == "MET"]
    if met:
        if c["MP-204-DOC1"].status != "MET" or c["MP-204-NMN2"].status == "INSUFFICIENT_INFO":
            return Recommendation.PEND, met[0].criterion_id
        return Recommendation.APPROVE, met[0].criterion_id
    if c["MP-204-NMN1"].status == "NOT_MET":
        return Recommendation.DENY, "MP-204-NMN1"
    if any(x.status == "INSUFFICIENT_INFO" for x in coverage) or any(
            c[k].status != "MET" for k in ("MP-204-DOC1", "MP-204-DOC2", "MP-204-DOC4")):
        return Recommendation.PEND, "MP-204-DOC"
    return Recommendation.DENY, "MP-204-C1-3"


def decide_pt(c: dict[str, CriterionResult]) -> tuple[Recommendation, str]:
    if c["MP-310-AUTH"].status == "NOT_APPLICABLE":
        return Recommendation.APPROVE, "MP-310-AUTH"
    if c["MP-310-MAX"].status == "NOT_MET":
        return Recommendation.DENY, "MP-310-MAX"
    if c["MP-310-C3"].status == "NOT_MET":
        return Recommendation.DENY, "MP-310-C3"
    clinical = [c["MP-310-C1"], c["MP-310-C2"], c["MP-310-C3"]]
    if all(x.status == "MET" for x in clinical):
        return Recommendation.APPROVE, "MP-310-C1-3"
    return Recommendation.PEND, "MP-310-DOC"


def missing_information(rec: Recommendation, criteria: list[CriterionResult], flags: list[ReviewerFlag]) -> list[str]:
    if rec != Recommendation.PEND:
        return []
    items = [MISSING_INFO_TEXT[x.criterion_id] for x in criteria
             if x.status in ("INSUFFICIENT_INFO", "NOT_MET") and x.criterion_id in MISSING_INFO_TEXT
             and not (x.criterion_id.startswith("MP-310") and x.status == "NOT_MET")]
    if any(fl.severity == "CRITICAL" and "cancer" in fl.message for fl in flags):
        items.append("Clarification of cancer history, which conflicts with an earlier note for this member")
    seen: set[str] = set()
    return [i for i in items if not (i in seen or seen.add(i))]


def cost_summary(rec: Recommendation, cost: dict | None, benefits: dict) -> str:
    if not cost:
        return "Cost estimate unavailable (no fee schedule or benefit match)."
    est = CostEstimate.model_validate(cost)
    parts = []
    if est.deductible_applied:
        parts.append(f"${est.deductible_applied:,.2f} deductible")
    if est.coinsurance_applied:
        parts.append(f"${est.coinsurance_applied:,.2f} coinsurance")
    if est.copay_applied:
        parts.append(f"${est.copay_applied:,.2f} in copays ({est.units} visits)")
    detail = f"${est.member_responsibility:,.2f}" + (f" ({' + '.join(parts)})" if parts else "")
    tail = (f" Deductible remaining after: ${est.deductible_remaining_after:,.2f}; "
            f"OOP max remaining after: ${est.oop_remaining_after:,.2f}.")
    if est.visits_remaining_after is not None:
        tail += f" PT/OT visits remaining this year after: {est.visits_remaining_after}."
    if rec == Recommendation.APPROVE:
        return f"Estimated member cost if rendered in-network: {detail}.{tail}"
    if rec == Recommendation.DENY:
        return (f"Not authorized: if performed in-network without authorization the claim may be denied and the "
                f"member is not billed (plan rule). For reference, the estimate had it been approved: {detail}.")
    return f"Pending determination. Estimate if approved and rendered in-network: {detail}.{tail}"


def _service_line(req: PriorAuthRequest, f: ClinicalFacts) -> str:
    codes = ", ".join(req.cpt_codes) or "not specified"
    return f"{req.requested_service or 'Requested service'} (CPT {codes})" if "CPT" not in req.requested_service \
        else req.requested_service


def draft_letter(req: PriorAuthRequest, f: ClinicalFacts, rec: Recommendation, basis: str,
                 policy_id: str, policy_title: str, criteria: list[CriterionResult],
                 missing: list[str]) -> str:
    c = _by_id(criteria)
    provider = req.ordering_provider or "Ordering Provider"
    provider_short = provider.split("(")[0].strip()
    salutation = f"Dear {provider_short},"
    member = f"{req.member_name or 'Member'} (ID: {req.member_id})" + (f", DOB {req.dob}" if req.dob else "")
    service = _service_line(req, f)
    subject = {Recommendation.APPROVE: "Notice of Approval", Recommendation.DENY: "Notice of Adverse Determination",
               Recommendation.PEND: "Request for Additional Information (Pended)"}[rec]
    L = [DRAFT_BANNER, "",
         f"Date: {review_date().isoformat()}",
         f"Subject: {subject} - Prior Authorization {req.request_id}",
         f"To: {provider}",
         f"Member: {member}",
         f"Service: {service}",
         f"Governing policy: Medical Policy {policy_id} - {policy_title}", "",
         salutation, ""]

    if rec == Recommendation.APPROVE:
        if f.service_category == ServiceCategory.PHYSICAL_THERAPY:
            first = (f.visits_completed or 0) + 1
            last = (f.visits_completed or 0) + (f.visits_requested or 0)
            L.append(f"Meridian Health Plan has approved {f.visits_requested} additional outpatient physical therapy "
                     f"visits (visits {first} through {last} of calendar year {review_date().year}) for "
                     f"{req.member_name}.")
        else:
            L.append(f"Meridian Health Plan has approved the request for {service} for {req.member_name}.")
        L += ["", "Clinical rationale:"]
        if basis == "MP-310-C1-3":
            L.append(f"The progress note dated {req.note_date} meets Medical Policy {policy_id} criteria for visits 13 and beyond:")
            for cid in ("MP-310-C1", "MP-310-C2", "MP-310-C3", "MP-310-MAX"):
                L.append(f"  - {c[cid].description}: " + "; ".join(c[cid].evidence))
        elif basis == "MP-310-AUTH":
            L.append(f"  - {'; '.join(c['MP-310-AUTH'].evidence)} (Medical Policy {policy_id}).")
        else:
            crit = c[basis]
            num = basis.split("-C")[-1]
            L.append(f"The office note dated {req.note_date} meets Medical Policy {policy_id}, Criterion {num}: "
                     f"\"{crit.policy_text.split('. ', 1)[-1]}\"")
            for e in crit.evidence:
                L.append(f"  - {e}")
        L += ["", "Authorization number and validity period: [assigned at finalization]."]
    elif rec == Recommendation.DENY:
        L.append(f"After review, the request for {service} for {req.member_name} does not meet medical "
                 f"necessity criteria in Medical Policy {policy_id}.")
        L += ["", "Clinical rationale:"]
        if basis == "MP-204-NMN1":
            L.append(f"Medical Policy {policy_id} lists as not medically necessary: \"{c['MP-204-NMN1'].policy_text.lstrip('- ')}\"")
            L.append(f"The note dated {req.note_date} documents:")
            dur = f.symptom_duration_weeks or 0
            L.append(f"  - Symptom duration of {dur:g} weeks (about {round(dur * 7)} days), below the 6-week threshold")
            exam = f"; neurological exam: {'; '.join(f.neuro_findings)}" if f.neuro_findings else ""
            L.append(f"  - No radiating symptoms{exam}")
            L.append("  - No red-flag findings")
            cons = "No physical therapy or supervised home exercise program"
            if f.analgesic_names:
                cons += f"; {', '.join(f.analgesic_names)}" + (
                    f" x {f.analgesic_weeks:g} weeks" if f.analgesic_weeks else " (duration not stated)")
            L.append(f"  - {cons}")
            L.append("  - None of the approval criteria (red flags, radiculopathy despite 6 weeks of conservative "
                     "therapy, or pre-surgical planning) is documented.")
            L += ["", "A new request may be submitted if red-flag symptoms develop, or if symptoms persist at least "
                      "6 weeks despite documented conservative therapy (physical therapy or a supervised home "
                      "exercise program plus an NSAID/analgesic trial)."]
        else:
            for x in criteria:
                if x.status == "NOT_MET":
                    L.append(f"  - Not met: {x.description}. Documented: " + "; ".join(x.evidence))
                    if x.policy_text:
                        L.append(f"    Policy {policy_id} states: \"{x.policy_text.split('. ', 1)[-1]}\"")
        L += ["", "You may request a peer-to-peer discussion with a Meridian physician reviewer. The member and "
                  "provider have the right to appeal; appeal instructions are enclosed."]
    else:
        L.append(f"The request for {service} for {req.member_name} has been pended because the submission does not "
                 f"include the documentation required by Medical Policy {policy_id}.")
        if c.get("MP-204-C1") and c["MP-204-C1"].status == "INSUFFICIENT_INFO":
            L += ["", f"The note states: \"{c['MP-204-C1'].evidence[0]}\" Under {policy_id} "
                      f"Criterion 1, a history of cancer with new back pain can support imaging, but the "
                      f"submission does not establish the back pain's onset or include an examination."]
        L += ["", "Please submit:"]
        L += [f"  {i}. {m}" for i, m in enumerate(missing, 1)]
        L += ["", "The request will be reviewed as soon as the information is received. If it is not received "
                  "within the plan's required timeframe, a determination will be made on the information available."]

    signer = "[Physician Reviewer Name], MD" if rec == Recommendation.DENY else "[Nurse Reviewer Name], RN"
    L += ["", "Sincerely,", signer, "Utilization Management, Meridian Health Plan"]
    return "\n".join(L)


def run(req: PriorAuthRequest, f: ClinicalFacts, policy_id: str, policy_title: str,
        criteria: list[CriterionResult], fin: dict, history: list[dict],
        extraction_notes: list[str], validation_warnings: list[str]) -> DecisionPacket:
    c = _by_id(criteria)
    if f.service_category == ServiceCategory.MRI_LUMBAR:
        rec, basis = decide_mri(c)
    elif f.service_category == ServiceCategory.PHYSICAL_THERAPY:
        rec, basis = decide_pt(c)
    else:
        rec, basis = Recommendation.PEND, "UNKNOWN_SERVICE"

    flags = build_flags(req, f, history, fin.get("benefits") or {}, extraction_notes, validation_warnings)
    if rec == Recommendation.DENY:
        flags.append(ReviewerFlag(severity="INFO", message=(
            "Adverse medical-necessity determinations generally require physician (medical director) review. "
            "Route this case per Meridian UM policy.")))
    if rec == Recommendation.APPROVE and c.get("MP-310-MAX") and c["MP-310-MAX"].status == "INSUFFICIENT_INFO":
        flags.append(ReviewerFlag(severity="WARNING", message="Request exceeds the annual cap; approve only the remaining visits."))
    missing = missing_information(rec, criteria, flags)
    letter = draft_letter(req, f, rec, basis, policy_id, policy_title, criteria, missing)

    severities = {fl.severity for fl in flags}
    if rec == Recommendation.PEND or "CRITICAL" in severities:
        confidence = "LOW"
    elif "WARNING" in severities:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"
    citations = [f"{policy_id} {x.criterion_id.split('-', 2)[-1]}: {x.policy_text}" for x in criteria
                 if x.policy_text and x.status in ("MET", "NOT_MET", "INSUFFICIENT_INFO")]
    return DecisionPacket(
        request_id=req.request_id, recommendation=rec, policy_id=policy_id, policy_title=policy_title,
        primary_basis=basis, criteria=criteria, flags=flags, missing_information=missing,
        cost_estimate=CostEstimate.model_validate(fin["cost"]) if fin.get("cost") else None,
        cost_summary=cost_summary(rec, fin.get("cost"), fin.get("benefits") or {}),
        letter=letter, citations=citations, confidence=confidence,
    )
