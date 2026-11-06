"""Policy Retriever agent: selects the governing policy, retrieves the
relevant chunks from the vector index, and evaluates each criterion.

Criterion status always means "favorable to coverage":
MET = requirement satisfied, NOT_MET = requirement fails (or an exclusion
applies), INSUFFICIENT_INFO = the note is silent, NOT_APPLICABLE = not relevant.
Evaluation is deterministic code over the extracted facts; the model never
decides whether a criterion is met.
"""
from __future__ import annotations

from datetime import date

from .. import config
from ..schemas import ClinicalFacts, CriterionResult, PolicyChunk, PriorAuthRequest, ServiceCategory
from ..vectorstore import PolicyIndex

POLICY_FOR_SERVICE = {
    ServiceCategory.MRI_LUMBAR: ("MP-204", "Lumbar Spine MRI"),
    ServiceCategory.PHYSICAL_THERAPY: ("MP-310", "Outpatient Physical Therapy"),
}

# Criterion id -> chunk id holding its exact policy wording.
CRITERION_CHUNKS = {
    "MP-204-C1": "MP-204-02", "MP-204-C2": "MP-204-03", "MP-204-C3": "MP-204-04",
    "MP-204-NMN1": "MP-204-05", "MP-204-NMN2": "MP-204-06",
    "MP-204-DOC1": "MP-204-07", "MP-204-DOC2": "MP-204-08", "MP-204-DOC3": "MP-204-09", "MP-204-DOC4": "MP-204-10",
    "MP-310-AUTH": "MP-310-02", "MP-310-C1": "MP-310-03", "MP-310-C2": "MP-310-04",
    "MP-310-C3": "MP-310-05", "MP-310-MAX": "MP-310-06",
}


def build_query(req: PriorAuthRequest, f: ClinicalFacts) -> str:
    parts = [req.requested_service]
    if f.service_category == ServiceCategory.MRI_LUMBAR:
        if f.radiculopathy:
            parts.append("radiculopathy persisting despite conservative therapy physical therapy NSAIDs")
        if f.symptom_duration_weeks is not None and f.symptom_duration_weeks < 6:
            parts.append("low back pain under 6 weeks without red flags")
        if f.cancer_history:
            parts.append("history of cancer with new back pain red-flag")
        if not f.neuro_exam_documented or not f.symptom_duration_weeks:
            parts.append("required documentation office note duration neurological exam")
    else:
        parts.append("visits 13+ authorization measurable functional goals objective progress continued improvement maximum visits")
    return " ".join(parts)


def retrieve(index: PolicyIndex, req: PriorAuthRequest, f: ClinicalFacts, k: int = 6) -> tuple[str, str, list[PolicyChunk]]:
    policy_id, title = POLICY_FOR_SERVICE.get(f.service_category, ("UNKNOWN", "Unknown service"))
    if policy_id == "UNKNOWN":
        return policy_id, title, []
    return policy_id, title, index.query(build_query(req, f), policy_id=policy_id, k=k)


def _cr(index: PolicyIndex, cid: str, desc: str, status: str, evidence: list[str]) -> CriterionResult:
    chunk = index.chunks.get(CRITERION_CHUNKS.get(cid, ""))
    return CriterionResult(criterion_id=cid, description=desc, status=status, evidence=evidence,
                           policy_text=chunk.text if chunk else "")


def _days_since(note_date: str | None) -> int | None:
    if not note_date:
        return None
    return (config.review_date() - date.fromisoformat(note_date)).days


def evaluate_mri(index: PolicyIndex, req: PriorAuthRequest, f: ClinicalFacts, history: list[dict]) -> list[CriterionResult]:
    out: list[CriterionResult] = []
    dur = f.symptom_duration_weeks

    # Criterion 1 - red flags
    flags_present = {
        "suspected cauda equina signs": f.cauda_equina_signs,
        "progressive neurological deficit": f.progressive_neuro_deficit,
        "suspected infection": f.infection_signs,
    }
    present = [k for k, v in flags_present.items() if v]
    if present:
        ev = [f"Documented: {', '.join(present)}"] + [f"Note: \"{s}\"" for s in f.red_flag_evidence]
        if f.neuro_findings:
            ev.append("Exam: " + "; ".join(f.neuro_findings))
        out.append(_cr(index, "MP-204-C1", "Red-flag symptoms", "MET", ev))
    elif f.cancer_history:
        if dur is not None:
            out.append(_cr(index, "MP-204-C1", "Red-flag symptoms (history of cancer with new back pain)", "MET",
                           [f.cancer_detail or "Cancer history documented", f"Back pain duration {dur:g} weeks"]))
        else:
            out.append(_cr(index, "MP-204-C1", "Red-flag symptoms (history of cancer with new back pain)",
                           "INSUFFICIENT_INFO",
                           [f.cancer_detail or "Cancer history documented",
                            "Onset/duration of back pain not documented, so 'new back pain' cannot be confirmed"]))
    elif all(v is False for v in [*flags_present.values(), f.cancer_history]):
        out.append(_cr(index, "MP-204-C1", "Red-flag symptoms", "NOT_MET",
                       ["Note documents absence of red flags (no bowel/bladder change, fever, or cancer history)"]))
    else:
        documented_absent = [k for k, v in {**flags_present, "cancer history": f.cancer_history}.items() if v is False]
        out.append(_cr(index, "MP-204-C1", "Red-flag symptoms", "NOT_MET",
                       ["No red flag documented as present"] +
                       ([f"Documented absent: {', '.join(documented_absent)}"] if documented_absent else [])))

    # Criterion 2 - radiculopathy >= 6 weeks despite conservative therapy
    ev: list[str] = []
    pt_ok = bool(f.pt_completed and (f.pt_weeks_stated or 0) >= 6) or bool(f.home_exercise_program)
    analgesic_ok = bool(f.nsaid_or_analgesic_trial)
    if f.radiculopathy:
        ev.append(f"Radiculopathy documented; duration {dur:g} weeks" if dur else "Radiculopathy documented; duration not stated")
        if f.neuro_findings:
            ev.append("Exam: " + "; ".join(f.neuro_findings))
        if f.pt_completed:
            pt = f"PT {f.pt_weeks_stated:g} weeks stated" if f.pt_weeks_stated else "PT documented"
            if f.pt_start:
                pt += f" ({f.pt_start} to {f.pt_end}, {f.pt_weeks_by_dates:g} weeks by dates)"
            ev.append(pt)
        if analgesic_ok:
            ev.append("Analgesic trial: " + ", ".join(f.analgesic_names)
                      + (f" x {f.analgesic_weeks:g} weeks" if f.analgesic_weeks else ""))
        if f.conservative_response:
            ev.append("Response: " + f.conservative_response)
        if dur is not None and dur >= 6 and pt_ok and analgesic_ok:
            status = "MET"
        elif dur is None or f.pt_completed is None or f.nsaid_or_analgesic_trial is None:
            status = "INSUFFICIENT_INFO"
        else:
            status = "NOT_MET"
    else:
        status = "NOT_MET" if f.radiculopathy is False else "INSUFFICIENT_INFO"
        ev.append("No radiating symptoms (documented)" if f.radiculopathy is False else "Radiculopathy not documented")
        if not (f.pt_completed or f.home_exercise_program):
            ev.append("No physical therapy or supervised home exercise documented")
        if f.nsaid_or_analgesic_trial and f.analgesic_names:
            ev.append(f"Analgesic: {', '.join(f.analgesic_names)}"
                      + (f" x {f.analgesic_weeks:g} weeks" if f.analgesic_weeks else " (duration not stated)"))
    out.append(_cr(index, "MP-204-C2", "Radiculopathy >= 6 weeks despite documented conservative therapy", status, ev))

    # Criterion 3 - pre-surgical planning
    out.append(_cr(index, "MP-204-C3", "Pre-surgical planning documented by spine surgeon",
                   "MET" if f.surgical_consideration else "NOT_MET",
                   ["Spine surgeon documents surgery under consideration"] if f.surgical_consideration
                   else ["No spine surgeon evaluation documented"]))

    # Exclusion 1 - uncomplicated LBP < 6 weeks without red flags
    if dur is not None and dur < 6 and not f.radiculopathy and not present and not f.cancer_history:
        out.append(_cr(index, "MP-204-NMN1", "Not excluded as uncomplicated low back pain < 6 weeks", "NOT_MET",
                       [f"Duration {dur:g} weeks (< 6)", "No radiculopathy", "No red flags"]))
    elif dur is None:
        out.append(_cr(index, "MP-204-NMN1", "Not excluded as uncomplicated low back pain < 6 weeks",
                       "INSUFFICIENT_INFO", ["Symptom duration not documented"]))
    else:
        out.append(_cr(index, "MP-204-NMN1", "Not excluded as uncomplicated low back pain < 6 weeks", "MET",
                       [f"Duration {dur:g} weeks" + (" with radiculopathy" if f.radiculopathy else "")]))

    # Exclusion 2 - repeat imaging within 12 months
    prior_mri = [h for h in history if h.get("service_category") == ServiceCategory.MRI_LUMBAR.value
                 and h.get("nurse_decision") == "APPROVE"]
    if prior_mri:
        out.append(_cr(index, "MP-204-NMN2", "Not a repeat MRI within 12 months without change in status",
                       "INSUFFICIENT_INFO",
                       [f"Lumbar MRI previously approved for this member ({h['request_id']})" for h in prior_mri]
                       + ["Confirm whether imaging was performed and whether clinical status changed"]))
    else:
        out.append(_cr(index, "MP-204-NMN2", "Not a repeat MRI within 12 months without change in status", "MET",
                       ["No prior lumbar MRI authorization on file for this member"]))

    # Required documentation
    days = _days_since(req.note_date)
    if days is None:
        out.append(_cr(index, "MP-204-DOC1", "Office note dated within 30 days", "INSUFFICIENT_INFO", ["Note is undated"]))
    else:
        out.append(_cr(index, "MP-204-DOC1", "Office note dated within 30 days", "MET" if 0 <= days <= 30 else "NOT_MET",
                       [f"Note dated {req.note_date} ({days} days before review)"]))
    out.append(_cr(index, "MP-204-DOC2", "Duration of symptoms documented",
                   "MET" if dur is not None else "INSUFFICIENT_INFO",
                   [f"{dur:g} weeks"] if dur is not None else ["Not documented"]))
    cons_documented = bool((f.pt_completed or f.home_exercise_program or f.nsaid_or_analgesic_trial))
    cons_complete = bool(f.pt_completed and f.pt_start and f.conservative_response) or bool(
        f.nsaid_or_analgesic_trial and f.analgesic_weeks and f.conservative_response)
    out.append(_cr(index, "MP-204-DOC3", "Conservative therapy type, dates, and response documented",
                   "MET" if cons_complete else "INSUFFICIENT_INFO",
                   [f.conservative_response or ("Partial: " + ", ".join(f.analgesic_names) if cons_documented else "Not documented")]))
    out.append(_cr(index, "MP-204-DOC4", "Neurological exam findings documented",
                   "MET" if f.neuro_exam_documented else "INSUFFICIENT_INFO",
                   ["; ".join(f.neuro_findings)] if f.neuro_findings else ["No neurological exam documented"]))
    return out


def evaluate_pt(index: PolicyIndex, req: PriorAuthRequest, f: ClinicalFacts, benefits: dict) -> list[CriterionResult]:
    out: list[CriterionResult] = []
    ytd = f.visits_completed if f.visits_completed is not None else benefits.get("pt_visits_ytd", 0)
    req_visits = f.visits_requested or 0
    first, last = ytd + 1, ytd + req_visits
    if last <= 12:
        out.append(_cr(index, "MP-310-AUTH", "Authorization required (visits 13+)", "NOT_APPLICABLE",
                       [f"Requested visits {first}-{last} fall within the first 12; no authorization required"]))
    else:
        out.append(_cr(index, "MP-310-AUTH", "Authorization required (visits 13+)", "MET",
                       [f"{ytd} visits completed; request covers visits {first}-{last}"]))
    measurable = [g for g in f.measurable_goals if any(ch.isdigit() for ch in g) or "independent" in g.lower()]
    out.append(_cr(index, "MP-310-C1", "Measurable functional goals documented",
                   "MET" if measurable else ("NOT_MET" if f.measurable_goals else "INSUFFICIENT_INFO"),
                   [f"Goals: {'; '.join(f.measurable_goals)}"] if f.measurable_goals else ["No goals documented"]))
    out.append(_cr(index, "MP-310-C2", "Objective progress since start of care",
                   "MET" if f.objective_progress else "INSUFFICIENT_INFO",
                   f.objective_progress or ["No objective measures documented"]))
    if f.maintenance_only:
        st, ev = "NOT_MET", ["Therapy described as maintenance/plateau"]
    elif f.expects_continued_improvement:
        st, ev = "MET", ["Therapist documents expected continued gains"]
    else:
        st, ev = "INSUFFICIENT_INFO", ["Expected continued improvement not stated"]
    out.append(_cr(index, "MP-310-C3", "Expected continued improvement (not maintenance)", st, ev))
    remaining = max(0, 30 - ytd)
    out.append(_cr(index, "MP-310-MAX", "Within 30-visit annual PT/OT maximum",
                   "MET" if last <= 30 else ("NOT_MET" if remaining == 0 else "INSUFFICIENT_INFO"),
                   [f"{ytd} completed + {req_visits} requested = {last} of 30"]
                   + ([f"Only {remaining} visits remain under the cap"] if last > 30 else [])))
    return out
