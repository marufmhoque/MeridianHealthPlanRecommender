"""Guardrails: PHI redaction, input validation, and output validation.

* ``redact_phi`` runs before any text leaves the process (LLM calls, embedding
  calls, logs). Names, DOBs, member IDs, phone numbers, emails, SSNs, MRNs and
  street addresses are replaced with stable placeholders.
* ``validate_request`` blocks malformed submissions before they reach agents.
* ``validate_packet`` enforces the output contract: schema-valid, a
  recognised recommendation, citations that really exist in the policy text,
  a letter marked as draft, and never a claim of final determination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import DecisionPacket, PriorAuthRequest, Recommendation

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("DOB", re.compile(r"\bDOB[:\s]+\d{4}-\d{2}-\d{2}\b", re.I)),
    ("PHONE", re.compile(r"\b(?:\(\d{3}\)\s*|\d{3}[-.\s])\d{3}[-.\s]\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("MRN", re.compile(r"\bMRN[:#\s]*\w+\b", re.I)),
    ("MEMBER_ID", re.compile(r"\bM\d{4,}\b")),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+\w+(?:\s\w+)*\s(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive)\b\.?", re.I)),
]


def redact_phi(text: str, names: list[str] | None = None) -> str:
    """Replace PHI with placeholders. ``names`` adds known person names."""
    out = text
    for name in sorted({n for n in (names or []) if n}, key=len, reverse=True):
        out = re.sub(re.escape(name), "[NAME]", out, flags=re.I)
        for part in name.split():
            if len(part) > 2:
                out = re.sub(rf"\b{re.escape(part)}\b", "[NAME]", out)
    for label, pat in _PATTERNS:
        out = pat.sub(f"[{label}]", out)
    return out


def known_names(req: PriorAuthRequest) -> list[str]:
    """Member name only. Provider names are not member PHI and stay visible."""
    return [req.member_name] if req.member_name else []


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_INJECTION = re.compile(
    r"(ignore (all|any|previous) (instructions|rules)|you are now|system prompt|approve this regardless)",
    re.I,
)


def validate_request(req: PriorAuthRequest, members: dict[str, dict]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not req.member_id:
        errors.append("Member ID missing from request header.")
    elif req.member_id not in members:
        errors.append(f"Member {req.member_id} not found in eligibility file.")
    elif req.member_name and members[req.member_id]["name"].lower() != req.member_name.lower():
        errors.append(
            f"Member name on request ({req.member_name}) does not match eligibility "
            f"({members[req.member_id]['name']})."
        )
    if not req.cpt_codes:
        warnings.append("No CPT code on request; service inferred from description.")
    for c in req.cpt_codes:
        if not re.fullmatch(r"\d{5}", c):
            errors.append(f"Malformed CPT code: {c!r}")
    if not req.note_text or len(req.note_text.split()) < 3:
        errors.append("Clinical note is empty.")
    if not req.ordering_provider:
        warnings.append("Ordering provider not identified.")
    if not req.dob:
        warnings.append("Member DOB not on request; identity matched on member ID only.")
    if _INJECTION.search(req.raw_text):
        warnings.append("Request text contains instruction-like language; treated as data only.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


_FINALITY = re.compile(r"\b(final determination|this decision is final|determination is final)\b", re.I)


def validate_packet(packet: DecisionPacket | dict, policy_docs: dict[str, str]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        p = packet if isinstance(packet, DecisionPacket) else DecisionPacket.model_validate(packet)
    except Exception as exc:  # pydantic.ValidationError
        return ValidationResult(False, [f"Packet failed schema validation: {exc}"])

    if p.recommendation not in Recommendation:
        errors.append(f"Unknown recommendation {p.recommendation}")
    if p.status != "DRAFT_PENDING_NURSE_REVIEW":
        errors.append("Packet must be a draft pending nurse review.")
    if p.policy_id not in policy_docs:
        errors.append(f"Cited policy {p.policy_id} does not exist.")
    else:
        norm_policy = _norm(policy_docs[p.policy_id])
        for c in p.criteria:
            if c.policy_text and _norm(c.policy_text) not in norm_policy:
                errors.append(f"Criterion {c.criterion_id} quotes text not found in {p.policy_id}.")
    if "DRAFT" not in p.letter.upper():
        errors.append("Letter is not watermarked as a draft.")
    if _FINALITY.search(p.letter):
        errors.append("Letter claims finality; only the nurse reviewer can finalize.")
    if p.policy_id not in p.letter:
        errors.append("Letter does not cite the governing policy.")
    if p.recommendation == Recommendation.PEND and not p.missing_information:
        errors.append("PEND recommendation must list the missing information.")
    if p.recommendation == Recommendation.DENY and not any(c.status == "NOT_MET" for c in p.criteria):
        warnings.append("DENY without any NOT_MET criterion; check reasoning.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("*", "")).strip().lower()
