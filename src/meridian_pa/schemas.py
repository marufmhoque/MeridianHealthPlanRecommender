"""Typed data contracts shared by every agent in the pipeline.

Every agent reads and writes plain dicts in the LangGraph state (so the
checkpointer can serialize them), but each dict is produced from, and
validated against, one of these Pydantic models.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    APPROVE = "APPROVE"
    DENY = "DENY"
    PEND = "PEND"


class ServiceCategory(str, Enum):
    MRI_LUMBAR = "MRI_LUMBAR"
    PHYSICAL_THERAPY = "PHYSICAL_THERAPY"
    UNKNOWN = "UNKNOWN"


class PriorAuthRequest(BaseModel):
    request_id: str
    member_id: Optional[str] = None
    member_name: Optional[str] = None
    dob: Optional[str] = None
    requested_service: str = ""
    cpt_codes: list[str] = Field(default_factory=list)
    ordering_provider: Optional[str] = None
    note_date: Optional[str] = None
    note_text: str = ""
    raw_text: str = ""
    source_file: Optional[str] = None


class ClinicalFacts(BaseModel):
    """Structured facts pulled from the provider note. None = not documented."""

    service_category: ServiceCategory = ServiceCategory.UNKNOWN
    age: Optional[int] = None
    sex: Optional[str] = None
    symptom_duration_weeks: Optional[float] = None
    radiculopathy: Optional[bool] = None
    # conservative therapy
    pt_completed: Optional[bool] = None
    pt_weeks_stated: Optional[float] = None
    pt_start: Optional[str] = None
    pt_end: Optional[str] = None
    pt_weeks_by_dates: Optional[float] = None
    home_exercise_program: Optional[bool] = None
    nsaid_or_analgesic_trial: Optional[bool] = None
    analgesic_weeks: Optional[float] = None
    analgesic_names: list[str] = Field(default_factory=list)
    conservative_response: Optional[str] = None
    # red flags (True present, False documented absent, None not documented)
    cauda_equina_signs: Optional[bool] = None
    progressive_neuro_deficit: Optional[bool] = None
    infection_signs: Optional[bool] = None
    cancer_history: Optional[bool] = None
    cancer_detail: Optional[str] = None
    red_flag_evidence: list[str] = Field(default_factory=list)
    # exam
    neuro_exam_documented: bool = False
    neuro_findings: list[str] = Field(default_factory=list)
    surgical_consideration: Optional[bool] = None
    # physical therapy specific
    visits_completed: Optional[int] = None
    visits_requested: Optional[int] = None
    measurable_goals: list[str] = Field(default_factory=list)
    objective_progress: list[str] = Field(default_factory=list)
    expects_continued_improvement: Optional[bool] = None
    maintenance_only: Optional[bool] = None
    diagnosis_summary: Optional[str] = None
    extraction_method: Literal["rules", "llm", "llm+rules"] = "rules"


class PolicyChunk(BaseModel):
    chunk_id: str
    policy_id: str
    section: str
    text: str
    score: Optional[float] = None


class CriterionResult(BaseModel):
    criterion_id: str
    description: str
    status: Literal["MET", "NOT_MET", "INSUFFICIENT_INFO", "NOT_APPLICABLE"]
    evidence: list[str] = Field(default_factory=list)
    policy_text: str = ""


class CostEstimate(BaseModel):
    service_type: str
    network: str = "in_network"
    units: int = 1
    allowed_amount_total: float = 0.0
    deductible_applied: float = 0.0
    coinsurance_applied: float = 0.0
    copay_applied: float = 0.0
    member_responsibility: float = 0.0
    plan_pays: float = 0.0
    deductible_remaining_after: float = 0.0
    oop_remaining_after: float = 0.0
    visits_remaining_after: Optional[int] = None
    applies_if: str = "approved and rendered in-network"
    notes: list[str] = Field(default_factory=list)
    breakdown: list[str] = Field(default_factory=list)


class ReviewerFlag(BaseModel):
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    message: str


class DecisionPacket(BaseModel):
    """What the nurse sees. Never a determination: always a draft."""

    request_id: str
    status: Literal["DRAFT_PENDING_NURSE_REVIEW"] = "DRAFT_PENDING_NURSE_REVIEW"
    recommendation: Recommendation
    policy_id: str
    policy_title: str
    primary_basis: str
    criteria: list[CriterionResult]
    flags: list[ReviewerFlag] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    cost_estimate: Optional[CostEstimate] = None
    cost_summary: str = ""
    letter: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
