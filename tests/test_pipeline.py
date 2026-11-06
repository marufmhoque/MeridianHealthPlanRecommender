"""End-to-end tests on synthetic requests (tests/fixtures)."""
from meridian_pa.data_loader import parse_request
from meridian_pa.evaluation import evaluate_run


def test_recommendations_and_bases_match_expected(run_all, expected):
    for rid, exp in expected.items():
        p = run_all[rid]["packet"]
        assert (p["recommendation"], p["primary_basis"]) == (exp["recommendation"], exp["basis"]), rid


def test_every_packet_is_a_draft(run_all):
    for s in run_all.values():
        p = s["packet"]
        assert p["status"] == "DRAFT_PENDING_NURSE_REVIEW"
        assert p["letter"].startswith("*** DRAFT")
        assert s["output_validation"]["ok"], s["output_validation"]


def test_red_flag_letter_quotes_note_and_exam(run_all):
    letter = run_all["CASE_RED_FLAG"]["packet"]["letter"]
    assert "saddle anesthesia" in letter and "Criterion 1" in letter


def test_pend_lists_missing_documentation(run_all):
    p = run_all["CASE_INCOMPLETE"]["packet"]
    missing = " ".join(p["missing_information"]).lower()
    assert "neurological" in missing and "duration" in missing and "30 days" in missing
    assert "Please submit" in p["letter"]


def test_pt_approval_visit_math(run_all):
    p = run_all["CASE_PT_PROGRESS"]["packet"]
    assert "visits 15 through 20" in p["letter"]
    assert p["cost_estimate"]["member_responsibility"] == 6 * 40


def test_deny_routes_to_physician_and_explains_billing(run_all):
    for rid in ("CASE_ACUTE", "CASE_PT_MAINTENANCE"):
        p = run_all[rid]["packet"]
        assert "[Physician Reviewer Name], MD" in p["letter"]
        assert "not billed" in p["cost_summary"]


def test_pt_date_discrepancy_flag(reviewer):
    req = parse_request(
        "PRIOR AUTH REQUEST - Member M1001 (Jordan Ellis), DOB 1979-03-14\n"
        "Requested: MRI lumbar spine without contrast (CPT 72148)\nOrdering: Dr A\nOffice note 2026-10-30:\n"
        "50M with low back pain radiating down left leg x 8 weeks. Completed 6 weeks PT (2026-09-01 to 2026-10-05) "
        "with minimal improvement. Ibuprofen x 6 weeks, partial relief. Exam: positive straight leg raise left, "
        "diminished left ankle reflex. No fever. No cancer history.", "T-PT")
    msgs = [f["message"] for f in reviewer.review(req)["packet"]["flags"]]
    assert any("span 4.9 weeks" in m for m in msgs)


def test_member_memory_flags_contradiction_and_duplicate(reviewer):
    head = "PRIOR AUTH REQUEST - Member M1001 (Jordan Ellis)\nRequested: MRI lumbar spine (CPT 72148)\nOrdering: Dr A\n"
    reviewer.review(parse_request(head + "Office note 2026-11-01:\nBack pain x 3 weeks. No cancer history.", "T-A"))
    st = reviewer.review(parse_request(head + "Note: Back pain. History of breast cancer (2019).", "T-B"))
    flags = st["packet"]["flags"]
    assert any(f["severity"] == "CRITICAL" and "T-A" in f["message"] for f in flags)
    assert any("duplicate" in f["message"] and "T-A" in f["message"] for f in flags)
    assert st["packet"]["recommendation"] == "PEND"


def test_evaluation_metrics(run_all, expected):
    agg = evaluate_run(run_all, expected)["aggregate"]
    assert agg["decision_accuracy"] == 1.0
    assert agg["faithfulness"] == 1.0
    assert agg["answer_relevance"] == 1.0
    assert agg["guardrail_pass_rate"] == 1.0
    assert agg["phi_redaction"] == 1.0
    assert agg["context_recall"] >= 0.75


def test_evaluation_without_expected_labels(run_all):
    agg = evaluate_run(run_all)["aggregate"]
    assert agg["decision_accuracy"] is None and agg["faithfulness"] == 1.0


def test_checkpointed_state_is_resumable(reviewer, requests):
    reviewer.review(requests["CASE_ACUTE"])
    snap = reviewer.graph.get_state({"configurable": {"thread_id": "CASE_ACUTE"}})
    assert snap.values["packet"]["recommendation"] == "DENY"
    assert [t["node"] for t in snap.values["trace"]][-1] == "guardrails_out"


def test_nurse_decision_feeds_repeat_imaging_check(reviewer, requests):
    reviewer.review(requests["CASE_CONSERVATIVE"])
    reviewer.memory.save_review("CASE_CONSERVATIVE", "APPROVE", "RN Test", "letter", "", "APPROVE")
    req = parse_request("PRIOR AUTH REQUEST - Member M1001 (Jordan Ellis)\nRequested: MRI lumbar spine (CPT 72148)\n"
                        "Ordering: Dr A\nNote: Back pain persists, repeat MRI requested.", "T-REPEAT")
    st = reviewer.review(req)
    nmn2 = next(c for c in st["packet"]["criteria"] if c["criterion_id"] == "MP-204-NMN2")
    assert nmn2["status"] == "INSUFFICIENT_INFO"
