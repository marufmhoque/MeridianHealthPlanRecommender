# Meridian Health Plan: Prior Authorization Assistant

An agentic decision-support prototype for utilization-management nurse reviewers. For each prior-authorization request it reads the provider note, checks the governing medical policy criterion by criterion, estimates the member's out-of-pocket cost, and drafts the provider letter. The result is a **decision packet** that a nurse verifies, edits, and finalizes.

> **The assistant never makes the determination.** Every packet is stamped `DRAFT_PENDING_NURSE_REVIEW`, every letter carries a draft banner, and the output guardrails reject any letter that claims finality. Only a licensed reviewer records the final decision.

---

## For nurse reviewers and nurse managers

Think of the assistant as a fast clinical research assistant that prepares the file before you open it. When a request arrives it does four things:

1. **Reads and summarizes the note.** It pulls out the facts that matter: symptoms and how long they have lasted, conservative treatment tried and how the patient responded, exam findings, red flags, and therapy goals and progress.
2. **Checks the policy line by line.** It finds the governing policy (MP-204 Lumbar Spine MRI, MP-310 Outpatient PT) and marks each criterion *Met*, *Not met*, or *Insufficient info*. Each mark sits next to the note evidence and the exact policy wording.
3. **Estimates the member's cost.** It looks up the member's plan, deductible, out-of-pocket maximum, and visit counts so far this year, then calculates what the member would owe.
4. **Drafts the letter.** It writes an approval, denial, or request for more information that cites the policy. You can edit the letter before you sign it.

It also points out things a busy reviewer could miss. For example, it flags when the PT dates in a note don't add up to the weeks the note claims, when two notes for the same member contradict each other, or when a request looks like a duplicate. It also reminds you when a denial needs physician review.

**What stays with you:** the clinical judgment and the final decision. The dashboard records your decision, any edits you make to the letter, and whether you overrode the AI recommendation. Those records feed future reviews and quality monitoring.

---

## Using it

Requests are not stored in this repository. You supply them and get the decision packets back.

**Dashboard (recommended).** Start the app, upload one or more request files (`.md` or `.txt`) in the sidebar, and click **Review uploaded files**. You can also paste a request. Each request appears in the review queue with:

* the AI recommendation (APPROVE / DENY / PEND), its policy basis, and a confidence level
* reviewer flags (date discrepancies, conflicting history across a member's requests, duplicates, routing reminders)
* a criterion-by-criterion policy audit next to the source note
* the member's estimated out-of-pocket cost and how it was calculated
* an editable draft provider letter and a form to record your final decision

You can download the letter, the full packet (JSON), and a summary (Markdown) for each request. **Download all results (.zip)** exports every packet, draft letter, finalized letter, the batch summary, and the evaluation.

**Command line.** Point the script at request files or folders. Results go to the folder you choose:

```bash
python scripts/run_all.py path/to/requests/ --out results/
```

The script writes `<REQUEST>.json`, `<REQUEST>_letter.txt`, `summary.md` and `evaluation.json`. If you have expected outcomes, add `--expected expected.json` (format: `{"REQ-001": {"recommendation": "APPROVE"}}`) to also score decision accuracy.

**Request format.** These are the same fields the mock requests use. A file named `req_001.md` gets the ID `REQ-001`; any other file name becomes the ID.

```text
PRIOR AUTH REQUEST - Member M1001 (Member Name), DOB 1979-03-14
Requested: MRI lumbar spine without contrast (CPT 72148)
Ordering: Provider Name, MD (Specialty)
Office note 2026-10-29:
<clinical note>
```

**Reference data the assistant uses** (in `data/`): medical policies MP-204 and MP-310, the Gold PPO benefit summary and its structured form (`plans.json`), member eligibility and accumulators (`members.json`), and a **fictional** fee schedule (`fee_schedule.json`). The mock data includes no prices, so dollar estimates for coinsurance services rely on those fictional amounts. Replace them with contracted rates.

---

## Architecture

```
          Streamlit dashboard: queue · policy audit · cost breakdown · letter editor · trace · evals
                                              │
                ┌─────────────────────────────▼──────────────────────────────┐
                │ guardrails_in  PHI redaction · member/CPT/name validation   │
                │                prompt-injection detection                   │
                └───────┬───────────────────────────────────────┬────────────┘
                        │ valid                                 │ invalid
   LangGraph   ┌────────▼─────────┐  ┌──────────────────┐  ┌────▼─────────┐
   StateGraph  │ clinical_        │→ │ policy_retriever │  │ reject_input │
               │ extractor        │  │ ChromaDB RAG +   │  └────┬─────────┘
               │ rules ∥ Claude   │  │ criterion engine │       │
               └──────────────────┘  └────────┬─────────┘       │
                                     ┌────────▼─────────┐       │
                                     │ financial_       │ tools: get_member_benefits,
                                     │ calculator       │        calculate_patient_cost
                                     └────────┬─────────┘       │
                                     ┌────────▼─────────┐       │
                                     │ decision_        │ recommendation · flags ·
                                     │ synthesizer      │ cost summary · letter
                                     └────────┬─────────┘       │
                ┌─────────────────────────────▼─────────────────▼────────────┐
                │ guardrails_out  schema · citations exist in policy · draft  │
                │                 banner · no finality claims · PEND lists gaps│
                └─────────────────────────────┬──────────────────────────────┘
                     short-term: SqliteSaver checkpoints (thread = request id)
                     long-term:  memory.db (member history, nurse decisions, overrides)
```

### Design choices

* **The model never decides whether a criterion is met.** Criterion evaluation is deterministic code over the extracted facts. Claude (when enabled) helps with extraction and with planning the tool calls. A coverage outcome stays explainable and testable, and a model error can't silently flip it.
* **The model never does arithmetic.** Dollar amounts come only from `calculate_patient_cost`. When Claude drives the financial agent, the packet uses the tool's output, not the model's text.
* **Every letter sentence traces to a source.** Letters are rendered from templates filled with note evidence, verbatim policy text, and tool output. The faithfulness metric checks that every number and quoted phrase in the letter exists in those sources.
* **Missing evidence leads to PEND, not DENY.** A request is denied only when the note positively documents a policy exclusion. Silence in the note produces a specific list of information to request.
* **Graceful degradation.** Without API keys, the whole pipeline runs offline and gives the same output every time. With a key, any Claude error, refusal, or schema mismatch falls back to the deterministic path.

### Rubric coverage

| Component | Implementation |
|---|---|
| Multi-agent orchestration | LangGraph `StateGraph` with Clinical Extractor, Policy Retriever, Financial Calculator, and Decision Synthesizer nodes, plus input and output guardrail nodes and a conditional reject route ([graph.py](src/meridian_pa/graph.py)) |
| Embeddings & vector DB | ChromaDB over criterion-level policy chunks. Uses `text-embedding-3-small` when `OPENAI_API_KEY` is set, otherwise a local hashed n-gram embedder so it runs offline ([vectorstore.py](src/meridian_pa/vectorstore.py)) |
| Model-driven tool calling | `get_member_benefits` and `calculate_patient_cost` with strict JSON-schema definitions. With `ANTHROPIC_API_KEY` set, Claude (`claude-opus-5-5`) plans the calls in a tool-use loop; otherwise a deterministic planner makes the same calls ([tools.py](src/meridian_pa/tools.py), [financial.py](src/meridian_pa/agents/financial.py)) |
| LLM extraction | Claude structured output (JSON schema) on the **PHI-redacted** note, merged with rule extraction. Disagreements on safety fields become reviewer flags ([extractor.py](src/meridian_pa/agents/extractor.py)) |
| Guardrails | PHI redaction before anything leaves the process; input validation (eligibility, name match, CPT format, empty note, injection phrases); output validation (schema, quoted policy text must exist, draft banner, no finality claims, PEND must list gaps) ([guardrails.py](src/meridian_pa/guardrails.py)) |
| LLM evaluation | Ragas-style metrics computed locally for every batch: faithfulness, answer relevance, context recall, guardrail pass, and PHI redaction. Decision accuracy is added when expected outcomes are supplied. CI gates on synthetic test cases ([evaluation.py](src/meridian_pa/evaluation.py)). Ragas or TruLens can replace these with LLM-judged versions. |
| Memory | Short-term: LangGraph `SqliteSaver` checkpoints per request, so a review can be resumed. Long-term: SQLite member history and nurse decisions, which drive duplicate detection, cross-note contradiction checks, the 12-month repeat-imaging check, and override-rate tracking ([memory.py](src/meridian_pa/memory.py)) |
| Front end & visual | Streamlit: review queue, flag banners, criterion-by-criterion audit beside the source note and retrieved policy chunks, cost breakdown chart, editable letter with final-decision capture, LangGraph trace, and evaluation dashboard ([streamlit_app.py](app/streamlit_app.py)) |
| Deployment & CI | Dockerfile + docker-compose (persistent state volume). GitHub Actions runs the tests, the evaluation gate (fails if any recommendation regresses), and a Docker build with tests inside the image ([ci.yml](.github/workflows/ci.yml)) |

---

## Quick start

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
```

```bash
python scripts/run_all.py path/to/requests/ --out results/   # review a folder of requests
```

```bash
pytest -q                                      # 30 tests (synthetic cases)
```

```bash
streamlit run app/streamlit_app.py             # dashboard at http://localhost:8501
```

```bash
docker compose up --build                      # same dashboard, in a container
```

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to enable Claude extraction and tool planning. You can also set `OPENAI_API_KEY` (and `pip install openai`) for OpenAI embeddings.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (already done) and create an app at [share.streamlit.io](https://share.streamlit.io) with **Main file path** `streamlit_app.py`.
2. Optional: under **App settings -> Secrets**, paste (see `.streamlit/secrets.toml.example`):

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

   Without it the app runs fully offline in deterministic mode. Secrets are read from `st.secrets`, falling back to environment variables, and are never written to disk. Do not set `MERIDIAN_REVIEW_DATE` on the cloud.
3. The policy index is built in memory on first load (cached for the life of the server process). Reviewer memory and checkpoints live in `.state/`, which is **ephemeral** on Streamlit Cloud: they reset whenever the app restarts or sleeps. Download results from the dashboard before closing.

## Repository layout

```
data/            policies, benefit summary, members, plans.json, fee_schedule.json (no requests)
src/meridian_pa/
  agents/        extractor.py · policy.py · financial.py · decision.py
  graph.py       LangGraph orchestration + checkpointing
  guardrails.py  PHI redaction, input/output validation
  vectorstore.py ChromaDB index + embedders
  tools.py       benefit lookup + cost calculator (+ Claude tool schemas)
  memory.py      long-term reviewer/member memory
  evaluation.py  metrics harness
  llm.py         Anthropic SDK wrapper with fallbacks
app/             Streamlit dashboard
scripts/         run_all.py (review request files from the command line)
tests/           unit + end-to-end tests on synthetic requests (tests/fixtures)
```

## Limitations and next steps

* Rule extraction is tuned to the note styles in the mock data. Real notes will need Claude extraction turned on and a larger labeled evaluation set.
* The fee schedule is fictional, and cost estimates assume in-network care with no prior claims pending.
* Approval letters omit authorization numbers and validity periods, which are assigned at finalization. Pend timeframes follow the plan's UM rules and aren't hard-coded.
* The Claude code paths (extraction and tool planning) are written to fall back safely, but they have not been run against the live API in this repository's CI. CI runs in deterministic mode.
