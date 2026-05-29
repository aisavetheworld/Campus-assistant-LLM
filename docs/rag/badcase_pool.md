# Badcase Pool — Project 2 RAG

12 representative failure cases from retrieval, generation, and the
validator system. Each entry follows the same template: **Query →
Symptom → Diagnosis → Fix → Status.**

Use this list as a starting point for regression testing and for
demoing what the constraint system catches.

---

## Section A — Retrieval failures (terminology mismatch)

These three queries were zero-recall before query-expansion v2.
All rescued by adding 7 QE triggers (commit `30ac6ad`).

### B-1. Dorm cost — "cost to live" vs "Contract Rates"

| | |
|---|---|
| **Query** | "How much does it cost to live in the on-campus dorms at UCSD?" |
| **Eval id** | `rag_eval_024` (housing) |
| **Symptom** | R@5 = 0. Top-5 was `fas_0034`, `fas_0041`, `hdh_ug_0037`, `ucship_coverage_001`, `ucship_0013` — all cost-of-attendance / insurance pages. The actual rate table (`ucsd_housing_assignment_002`) was nowhere. |
| **Diagnosis** | The target page is titled *"HDH Undergrad Housing — Contract Rates"*. Neither dense ("cost to live") nor BM25 ("dorms" not in the page) reached it. |
| **Fix** | QE trigger: `cost to live → housing contract rates, housing rates and services, room and board`. |
| **Result** | R@5 = 1.0; `ucsd_housing_assignment_002` now rank 2. |
| **Status** | ✅ Fixed in QE v2. |

### B-2. Summer F-1 enrollment — "summer quarter" vs "vacation term"

| | |
|---|---|
| **Query** | "I've already completed three quarters at UCSD. Do I have to enroll full-time during summer quarter to keep my F-1 status?" |
| **Eval id** | `rag_eval_030` (international_students) |
| **Symptom** | R@5 = 0. Retrieved CPT pages, unrelated ISEO pages. |
| **Diagnosis** | `ucsd_iseo_visa_status_005` calls summer the "vacation term" / "annual vacation". The user said "summer quarter". Zero lexical overlap on the discriminative phrase. |
| **Fix** | QE trigger: `summer quarter / summer F-1 → vacation term, annual vacation, summer session enrollment`. |
| **Result** | R@5 = 1.0; target now rank 3. |
| **Status** | ✅ Fixed in QE v2. |

### B-3. Federal loans — "federal loans" vs "Federal Direct Loans"

| | |
|---|---|
| **Query** | "What federal loans are available to UCSD students and how do I apply?" |
| **Eval id** | `rag_eval_054` (financial_aid) |
| **Symptom** | R@5 = 0. Retrieved 5 other FAS pages (none about direct loans). |
| **Diagnosis** | Target page (`ucsd_fas_0050`) titled "Federal Direct Loans Program". "Federal loans" is a near-synonym but BM25 didn't match strongly and dense was pulled toward general "loans" overview pages. |
| **Fix** | QE trigger: `federal loans / federal loan → federal direct loans, subsidized loan, unsubsidized loan`. |
| **Result** | R@5 = 1.0; both expected pages (`fas_0050`, `fas_0051`) in top-2. |
| **Status** | ✅ Fixed in QE v2. |

---

## Section B — Generation failures (caught by constraints)

These are the failures the post-hoc validators catch. Without the
constraint system, they would reach the user.

### B-4. Base hallucinated "guarantee" — overconfident promise

| | |
|---|---|
| **Query** | "What are the requirements to be eligible for on-campus housing at UCSD?" |
| **Eval id** | `rag_answer_eval_040` (housing, 45-q v1) |
| **Symptom (no constraints)** | Base answer used the word *guarantee* in a promise context: *"…the university will guarantee your housing assignment…"* |
| **Diagnosis** | The retrieved chunks discuss "housing guarantee" (a UCSD program name for first-year students), and the model lifted the word into a promise. `_GUARANTEE_PATTERN` triggered `no_absolute_promise`. |
| **Fix** | `no_absolute_promise` validator + fix-hint retry. Retry prompt added *"Do not use absolute-promise phrases: ['guarantee']."* |
| **Result** | Base + constraints → 45/45 (was 40/45). |
| **Status** | ✅ Caught in 95-q eval. |

### B-5. Base "you are eligible for OPT" — forbidden eligibility claim

| | |
|---|---|
| **Query** | "I'm finishing my bachelor's degree in three months. What are the basic eligibility requirements for post-completion OPT?" |
| **Eval id** | `rag_answer_eval_019` (international_students, 45-q v1) |
| **Symptom (no constraints)** | Base answer said *"Based on this, you are eligible for OPT…"* — making a per-student determination ISEO is supposed to make. |
| **Diagnosis** | `forbidden_claims = ['you are eligible for OPT', ...]` in eval seed. `no_forbidden_claims` validator caught it. |
| **Fix** | Retry prompt added *"Do not assert: ['you are eligible for OPT']. Use conditional language instead."* |
| **Result** | Base + constraints → answer becomes "To be eligible you typically need to… Contact ISEO to confirm." |
| **Status** | ✅ Caught in 95-q eval. |

### B-6. Base echoed the query's "week 5" — token leakage as hallucination

| | |
|---|---|
| **Query** | "I'm a graduate student and want to change my grading option to S/U in **week 5**. What do I need to do?" |
| **Eval id** | `rag_answer_eval_024` (course_enrollment, 45-q v1) |
| **Symptom (no constraints)** | Base answer repeated *"…before the end of Week 6 of the quarter. Since you are considering making this change in Week 5…"* |
| **Diagnosis** | "Week 5" was in the user's question but not in retrieved context. `no_hallucinated_deadline` flagged "week 5" as not present in sources. False positive in intent (model wasn't asserting a deadline) but real signal: model echoes user vocabulary which may then be misread as policy. |
| **Fix** | The validator catches it regardless of intent. Retry hint: *"Do not state these dates/weeks (not in sources): ['week 5']."* |
| **Result** | Caught in 95-q eval. **Open improvement F8:** could allow query-echoed dates without flagging. |
| **Status** | ✅ Caught; known false-positive class. |

### B-7. DPO hallucinated "week 4" / "week 9" — true hallucination

| | |
|---|---|
| **Query** | "What is the last day I can drop a course without a W on my transcript this quarter?" |
| **Eval id** | `rag_answer_eval_004` / `006` (course_enrollment, 45-q v1) |
| **Symptom (no constraints)** | DPO answered with *"…by the end of Week 4 of the quarter…"* — week 4 was nowhere in retrieved chunks. (Week 4 is the *actual* UCSD policy; the model is "correctly" using training-data knowledge.) |
| **Diagnosis** | `no_hallucinated_deadline` correctly flagged because the assertion isn't traceable to retrieved sources. From the user's perspective: the model is right but the system cannot verify. |
| **Fix** | Retry rephrases ("by the policy deadline stated on the academic calendar — check the link") without inventing a number. |
| **Result** | DPO + constraints → 45/45 (was 43/45). |
| **Status** | ✅ Caught; correct behavior even though the *fact* is right. |

---

## Section C — Validator-system failures (known gaps)

These are cases where the validator itself is wrong, not the model.

### B-8. DPO empty-body answer that passed all 11 checks

| | |
|---|---|
| **Query** | "I want to drop below full-time enrollment this quarter due to a medical issue. Will this affect my F-1 status?" |
| **Eval id** | `rag_answer_eval_002` (international_students, 45-q v1) |
| **Symptom** | DPO output was literally just `Sources:\n- Rcl — Reduced Course Loads (https://…)` — no answer body. Eval scored **11/11**. |
| **Diagnosis** | `answer_not_empty` only tests whitespace; `cites_source` saw the Sources header; `uses_retrieved_context` saw the source titles contain content words. Vacuous pass on everything else. |
| **Fix planned** | Add `check_answer_has_body`: require ≥30 non-whitespace chars **before** the `Sources:` header. |
| **Status** | ⚠️ Open (F9). Defer ~30 min patch. |

### B-9. Fallback message fails `cites_source` by design

| | |
|---|---|
| **Query** | Any low-confidence or retry-exhausted query — e.g. `rag_eval_087` (UC SHIP on leave). |
| **Symptom** | Pipeline correctly returns fallback message. Eval reports `cites_source` failed (no `Sources:` section in the fallback). |
| **Diagnosis** | The fallback is *"I could not verify this from the retrieved official sources…"* — by design it cites nothing because there is nothing reliable to cite. The validator does not distinguish a fallback from a normal answer. |
| **Fix planned** | Either (a) exempt fallback messages from grounding-style checks, or (b) append a "Sources retrieved (may be relevant):" section to the fallback so users still get pointers. |
| **Result** | 5/95 queries (DPO and Base) show this artifact. **All 5 reported failures per model are this class.** |
| **Status** | ⚠️ Open (F13). With fix: both models hit 95/95 (100%). |

### B-10. Reranker active demotion of correct chunks

| | |
|---|---|
| **Query** | All 95 queries when MiniLM or BGE rerank is enabled. |
| **Symptom** | R@5 drops from 0.837 → 0.67-0.71 across all rerank configurations. Zero-recall@5 grows from 3 → 15-20. |
| **Diagnosis** | Candidate cR@20 = 0.958 — the hybrid pool is already saturated. Off-the-shelf rerankers are trained on MS-MARCO / general web, not UCSD policy text, so they actively demote correct chunks. |
| **Fix** | Keep `rerank=False` as default. Code retained for domain-finetuning later. |
| **Status** | ✅ Documented; do not re-enable without a fine-tuned reranker. |

---

## Section D — Retry rescue cases (system working)

### B-11. Retry caught a citation-format slip

| | |
|---|---|
| **Query** | "What is the deadline to add a course this quarter, and do I need an authorization code?" |
| **Eval id** | `rag_eval_011` (course_enrollment) — retried by both DPO and Base. |
| **Symptom on attempt 1** | Answer body was good but Sources section was missing or used a free-form list that didn't match retrieved titles. `cites_source` failed. |
| **Retry hint injected** | *"In the Sources section, list the exact source titles or URLs from the retrieved context."* |
| **Result** | Attempt 2 passed all 11 checks for both models. |
| **Status** | ✅ Shows the retry+hint mechanism rescuing format mistakes, not just content. |

### B-12. Pre-gen fallback on a "weird" query (production behavior)

| | |
|---|---|
| **Query** | "Can I bring my pet lizard to live in the dorms with me?" (demo example, not in eval) |
| **Symptom** | Top hybrid score < 0.35. Pipeline returns fallback without spending GPU time. |
| **Diagnosis** | Working as designed — when retrieval has no high-confidence match the pipeline refuses to guess. Better than letting the model freestyle "pets are allowed if registered" hallucinations. |
| **Status** | ✅ Demonstrates the pre-gen confidence gate. Use this in interviews. |

---

## Summary table

| # | Class | Example id | Resolved? |
|---|-------|-----------|-----------|
| B-1 | retrieval — terminology mismatch | rag_eval_024 | ✅ QE v2 |
| B-2 | retrieval — terminology mismatch | rag_eval_030 | ✅ QE v2 |
| B-3 | retrieval — terminology mismatch | rag_eval_054 | ✅ QE v2 |
| B-4 | generation — absolute promise | rag_answer_eval_040 | ✅ validator + retry |
| B-5 | generation — forbidden eligibility | rag_answer_eval_019 | ✅ validator + retry |
| B-6 | generation — query-echoed date | rag_answer_eval_024 | ✅ validator + retry (false-positive class) |
| B-7 | generation — real hallucination | rag_answer_eval_004/006 | ✅ validator + retry |
| B-8 | validator gap — empty body | rag_answer_eval_002 | ⚠️ Open (F9) |
| B-9 | validator gap — fallback cites_source | 5 queries × 2 models | ⚠️ Open (F13) |
| B-10 | reranker degrades retrieval | all 95 | ✅ disabled |
| B-11 | retry rescue (format) | rag_eval_011 | ✅ retry hint |
| B-12 | pre-gen fallback (weird query) | demo example | ✅ confidence gate |

**Coverage:** 9 fixed, 2 open (both ~30-min patches), 1 (reranker) explicitly out of scope.
