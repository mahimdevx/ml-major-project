# Compliance Check — Integrated Report vs Template & Rubric
### CSCI433/933 Assignment 2 — Shakespeare-Aware RAG System

Audit of `report/integrated/main.pdf` against (a) the report template and
(b) the grading rubric. Status key: **✓ done** · **△ depends on code/eval
re-run** (outside report writing) · **✗ action needed by you**.

---

## A. Report template — section-by-section

### Abstract — ✓
Problem, system, model/RAG approach, dataset, evaluation method, and main
findings, all present. **192 words** (target 150–200).

### I. Introduction & Problem Formulation (10 marks) — ✓
- what problem the system solves ✓ · why Shakespeare is a domain-adaptation
  challenge ✓ · intended user ✓ · what the system does ✓ · constraints
  that shaped design ✓. No generic literature review.

### II. Dataset & Preprocessing (15 marks) — ✓
- dataset structure ✓ · chunking (scene + utterance) ✓ · metadata
  (play/act/scene/speaker/source_id) ✓ · cleaning/filtering/normalisation ✓
  · supplementary material (none external; enrichment from provided fields,
  explicitly distinguished from primary text) ✓.
- II.A Dataset Schema ✓ · II.B Chunking Strategy with short-vs-long
  trade-off ✓.

### III. Methodology (20 marks) — ✓
- III.A Baseline: what it does, model/tool, information access, why it is a
  fair comparison ✓.
- III.B RAG: embedding model ✓ · retrieval/indexing ✓ · number of chunks
  (k=5) ✓ · prompt structure ✓ · generation model ✓ · how context is
  incorporated ✓.
- III.C Model Choice & Justification: SLM/deployability principles ✓;
  local-vs-hosted justification ✓.

### IV. System Design & Implementation (15 marks) — ✓ *(your section)*
- IV.A System Pipeline: all **seven** template steps explicitly enumerated
  — user query, query embedding, retrieval, prompt construction, response
  generation, **evidence display**, **logging/evaluation output** ✓ ·
  system diagram (Fig. 1) ✓.
- IV (User Interface): system/mode/k selection, validation, evidence-first
  display, stylised tagging, graceful failure, demo mode — targets the
  *usability* marks ✓.
- IV.B Implementation Details: language/libraries ✓ · repository structure
  ✓ · how to run ✓ · dependencies ✓ · cached index ✓ · limitations ✓ · no
  large code blocks in main body ✓.

### V. Evaluation Design (20 marks) — ✓
- V.A Questions: 15 total, 5 instructor + 10 group, with what the
  group questions test ✓.
- V.B Scoring Criteria: 1–5 scale with 1/3/5 defined; correctness,
  grounding, retrieval relevance, usefulness, style ✓.
- V.C Table format: compact summary in body (Table II), full table in
  appendix ✓.

### VI. Results & Analysis (20 marks) — ✓
- VI.A Baseline-vs-RAG comparison: correctness, grounding, where RAG
  helped / did not, unexpected behaviour ✓.
- VI.B Qualitative examples: concise in body; full worked examples
  (question + retrieved evidence + response + interpretation) in
  Appendix C ✓.
- VI.C Failure analysis: **four** failure cases, each with *what went
  wrong* **and** *how to improve* ✓ (template asks for ≥3).

### VII. Responsible GenAI (10 marks) — ✓ *(your section)*
Tools, uses, verification, accept/modify/reject, and ownership ✓; full
per-role log in Appendix D ✓.

### VIII. Conclusion — ✓ *(your section)*
Objectives met, what evaluation revealed, most important design choices,
what to improve with more time ✓.

### References — ✓
BibTeX via `ieeetr`; academic + model/infrastructure sources.

### Appendices — ✓ (template-aligned lettering)
A Full Evaluation Table · B Representative Prompts · C Additional Output
Examples · D GenAI Usage Log · (plus, as you requested) E Member
Contributions · F Component Limitations & Integration Findings.

---

## B. Submission requirements (spec p.6–7)

| Requirement | Status |
|---|---|
| Working system / CLI demonstrating the interaction types | ✓ `src/main.py` |
| Source repo with setup, execution, component description | ✓ `README.md` |
| LaTeX report following the template | ✓ `report/integrated/` |
| Evaluation appendix (questions, evidence, responses, scoring, errors) | ✓ Appendices A & C |
| Demonstration (5–10 min) | ✓ script in `DEMO.md` (record the video) |
| GenAI usage log | ✓ Appendix D |
| Main report ≤ 10 pages (abstract→conclusion) | ✓ **7 pages**; refs/appendices excluded |
| IEEE-style formatting | ✓ two-column IEEE-style layout |

---

## C. Rubric — how the report targets HD

- **C.2.1 Problem Formulation (HD):** precise, motivated, tied to domain
  adaptation + SLM constraints — ✓.
- **C.2.2 Dataset (HD):** clear schema, justified preprocessing, thoughtful
  chunking with trade-off — ✓.
- **C.2.3 RAG Methodology (HD):** retrieval/generation integration and
  justification, reranking rationale — ✓ in the report; △ end-to-end
  *demonstrated* integration depends on running with Gemma.
- **C.2.4 System Implementation (HD, your section):** clean, organised,
  reproducible, explicit run steps, usability features — ✓ in report and
  `src/main.py`; △ full reproducibility needs the index/build scripts (see
  §E).
- **C.2.5 Evaluation & Analysis (HD):** metrics, comparison, insightful
  success/failure analysis — ✓ as written; △ scores currently reflect the
  placeholder generator, so re-running with Gemma raises this from
  "honest but low numbers" toward HD.
- **C.2.6 Technical Report (HD, your section):** clear, well-structured,
  concise, professional, within limit — ✓.
- **C.2.7 Responsible GenAI (HD, your section):** transparent, reflective,
  critical, documented — ✓.

---

## D. Your part (Integration & Reporting Lead) — HD self-check

| Deliverable | HD evidence |
|---|---|
| System interface | Validated menus, evidence-first output, stylised tagging, graceful failure, scripted demo mode; usability argued in IV |
| Repository organisation | Spec-aligned tree, accurate `README.md`, separation of originals vs integrated report |
| Report integration | Single coherent ≤10-page report; de-duplicated preambles/bibs/GenAI; consistent naming; reconciled framings; contributions + limitations appendices |
| Demonstration prep | Timed `DEMO.md` with commands, per-mode queries, talking points, fallback |
| Report quality | IEEE style, 7-page body, no overfull boxes, no undefined refs, figures/tables/longtables render cleanly |

---

## E. Items that still need YOU (affect marks beyond report writing)

1. **✗ Insert group number + student names** in `report/integrated/main.tex`.
2. **△ Re-run `evaluate.py` with Gemma 3 4B** and update Table II + the
   Appendix A longtable (raises C.2.5; one-line change in
   `INTEGRATION_NOTES.md`). You asked to defer model work — fine; just note
   the current scores are honestly labelled as the placeholder generator.
3. **△ Include the ChromaDB index + index-build scripts** so the system is
   reproducible at assessment (spec requires runnable system); see
   `INTEGRATION_NOTES.md` §4.
4. **✗ Record the 5–10 minute demonstration** using `DEMO.md`.

The report itself is template-complete, within the page limit, and written
to HD standard. Items 2–3 are about the *system/evaluation* (not report
writing) and are flagged honestly in the report so the submission is safe
as-is.
