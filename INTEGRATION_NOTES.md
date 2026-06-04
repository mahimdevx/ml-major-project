# Integration Notes
### Shakespeare-Aware RAG System — CSCI433/933 Assignment 2

Prepared by the Integration & Reporting Lead. This records how the four
members' work was assembled into one report and repository, the
consistency decisions made during integration, and the one open item the
team should close before submission.

---

## 1. What was integrated

| Section of report          | Source / owner                  |
|----------------------------|---------------------------------|
| Abstract, Introduction (I) | Integration Lead                |
| Dataset & Preprocessing (II)| Data Engineering Lead          |
| Methodology (III)          | Model Development Lead          |
| System Design (IV)         | Integration Lead (interface, repo, run) + Model Lead (impl. detail) |
| Evaluation (V), Results (VI)| Evaluation Lead                |
| Responsible GenAI (VII)    | Integration Lead (consolidated all members) |
| Conclusion (VIII)          | Integration Lead                |
| Appendices A–C             | Integration Lead (assembled)    |

The three separate per-member `.tex` documents (each previously a full
standalone IEEE document) were merged into a single modular project under
`report/`: one `config.tex` preamble, one `main.tex` driver, one section
per file, and one merged `references.bib`. Duplicate title blocks,
preambles, bibliographies, and three overlapping GenAI sections were
de-duplicated into single canonical versions.

---

## 2. Consistency decisions

- **System naming.** Unified to *Baseline / Standard RAG / Reranked RAG*
  for the deployed system. The Evaluation section keeps *RAG (scene)* and
  *Enhanced RAG (utterance)* where those names describe the specific
  indices the batch harness compared, with a note linking the two framings.
- **Generation model.** The deployable, demonstrable system uses **Gemma 3
  4B** (this is what `src/main.py`, `baseline.py`, `rag_chatbot.py`, and
  `rag_chatbot_reranked.py` actually call via Ollama). The methodology and
  system sections state this throughout.
- **Two indices, both real.** The data layer produced both a scene-level
  index (229 chunks) and an utterance-level ChromaDB index (2,622
  utterances). The interactive system uses the utterance index; the batch
  harness compared both. The report describes both rather than hiding one.
- **LaTeX class.** The teacher's template uses `IEEEtran`; the team's
  machines and some markers may not have it installed. The report uses the
  provided `article + geometry + titlesec` config (two-column) which
  reproduces the IEEE look and compiles in any standard TeX install. Swap
  to `\documentclass[conference]{IEEEtran}` if preferred — the section
  files are class-independent.
- **No `minted`.** Standardised on `listings` so the report compiles
  without `--shell-escape`.

---

## 3. OPEN ITEM (close before submission)

**The batch evaluation used a placeholder generator, not the deployed model.**

`src/evaluate.py` sets `GENERATION_MODEL = "distilgpt2"` (its own comment
says `# replace with your group's SLM`). So the numbers in
`results/evaluation_results.csv` and the Results section describe
distilgpt2, **not** the Gemma 3 4B system we demo. Retrieval scores (5/5)
are model-independent and are fine; the correctness/grounding/usefulness
numbers under-represent the real system.

This is the single biggest risk to the *"alignment between reported
results and demonstrated behaviour"* assessment criterion. The report is
written to be **honest about this today** (Evaluation §"Configuration
note", Results, Conclusion) so it is safe to submit as-is — but re-running
with Gemma will materially raise the reported scores.

**The fix (≈10 lines).** Replace the `transformers`-pipeline generator in
`evaluate.py` with the same Ollama call the rest of the system uses:

```python
import ollama
GENERATION_MODEL = "gemma3:4b"

def _gen(prompt: str) -> str:
    return ollama.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "content", "content": prompt}],
    )["message"]["content"]
```

Then have `BaselineSystem.answer`, `RAGSystem.answer`, and
`EnhancedRAGSystem.answer` call `_gen(prompt)` instead of
`self.generator(prompt)`. Re-run `python src/evaluate.py`, then update:

- `report/results.tex` → Table `tab:results-summary` numbers + narrative;
- `report/appendix.tex` → the full-results longtable.

The Results section's failure-analysis structure is written so this re-run
**updates the numbers without changing the prose scaffolding**.

---

## 4. Repository completeness (check before submission)

The uploaded code snapshot was missing two artefacts needed to run/reproduce
end-to-end. Confirm they are included in the final repository:

- **`data/chroma_utterances/`** — the pre-built ChromaDB index the RAG
  systems load at startup. Without it, `rag_chatbot.py` /
  `rag_chatbot_reranked.py` cannot retrieve.
- **The index-build scripts** referenced by `evaluate.py`
  (`build_utterance_chroma.py` for the ChromaDB index and `build_index.py`
  for the scene-level `embeddings.npy`). These are needed to regenerate the
  indices and for the scene-level evaluation path.

If embeddings/indices are committed, recomputation at assessment time is not
required (the specification allows cached indices).

## 5. What the Integration Lead changed vs. preserved

- **Preserved, byte-for-byte:** every member's report
  (`report/Varshini/`, `report/bobby/`, `report/moinul/`) and every pipeline
  / evaluation / data script (`baseline.py`, `rag_chatbot.py`,
  `rag_chatbot_reranked.py`, `real_retriever.py`, `evaluate.py`). Verified by
  diff against the original archive.
- **Added (Integration Lead's own deliverables):** the integrated report
  (`report/integrated/`), the unified interface (`src/main.py`), the
  top-level `README.md`, `DEMO.md`, and this file.
- **Note on `src/main.py`:** the system interface is the Integration Lead's
  role; the new `main.py` supersedes the earlier basic launcher. No other
  source file was modified. `src/README.md` (detailed run notes) is left
  untouched.
- **Removed:** the stray empty file `results/koko` (not anyone's work).

## 6. Build checklist

- [ ] Insert group number and student names/numbers in `report/main.tex`.
- [ ] (Recommended) Re-run `evaluate.py` with Gemma; update result tables.
- [ ] `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`
- [ ] Confirm main body (abstract→conclusion) ≤ 10 pages.
- [ ] Record the 5–10 min demo (see `DEMO.md`).
- [ ] Include the GenAI usage log (Appendix C) — already in the report.
