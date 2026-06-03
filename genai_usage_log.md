# GenAI Usage Log

**CSCI433/933 Assignment 2 --- Shakespeare-Aware RAG System**

This log records every documented use of generative AI during the
project. Per the assignment's transparency requirement, this is also
reproduced as Appendix C of the submitted report
(`report/final/report.tex`).

**Tool used:** Claude (Anthropic, https://claude.ai). No other
generative AI tool was used by the group.

**General method:** Claude was consulted as a reference and design
sounding board, not as a code generator. For every consultation, we
documented the prompt, summarised the response in our own words,
verified the output against our actual system or against
authoritative sources, and recorded what we accepted, modified, or
rejected.

---

## Model and RAG Lead Entries

### M1. Comparing available small language models

- **Prompt:** "Give me a list of free to download small language
  models that I can use to build a RAG pipeline. It must run locally
  on any standard computer. Give me specifications like how many
  parameters, context window, etc., and a scale to judge their
  speciality."
- **Useful output:** Structured comparison of seven locally runnable
  models (Gemma 3 4B, Mistral 7B Q4, Llama 3.2 3B, Phi-3.5 Mini,
  Qwen2.5 3B, TinyLlama 1.1B, Gemma 4 E4B) with parameter count,
  context window, Q4 RAM estimate, and rough ratings.
- **Verification:** RAM estimates cross-checked against the Ollama
  model library page.
- **What we did:** Used as a shortlist source; did not paste into the
  report. The final candidate table in the report is our own.

### M2. Confirming Gemma 3 4B as the right choice

- **Prompt:** "I think Gemma 3 4B is the most suitable choice for my
  project to answer questions from Shakespeare literature retrieval.
  What are your opinions on that?"
- **Useful output:** Argument that a hosted model with strong prior
  Shakespeare knowledge would mask retrieval contribution, making
  baseline-versus-RAG evaluation inconclusive. Gemma 3 4B has
  limited prior knowledge --- so retrieval makes a visible
  difference.
- **Verification:** Ran "Why does Macbeth kill Duncan?" locally
  without retrieval; confirmed the answer was vague and missed key
  details.
- **What we did:** Used this reasoning in the model justification
  section of the report.

### M3. Assessing model size from parameter count

- **Prompt:** "How to assess model size based on the number of
  parameters?"
- **Useful output:** GB ≈ parameters × bits-per-parameter / 8e9.
  Shortcut: parameters (in B) × 0.5 ≈ GB at 4-bit. Distinction
  between RAM and VRAM.
- **Verification:** Cross-checked against the Ollama model library
  page and Google's QAT release notes for Gemma 3.
- **What we did:** Used to eliminate models above the 8 GB ceiling.

### M4. Cross-encoder vs bi-encoder

- **Prompt:** "What does a cross encoder do differently than
  MiniLM-L6-v2? They are both doing semantic search. What will it
  actually improve?"
- **Useful output:** Bi-encoders encode query and document
  separately; cross-encoders concatenate both and attend across
  them. Illustrated with the "Stars, hide your fires" example.
- **Verification:** Ran both systems on the same query and observed
  different passages.
- **What we did:** Adopted the two-stage retrieve-then-rerank
  pipeline; reasoning appears in the methodology section.

### M5. Choosing the retrieval depth k

- **Prompt:** "How do I decide how many top-k should I use for
  retrieval? I know that the more k I choose, the slower output I
  will get. But what is the sweet spot and how much is worth
  sacrificing?"
- **Useful output:** Three-sided trade-off (speed, relevance, noise);
  recommended k=5 for standard pipeline, 15-to-5 for reranked.
- **Verification:** Ran the same query at k=3, 5, 8 and observed
  that at k=8 retrieval included unrelated lines.
- **What we did:** Kept k=5 as the default, 15-to-5 for the reranked
  pipeline.

### M6. Embedding models on archaic text

- **Prompt:** "How will the embedding model handle archaic text?"
- **Useful output:** Pretrained embeddings cluster "kill / murder /
  slay / do the deed" together. Short archaic utterances embed in
  isolated regions without enrichment; scene summaries bridge the
  gap.
- **Verification:** Confirmed retrieval surfaced utterances containing
  "deed", "dark desires", "ambition" for "Why does Macbeth kill
  Duncan?".
- **What we did:** Used in the methodology paragraph on enrichment.

### M7. What fine-tuning would change

- **Prompt:** "If I fine-tune the SLM, how will it improve the
  performance? How will the parameters change and what will the
  extra layers represent? Will it have more knowledge about
  Shakespeare or will it only predict more Shakespearean tone as
  the next word?"
- **Useful output:** LoRA injects small adapter matrices into
  existing attention layers, not new top-level layers. Adapters
  shift the next-token distribution toward training-data patterns,
  not new factual knowledge. Fine-tuning improves stylised output
  but not plot-fact reliability.
- **What we did:** Used to justify the decision to skip fine-tuning.

### M8. Why 384 dimensions

- **Prompt:** "Why does our vector space have 384 dimensions? What
  will happen if we add more dimensions or take away some from
  this?"
- **Useful output:** Architectural choice in
  `all-MiniLM-L6-v2`. More dimensions allow finer distinctions;
  fewer cause cluster overlap.
- **Verification:** `model.encode("test").shape` returns `(384,)`.
- **What we did:** Used in the methodology section to justify
  MiniLM-L6-v2 over larger embedding models.

### M9. Replacing "surface form"

- **Prompt:** "What does it mean with surface form and why it is
  relevant?"
- **Useful output:** Linguistics term for the written characters of
  a word; suggested replacing with plainer language.
- **What we did:** Replaced "surface form" with "regardless of how
  they are written or phrased" in the methodology section.

### M10. User-configurable retrieval depth

- **Prompt:** "Should I let the user choose the top k?"
- **Useful output:** Useful for the demo and evaluation; less useful
  for an end user with no Shakespeare background. Keep k=5 default
  and document the choice.
- **What we did:** Kept the configurable input in `main.py` and
  documented the choice in the implementation details section.

---

## Data Engineering Lead Entries

### B1. Scene-level vs utterance-level chunking

- **Task:** Explored the trade-offs between scene-level and
  utterance-level chunking strategies for RAG retrieval quality.
- **Useful output:** Structured comparison of both approaches
  including context preservation, retrieval noise, and corpus size
  considerations.
- **Verification / what we did:** Evaluated both options against the
  assignment requirements and selected scene-level chunking as the
  primary strategy based on our own assessment of dataset size and
  query types.

### B2. Text enrichment with metadata

- **Task:** Investigated how text enrichment using metadata fields
  affects embedding quality and retrieval accuracy.
- **Useful output:** Explanation of how prepending scene summaries
  and keywords to chunk text improves semantic alignment between
  queries and retrieved passages.
- **Verification / what we did:** Verified empirically by comparing
  retrieval results with and without enrichment on the
  instructor-provided questions; confirmed improved relevance.

### B3. Vector index choice

- **Task:** Explored options for building a persistent vector index
  suitable for offline deployment on a standard laptop.
- **Useful output:** Overview of FAISS, ChromaDB, and numpy cosine
  similarity with trade-offs for each at different corpus scales.
- **Verification / what we did:** Independently decided to use cosine
  similarity for the scene-level index (229 chunks) and ChromaDB for
  the utterance-level index (2,622 chunks) based on scale and
  interface requirements.

### B4. Handling the 512-token limit

- **Task:** Consulted on best practices for handling the 512-token
  limit in transformer-based embedding models during chunking.
- **Useful output:** Recommendation to use word-level overlap
  splitting to prevent silent truncation at chunk boundaries.
- **Verification / what we did:** Implemented a 400-word maximum
  chunk size with 50-word overlap and verified no scene content was
  being truncated by inspecting chunk lengths across all three
  plays.

---

## Evaluation Lead Entries

### V1. Evaluation design

- **Task:** Explored evaluation design for comparing three RAG
  systems on a shared question set.
- **Useful output:** Suggested a 1--5 rubric covering correctness,
  grounding, retrieval relevance, usefulness, and style.
- **Verification / what we did:** Refined the criteria to match the
  assignment specification and added our own questions targeting
  expected failure modes.

### V2. Group-designed evaluation questions

- **Task:** Consulted on how to design group-designed evaluation
  questions that cover a meaningful range of question types.
- **Useful output:** Suggested question types including concept
  explanation, multi-scene, minor characters, beginner explanation,
  and stylised generation.
- **Verification / what we did:** Selected and wrote the final 10
  questions ourselves based on our own understanding of the system
  and the plays.

### V3. LLM-as-judge

- **Task:** Explored the LLM-as-judge approach for automated
  first-pass scoring.
- **Useful output:** Explanation of how automated scoring works and
  where it is unreliable for small generation models.
- **Verification / what we did:** Used only as a starting point; all
  scores were manually reviewed against the actual answers before
  being included in the report.

### V4. Interpreting the retrieval-generation gap

- **Task:** Consulted on how to interpret and present evaluation
  results that show retrieval working but generation failing.
- **Useful output:** Explanation of the retrieval-generation gap and
  how to frame it clearly in an analysis section.
- **Verification / what we did:** Identified this pattern
  independently from the CSV results first; Claude helped articulate
  it clearly in the report.

### V5. Structuring the evaluation sections

- **Task:** Helped structure the evaluation and results sections
  based on the actual evaluation output.
- **Useful output:** Draft structure covering evaluation setup,
  results table, qualitative examples, and failure analysis.
- **Verification / what we did:** All analysis, numbers, and examples
  were taken directly from the evaluation CSV and written by us;
  Claude provided the section structure only.

---

## Integration and Reporting Lead Entries

### I1. Merging three per-lead .tex files

- **Task:** Consulted on how to merge three separate per-lead `.tex`
  files (Bobby's dataset section, Moinul's methodology and system
  sections, Varshini's evaluation and results sections) into a
  single IEEE-format report without losing any author's voice.
- **Useful output:** Suggested keeping each lead's prose verbatim
  and writing only the orchestration layer (abstract, introduction,
  conclusion, section bridges) at the integration stage; harmonising
  `lstlisting` vs `minted` and unifying the bibliography.
- **Verification / what we did:** Followed this approach. Each
  lead's prose appears in `report/final/report.tex` with only minor
  formatting harmonisation (all code blocks use `lstlisting`); the
  abstract, introduction, and conclusion were written at the
  integration stage; the per-lead drafts are preserved under
  `report/<name>/` so any author can confirm their contribution is
  intact.

### I2. Submission checklist

- **Task:** Asked for a checklist of what an integrated submission
  should contain beyond the report itself.
- **Useful output:** Reminders to include a single top-level README,
  consolidated requirements, demo questions for the live
  demonstration, a clear separation between production-system and
  evaluation-harness model choices, and a combined GenAI usage log.
- **Verification / what we did:** Implemented all items: top-level
  `README.md`, `demo/demo_questions.md`, `demo/smoke_test.sh`, this
  consolidated `genai_usage_log.md`, and a polished `src/main.py`
  with both interactive and non-interactive modes.

### I3. Flagging the distilgpt2 vs gemma3:4b inconsistency

- **Task:** Asked how to flag in the report the fact that
  `evaluate.py` uses distilgpt2 for the LLM-as-judge run while the
  production system delivered to the user uses Gemma 3 4B, without
  rewriting the evaluation methodology written by another lead.
- **Useful output:** Suggested treating it as an honestly-reported
  limitation in the failure analysis and conclusion sections, with
  the production model justified separately in the methodology
  section.
- **Verification / what we did:** Added explicit sentences in the
  results, failure analysis, and conclusion sections making the
  distinction visible. Did not modify the evaluation harness itself,
  which was outside the integration role.

---

*This log will continue to be updated if any further AI consultation
occurs before submission.*
