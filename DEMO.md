# Demonstration Guide
### Shakespeare-Aware RAG System — CSCI433/933 Assignment 2

A 5–10 minute walkthrough that shows every required capability:
system overview, example queries, retrieved evidence, generated answers,
the baseline-vs-RAG contrast, and clearly-labelled stylised generation.

**Before you start**
- Run from the project root.
- Confirm the model is available: `ollama list` should show `gemma3:4b`.
- Confirm the index loads (see README §2).
- Have this script open; the questions below are chosen to make the
  system's strengths and one honest limitation visible.

---

## 0. One-minute overview (say this)
> "Our system adapts a small local model, Gemma 3 4B, to three Shakespeare
> plays using retrieval-augmented generation. We never retrain the model —
> instead, every answer is grounded in passages we retrieve from the text
> and show to the user. We built three systems: a no-retrieval baseline, a
> standard RAG pipeline, and a reranked pipeline. I'll demonstrate each."

Optionally run the scripted version first for a clean, reproducible pass:
```bash
python src/main.py --demo
```

---

## 1. Baseline vs RAG — the core contrast (≈2 min)
Show *why retrieval matters* on the same question.

```bash
python src/main.py
# choose 1 (Baseline)
Question: Why does Macbeth kill Duncan?
```
> Point out: the baseline answers from the model's limited prior knowledge
> only — no evidence, prone to vagueness or hallucination.

```bash
python src/main.py
# choose 2 (RAG), mode 1 (qa), k = 8
Question: Why does Macbeth kill Duncan?
```
> Point out: **retrieved evidence is shown first** — with play, act, scene,
> speaker — and the answer is built from it. This is the grounding the
> baseline lacks.

---

## 2. Concept explanation for a beginner (≈1 min)
```bash
# RAG (2), mode 2 (concept), k = 5
Question: Who is Lady Macbeth?
```
> Point out: beginner-friendly explanation, no assumed literary background.

---

## 3. Evidence-only mode (≈1 min)
```bash
# RAG (2), mode 3 (evidence), k = 5
Question: What does Hamlet say to his father's ghost?
```
> Point out: returns source passages with full metadata and **no
> generation** — useful for verifying what the model is actually given.

---

## 4. Reranking on a multi-scene question (≈1.5 min)
```bash
# RAG + Reranking (3), mode 1 (qa), k = 5
Question: Why does Hamlet delay taking revenge?
```
> Point out: the cross-encoder rescores candidates and the max-2-per-scene
> diversity filter pulls evidence from across Acts 1, 3, 4, and 5 rather
> than one scene — important for a question whose answer spans the play.

---

## 5. Stylised generation — clearly labelled (≈1 min)
```bash
# RAG (2), mode 4 (stylised), k = 5
Question: Describe Juliet's conflict after meeting Romeo
```
> Point out: output carries `[Stylised response — creative, not factual]`,
> stays under 150 words, and is followed by a plain-English gloss. We never
> present stylised text as evidence.

---

## 6. If asked about design choices (be ready to answer)
- **Why Gemma 3 4B / local?** Fits 8 GB RAM at ~2.5 GB; its limited
  Shakespeare knowledge makes the RAG-vs-baseline comparison genuine.
- **Why utterance-level + enrichment?** Precise, speaker-attributed
  evidence; modern-English summaries/keywords bridge the archaic-vocabulary
  gap so retrieval works on meaning, not word overlap.
- **Why reranking + diversity?** Bi-encoder retrieval is fast but coarse;
  the cross-encoder scores query–passage pairs jointly; diversity prevents
  one scene from dominating multi-act answers.
- **Honest limitation:** the batch evaluation harness used a stand-in
  generator (distilgpt2), so its generation scores under-represent the
  deployed Gemma system; retrieval scored 5/5 and transfers directly.
  Aligning the harness to Gemma is our top next step (see
  `INTEGRATION_NOTES.md`).

---

## 7. Fallback if Ollama is unavailable on the demo machine
- Use **evidence mode** (mode 3) — it needs only the index, no model, and
  still demonstrates retrieval quality and metadata.
- Show `results/evaluation_results.csv` to discuss measured behaviour.
