# Shakespeare-Aware RAG System
### CSCI433/933 Machine Learning Algorithms and Applications — Assignment 2

A Retrieval-Augmented Generation (RAG) system that lets a user with **no
prior Shakespeare knowledge** ask questions about *Hamlet*, *Macbeth*, and
*Romeo and Juliet* and receive accurate, **evidence-grounded** answers. A
compact local model (Gemma 3 4B via Ollama) is adapted to the domain
through retrieval and prompt design — no pretraining, no full fine-tuning.

---

## 1. Repository structure

```
ml-major-project/
├── README.md                     ← this file (setup + how to run)
├── DEMO.md                       ← 5–10 min demonstration script
├── INTEGRATION_NOTES.md          ← integration decisions & open items
├── requirements.txt              ← pinned dependencies
├── data/
│   ├── raw/                      ← instructor JSONL files (do not modify)
│   ├── processed/                ← cleaned chunks + metadata
│   └── chroma_utterances/        ← pre-built ChromaDB index (2,622 utterances)
├── src/
│   ├── main.py                   ← single entry point (menu + demo mode)
│   ├── baseline.py               ← System 1: no retrieval
│   ├── rag_chatbot.py            ← System 2: standard RAG
│   ├── rag_chatbot_reranked.py   ← System 3: RAG + cross-encoder reranking
│   ├── real_retriever.py         ← ChromaDB retrieval adapter
│   └── evaluate.py               ← batch evaluation over 15 questions
├── prompts/
│   ├── system_prompt.txt         ← RAG prompt (qa / concept)
│   ├── baseline_prompt.txt       ← baseline prompt
│   └── stylised_prompt.txt       ← stylised-mode prompt
├── results/
│   ├── evaluation_results.csv    ← per-question scores
│   └── evaluation_results.json   ← full detail incl. retrieved chunks
├── docs/
│   └── architecture.md           ← component overview
└── report/                       ← LaTeX technical report (compiles to PDF)
    ├── main.tex                  ← assembly file (\input s sections)
    ├── config.tex                ← shared preamble (IEEE-style)
    ├── abstract.tex … conclusion.tex, appendix.tex
    ├── pipeline_diagram.tex
    └── references.bib
```

This layout follows the structure recommended in the assignment
specification (`data/ src/ prompts/ results/ report/`).

---

## 2. Setup

**Requirements:** Python 3.9+, and [Ollama](https://ollama.com) for local
model inference. The system runs on a standard laptop with no GPU
(~2.5 GB RAM for the model).

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama (https://ollama.com), then pull the model
ollama pull gemma3:4b
```

The ChromaDB index lives at `data/chroma_utterances/`. **If it is present**
(pre-built by the data pipeline), no indexing step is needed — verify it
loads:

```bash
python -c "import chromadb; c=chromadb.PersistentClient(path='data/chroma_utterances'); print(c.get_collection('shakespeare_utterances').count(), 'utterances')"
# expected: 2622 utterances
```

If the index is **absent** (it is not part of every snapshot), rebuild it
with the data-engineering index-build script before running the RAG
systems. See `INTEGRATION_NOTES.md` §"Repository completeness".

---

## 3. How to run

Always run **from the project root** so that `data/` and `prompts/` resolve.

```bash
python src/main.py            # interactive menu
python src/main.py --demo     # scripted run of the 5 instructor questions
```

The interface walks you through three choices:

1. **System** — Baseline (1), RAG (2, default), or RAG + Reranking (3).
2. **Mode** (RAG systems only):

   | # | Mode       | Use it for                                   |
   |---|------------|----------------------------------------------|
   | 1 | `qa`       | events and motivations ("Why does …")        |
   | 2 | `concept`  | character/idea explanations ("Who is …")     |
   | 3 | `evidence` | show source passages only, no generation     |
   | 4 | `stylised` | creative Shakespearean reply (≤150 words)     |

3. **Passages (`k`)** — how many to retrieve (default 5).

**Retrieved evidence is always displayed before the answer**, with play,
act, scene, and speaker. **Stylised output is always labelled
`[Stylised response — creative, not factual]`** and must not be treated as
evidence.

Each pipeline script can also be run on its own
(`python src/baseline.py`, `python src/rag_chatbot.py`,
`python src/rag_chatbot_reranked.py`).

---

## 4. System components

| Component   | Where                       | What it does                                            |
|-------------|-----------------------------|---------------------------------------------------------|
| Interface   | `src/main.py`               | Menu, input validation, evidence display, demo mode     |
| Baseline    | `src/baseline.py`           | Prompt-only Gemma 3 4B, no retrieval (comparison point) |
| Standard RAG| `src/rag_chatbot.py`        | MiniLM query embedding → ChromaDB top-*k* → generation  |
| Reranked RAG| `src/rag_chatbot_reranked.py`| + cross-encoder rerank + max-2-per-scene diversity     |
| Retriever   | `src/real_retriever.py`     | ChromaDB adapter returning docs + metadata              |
| Evaluation  | `src/evaluate.py`           | Runs 15 questions through all systems → CSV/JSON        |

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Generator: `gemma3:4b`.

---

## 5. Evaluation

```bash
python src/evaluate.py        # writes results/evaluation_results.{csv,json}
```

> **Note for assessors / team:** `src/evaluate.py` currently uses a
> lightweight stand-in generator (`distilgpt2`) for the batch run, while
> the interactive system in `src/main.py` uses **Gemma 3 4B**. Retrieval
> scores are model-independent and transfer directly; the end-to-end
> generation scores reflect the stand-in. See `INTEGRATION_NOTES.md` for
> the one-line change that aligns the harness with the deployed model.

---

## 6. Report

The technical report lives in `report/` as a modular LaTeX project.

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

It compiles to a 10-page IEEE-style PDF (main body ≤10 pages; references
and appendices excluded from the limit, per the specification).
