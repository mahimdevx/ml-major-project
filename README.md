# Shakespeare-Aware RAG System

**CSCI433/933 Machine Learning Algorithms and Applications --- Assignment 2**
**Group Project: Domain Adaptation with Small Language Models**

A retrieval-augmented generation (RAG) system that adapts a pretrained
Small Language Model (Gemma 3 4B, served locally via Ollama) to support
beginner-friendly interaction with three Shakespeare plays: *Hamlet*,
*Macbeth*, and *Romeo and Juliet*.

The system implements all four interaction types required by the
specification:
- Concept explanation
- Question answering grounded in the plays
- Evidence-only retrieval (no generation)
- Stylised Shakespearean generation, clearly labelled as creative

A standard, RAG-based, and reranked-RAG pipeline are all provided for
comparison.

---

## Quick Start

Run from the project root.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull the model
#    https://ollama.com
ollama pull gemma3:4b

# 3. Launch the interactive entry point
python src/main.py
```

The first run will pull the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB) and the cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB) from the HuggingFace
hub. After that, the system runs entirely offline.

### Non-interactive use (for the demo and scripted runs)

```bash
# Standard RAG, qa mode, retrieve 5 passages
python src/main.py --system rag --mode qa --k 5 \
    --query "Why does Macbeth kill Duncan?"

# Enhanced RAG with reranking, evidence-only mode
python src/main.py --system reranked --mode evidence --k 8 \
    --query "What is the role of Lady Macbeth?"

# Baseline (no retrieval) for comparison
python src/main.py --system baseline \
    --query "Who is Hamlet?"
```

Run `python src/main.py --help` for the full argument list.

---

## Repository Structure

```
ml-major-project/
├── README.md                  <- this file (top-level entry point)
├── requirements.txt           <- pinned Python dependencies
├── data/
│   ├── raw/                   <- instructor JSONL files (scenes + utterances)
│   ├── processed/             <- cleaned chunks + scene-level embeddings
│   └── chroma_utterances/     <- ChromaDB utterance index (built on first run)
├── src/
│   ├── main.py                <- entry point (interactive + CLI)
│   ├── baseline.py            <- System 1: no retrieval, Gemma 3 4B only
│   ├── rag_chatbot.py         <- System 2: utterance-level RAG
│   ├── rag_chatbot_reranked.py<- System 3: RAG + cross-encoder rerank
│   ├── real_retriever.py      <- ChromaDB adapter
│   ├── evaluate.py            <- batch evaluation harness (15 questions)
│   └── README.md              <- developer-facing system details
├── prompts/
│   ├── system_prompt.txt      <- main RAG prompt (qa, concept modes)
│   ├── baseline_prompt.txt    <- no-retrieval baseline prompt
│   └── stylised_prompt.txt    <- Shakespearean-style generation prompt
├── results/
│   ├── evaluation_results.csv <- 15 questions x 3 systems, scored
│   └── evaluation_results.json<- same data with retrieved chunks
├── report/
│   ├── final/                 <- INTEGRATED REPORT - submit this
│   │   ├── report.tex
│   │   ├── pipeline_diagram.tex
│   │   └── references.bib
│   ├── bobby/                 <- per-lead drafts (Data Engineering)
│   ├── moinul/                <- per-lead drafts (Model & RAG)
│   └── Varshini/              <- per-lead drafts (Evaluation)
├── demo/
│   ├── demo_questions.md      <- prepared questions for the live demo
│   └── smoke_test.sh          <- 30-second sanity check
├── docs/
│   └── architecture.md
└── genai_usage_log.md         <- consolidated GenAI usage log
```

---

## System Architecture

```
User Query
   |
   v
[ Embedding (MiniLM-L6-v2, 384-dim) ]
   |
   v
[ ChromaDB retrieval over 2,622 utterances ]
   |
   v
[ Optional: cross-encoder rerank + scene-level diversity ]
   |
   v
[ Prompt construction with retrieved context ]
   |
   v
[ Gemma 3 4B via Ollama ]
   |
   v
[ Generated answer + retrieved evidence displayed to user ]
```

For full design rationale, see `report/final/report.tex` (the
integrated submission report) or `src/README.md` (developer-facing
notes from the RAG Lead).

---

## Interaction Modes (Systems 2 and 3)

| Mode      | When to use                                | Example question |
|-----------|--------------------------------------------|------------------|
| `qa`      | Contextual questions about events          | "Why does Macbeth kill Duncan?" |
| `concept` | Character or concept explanations          | "Who is Lady Macbeth?" |
| `evidence`| Show source passages only, no generation   | "What does Hamlet say to his father's ghost?" |
| `stylised`| Creative Shakespearean response (<=150 wd) | "Describe Juliet's conflict in Early Modern English" |

Stylised responses are always labelled `[Stylised response --- creative, not factual]`
and are not to be treated as evidence.

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| RAM       | 8 GB minimum (Gemma 3 4B uses ~2.5 GB under Q4_K_M) |
| Disk      | ~3 GB for model + ~400 MB for embeddings/index |
| GPU       | Not required |
| OS        | macOS, Linux, or Windows |

Generation typically takes 15--20 seconds per response on CPU. The
reranker adds noticeable latency in the enhanced pipeline.

---

## Reproducibility

- The ChromaDB utterance index is built once and persisted under
  `data/chroma_utterances/`. It does not need to be rebuilt for the
  demonstration.
- Random seeds are not fixed for the generation step because Gemma 3 4B
  is queried through Ollama, where seed control is per-request.
  Retrieval is deterministic given the same query and index.
- The evaluation results in `results/` are committed; they were
  generated by `python src/evaluate.py` and reviewed manually.

---

## Team Roles

| Role | Responsibility |
|------|----------------|
| Data Engineering Lead | Dataset cleaning, chunking, metadata enrichment, index building |
| Model & RAG Lead      | Model selection, RAG pipeline, reranking, prompt design |
| Evaluation Lead       | Question set, scoring rubric, LLM-as-judge, failure analysis |
| Integration & Reporting Lead | Entry point, repository structure, report integration, demo prep |

Each lead's individual contributions to the report are preserved under
`report/<name>/`; the integrated submission is `report/final/report.tex`.

---

## Documentation Map

- For a **demo / live demonstration**: see `demo/demo_questions.md`.
- For **system internals and design choices**: see `src/README.md`.
- For the **submitted report**: see `report/final/report.tex`.
- For the **GenAI usage log**: see `genai_usage_log.md` or
  Appendix C of the report.

---

## License

Base text obtained from Project Gutenberg. Coursework submission for
CSCI433/933 at the University of Wollongong.
