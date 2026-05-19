# Shakespeare-Aware RAG System
## CSCI433/933 Assignment 2 — Group Project

A Retrieval-Augmented Generation (RAG) chatbot that helps a user with no prior Shakespeare knowledge ask questions about Hamlet, Macbeth, and Romeo and Juliet, and receive accurate, evidence-grounded answers.

---

## System Overview

We built three systems for comparison:

| System | File | Description |
|--------|------|-------------|
| 1 — Baseline | `src_mhd/baseline.py` | No retrieval — raw Gemma 3 4B only |
| 2 — RAG | `src_mhd/rag_chatbot.py` | Utterance-level ChromaDB retrieval |
| 3 — RAG + Reranking | `src_mhd/rag_chatbot_reranked.py` | RAG + cross-encoder reranking + diversity |

---

## Setup

### Requirements
- Python 3.9+
- Ollama installed with Gemma 3 4B pulled

### Step 1 — Install Ollama
Download from https://ollama.com and install.

Then pull the model:
```bash
ollama pull gemma3:4b
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Verify the index exists
```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma_utterances')
col = client.get_collection('shakespeare_utterances')
print(f'Index loaded: {col.count()} utterances')
"
```

Expected output:
```
Index loaded: 2622 utterances
```

If this fails, contact the Data Engineering Lead — the index needs to be rebuilt.

---

## How to Run

Always run from the **project root** (`assignment2/`), not from inside `src/mhd`.

### Single entry point — recommended
```bash
python src_mhd/main.py
```

This launches an interactive menu:
```
=== Shakespeare-Aware RAG System ===
CSCI433/933 Assignment 2

Systems:
  1 — Baseline        (no retrieval, raw LLM)
  2 — RAG             (utterance-level retrieval)
  3 — RAG + Reranking (retrieval + cross-encoder reranking)

Choose system (1/2/3, default: 2):
```

### Run systems individually
```bash
python src/baseline.py              # System 1
python src/rag_chatbot.py           # System 2
python src/rag_chatbot_reranked.py  # System 3
```

---

## The Four Interaction Modes

Systems 2 and 3 support four modes. Select after choosing a system:

| Mode | When to use | Example question |
|------|-------------|-----------------|
| `qa` | Contextual questions about events and motivations | "Why does Macbeth kill Duncan?" |
| `concept` | Character or concept explanations | "Who is Lady Macbeth?" |
| `evidence` | Show source passages only, no generation | "What does Hamlet say to his father's ghost?" |
| `stylised` | Creative response in Shakespearean style (≤150 words) | "Describe Juliet's conflict in Early Modern English" |

**Important:** Stylised responses are always labelled `[Stylised response — creative, not factual]`. Never treat them as evidence.

---

## Number of Passages

After selecting mode, you will be asked:
```
Number of passages to retrieve (default: 5):
```

- **3–5** — focused retrieval, faster answers
- **8–10** — broader coverage, better for complex questions
- Recommended: **8** for instructor evaluation questions

---

## Dataset

| File | Contents |
|------|---------|
| `data/raw/` | Professor's raw JSONL files — do not modify |
| `data/processed/` | Embeddings and chunk metadata |
| `data/chroma_utterances/` | ChromaDB utterance collection (2622 utterances) |

**Utterance counts:**
```
Total raw:                3613
Stage directions removed: 991
Final count:              2622
```

---

## Project Structure

```
assignment2/
├── data/
│   ├── raw/                          ← professor's JSONL files
│   ├── processed/                    ← embeddings + metadata
│   └── chroma_utterances/            ← ChromaDB index
├── src_mhd/
│   ├── main.py                       ← entry point
│   ├── baseline.py                   ← System 1
│   ├── rag_chatbot.py                ← System 2
│   ├── rag_chatbot_reranked.py       ← System 3
│   ├── real_retriever.py             ← ChromaDB adapter
│   └── mock_retriever.py             ← used during development
├── prompts/
│   ├── system_prompt.txt             ← main RAG prompt
│   ├── baseline_prompt.txt           ← baseline prompt
│   └── stylised_prompt.txt           ← stylised mode prompt
├── requirements.txt
├── README.md
└── genai_usage_log.md
```

---

## Design Decisions

### Why Gemma 3 4B via Ollama?
- Runs locally on any group member's laptop (~2.5GB RAM)
- Limited Shakespeare knowledge forces retrieval to do real work
- Makes baseline vs RAG comparison genuine and evaluatable

### Why utterance-level retrieval?
- Scene chunks are 300–500 words — too broad, adds noise
- Utterances are 5–20 words — precise, speaker-attributed evidence
- Enables exact quote display with play/act/scene/speaker metadata

### Why ChromaDB?
- Specified in assignment tech stack
- Handles embedding and retrieval automatically
- Persistent storage — index built once, queried many times

### Why reranking?
- Vector similarity finds mathematically close embeddings — not always relevant
- Cross-encoder reads query + passage together for true relevance scoring
- Shakespeare's archaic vocabulary creates a gap between modern queries and text

### Why diversity constraint (max 2 per scene)?
- Without it, reranker concentrates on one scene
- Diversity ensures coverage across the play's narrative arc
- Tested: Q3 (Hamlet's delay) retrieved evidence from Act 1, 3, 4, and 5

### Why numpy for scene index?
- Explicitly listed as acceptable in spec appendix
- Data Engineering Lead's implementation is clean and functional
- No practical advantage to rebuilding as ChromaDB for scenes

---

## Known Limitations

1. **Motivation questions retrieve aftermath passages** — "Why does Macbeth kill Duncan?" tends to retrieve Act 2 guilt passages rather than Act 1 motivation passages. Increasing n_results to 8–10 and using the reranked system helps significantly.

2. **Cross-play questions are weak** — questions spanning multiple plays may retrieve irrelevant passages.

3. **Very short utterances** — some utterances are only 5 words. Even with enrichment the embedding may not carry enough semantic signal.

4. **Stylised mode may introduce creative details** — always label stylised output as creative and verify against evidence mode.

---

## For the Evaluation Lead

### What to run for formal evaluation

For each evaluation question, run it through **all three systems** and record results:

```bash
# System 1 — Baseline
python src/baseline.py

# System 2 — RAG
python src/rag_chatbot.py
# recommended: mode=qa, n_results=8

# System 3 — RAG + Reranking
python src/rag_chatbot_reranked.py
# recommended: mode=qa, n_results=8
```

### Recommended settings for evaluation
- Mode: `qa` for instructor questions
- n_results: `8` for all systems
- Use `evidence` mode to verify what was retrieved independently

### Smoke test results (pre-verified)

**Confirmed working:**

| Question | System | Verdict |
|----------|--------|---------|
| Why does Macbeth kill Duncan? | RAG (n=8) | ✓ Cites Act 1 planning + Act 2 aftermath |
| Why does Macbeth kill Duncan? | Reranked (n=8) | ✓ Spans Act 1–3, diverse scenes |
| Why does Hamlet delay revenge? | Reranked (n=5) | ✓ Retrieved Act 1, 3, 4, 5 passages |
| How does family feud shape R&J? | RAG (n=8) | ✓ Prologue to Act 5 reconciliation |

**Known failure cases (document in error analysis):**

| Question | System | Issue |
|----------|--------|-------|
| Why does Macbeth kill Duncan? | RAG (n=5) | Retrieves only Act 2 guilt — misses motivation |
| Why does Macbeth kill Duncan? | Reranked (no diversity) | All results from Act 2 Scene 2 only |

### Scoring reminders
- Baseline scores 1/5 on grounding — it shows no evidence by design
- Stylised mode is scored on style quality only — not correctness
- Evidence mode needs no grounding score — it shows raw passages only

---

## Requirements

```
sentence-transformers
chromadb
ollama
jsonlines
pandas
numpy
```

Install all:
```bash
pip install -r requirements.txt
```

---

*For questions about the system → contact the RAG Lead*
*For questions about the dataset → contact the Data Engineering Lead*
*For questions about evaluation → contact the Evaluation Lead*