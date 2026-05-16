# System Architecture

## Planned Pipeline

```text
User Query
    ↓
Embedding-based Retrieval
    ↓
Top-k Shakespeare Chunks
    ↓
Prompt Construction
    ↓
Small Language Model
    ↓
Grounded Answer + Evidence
```

## Core Components

### 1. Dataset Layer
- Hamlet
- Macbeth
- Romeo and Juliet
- structured metadata

### 2. Retrieval Layer
- sentence-transformers embeddings
- FAISS retrieval
- chunk ranking

### 3. Generation Layer
- prompt engineering
- grounded generation
- beginner-friendly explanations
- Shakespearean style mode

### 4. Evaluation Layer
- correctness
- grounding
- usefulness
- retrieval relevance
- hallucination analysis

## Planned Experiments

- MiniLM vs BGE embeddings
- Scene vs speaker chunking
- Baseline vs RAG
- Hallucination comparison
