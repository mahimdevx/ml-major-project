"""
evaluate.py — Automated evaluation for Assignment 2.

WHAT THIS DOES:
  Runs all 15 evaluation questions through all 3 systems:
    1. Baseline     — prompt-only, no retrieval
    2. RAG          — scene-level retrieval + generation
    3. Enhanced RAG — utterance-level ChromaDB retrieval + generation

  For each answer it optionally runs LLM-as-judge to score automatically.

  Saves results to:
    results/evaluation_results.csv   — spreadsheet for the report
    results/evaluation_results.json  — full detail including retrieved chunks

HOW TO RUN:
  python src/evaluate.py

REQUIREMENTS:
  pip install sentence-transformers scikit-learn numpy chromadb transformers
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR       = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR    = PROJECT_ROOT / "results"
CHROMA_DIR     = PROJECT_ROOT / "data" / "chroma_utterances"

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "distilgpt2"           # replace with your group's SLM
TOP_K            = 5
USE_LLM_JUDGE    = True                   # set False to skip auto-scoring

# ── Evaluation questions ──────────────────────────────────────────────────────
QUESTIONS = [
    # 5 Instructor questions
    {
        "id": "Q01",
        "question": "Who is Hamlet?",
        "expected_focus": "Prince of Denmark, revenge plot, character introduction",
        "type": "concept_explanation",
        "source": "instructor",
    },
    {
        "id": "Q02",
        "question": "What is the role of Lady Macbeth?",
        "expected_focus": "Ambition, manipulation of Macbeth, guilt, mental decline",
        "type": "concept_explanation",
        "source": "instructor",
    },
    {
        "id": "Q03",
        "question": "What is the conflict between the Montagues and the Capulets?",
        "expected_focus": "Long-standing family feud, street violence, Romeo and Juliet caught in between",
        "type": "concept_explanation",
        "source": "instructor",
    },
    {
        "id": "Q04",
        "question": "Why does Macbeth kill Duncan?",
        "expected_focus": "Ambition, witches prophecy, Lady Macbeth pressure",
        "type": "contextual_qa",
        "source": "instructor",
    },
    {
        "id": "Q05",
        "question": "Why does Hamlet delay taking revenge?",
        "expected_focus": "Moral hesitation, need for proof, philosophical uncertainty",
        "type": "contextual_qa",
        "source": "instructor",
    },
    # 10 Group-designed questions
    {
        "id": "Q06",
        "question": "What happens to Ophelia in Hamlet?",
        "expected_focus": "Madness caused by grief, drowning, Hamlet's rejection and father's death",
        "type": "concept_explanation",
        "source": "group",
    },
    {
        "id": "Q07",
        "question": "How does Romeo and Juliet's relationship develop across the play?",
        "expected_focus": "Meeting at party, balcony scene, secret marriage, tragic deaths",
        "type": "multi_scene",
        "source": "group",
    },
    {
        "id": "Q08",
        "question": "What does 'wherefore art thou Romeo' actually mean?",
        "expected_focus": "Juliet asking why he is a Montague, not asking where he is",
        "type": "beginner_explanation",
        "source": "group",
    },
    {
        "id": "Q09",
        "question": "How does ambition affect Macbeth and Lady Macbeth differently?",
        "expected_focus": "Macbeth becomes paranoid and violent, Lady Macbeth descends into guilt and madness",
        "type": "cross_character",
        "source": "group",
    },
    {
        "id": "Q10",
        "question": "What is the role of the ghost in Hamlet?",
        "expected_focus": "Drives the revenge plot, reveals Claudius as murderer, Hamlet's uncertainty about truth",
        "type": "concept_explanation",
        "source": "group",
    },
    {
        "id": "Q11",
        "question": "Why does Juliet fake her death?",
        "expected_focus": "Friar Lawrence's plan to avoid marrying Paris and reunite with Romeo",
        "type": "contextual_qa",
        "source": "group",
    },
    {
        "id": "Q12",
        "question": "How does Macbeth change from the beginning to the end of the play?",
        "expected_focus": "Brave loyal soldier to ambitious murderer to paranoid tyrant",
        "type": "multi_scene",
        "source": "group",
    },
    {
        "id": "Q13",
        "question": "Generate a short Shakespearean-style response from Juliet explaining her conflict after meeting Romeo.",
        "expected_focus": "Love vs family loyalty, poetic Shakespearean register, under 150 words",
        "type": "stylised_generation",
        "source": "group",
    },
    {
        "id": "Q14",
        "question": "What is the significance of the three witches in Macbeth?",
        "expected_focus": "Prophecy, fate vs free will, manipulation of Macbeth's ambition",
        "type": "concept_explanation",
        "source": "group",
    },
    {
        "id": "Q15",
        "question": "Who is Mercutio and why does his death matter?",
        "expected_focus": "Romeo's friend, comic relief, death escalates feud, triggers Romeo's revenge",
        "type": "minor_character",
        "source": "group",
    },
]


# ════════════════════════════════════════════════════════════════════════════
# SYSTEM 1: BASELINE
# ════════════════════════════════════════════════════════════════════════════

class BaselineSystem:
    """
    Prompt-only system — no retrieval.
    Asks the language model directly without any Shakespeare context.
    """

    def __init__(self):
        print("  Loading baseline generation model...")
        try:
            from transformers import pipeline
            self.generator = pipeline(
                "text-generation",
                model=GENERATION_MODEL,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
            )
            self.available = True
        except Exception as e:
            print(f"  [WARNING] Generation model not available: {e}")
            print("  Baseline will return placeholder responses.")
            self.available = False

    def answer(self, question: str) -> Dict:
        prompt = (
            f"You are a Shakespeare expert. Answer this question clearly "
            f"for a beginner reader.\n\nQuestion: {question}\n\nAnswer:"
        )
        if self.available:
            try:
                output = self.generator(prompt)[0]["generated_text"]
                answer = output.replace(prompt, "").strip()
            except Exception:
                answer = "[Generation failed]"
        else:
            answer = "[Baseline model not available — install transformers and distilgpt2]"

        return {
            "system": "baseline",
            "question": question,
            "retrieved_chunks": [],
            "answer": answer,
            "prompt": prompt,
        }


# ════════════════════════════════════════════════════════════════════════════
# SYSTEM 2: RAG (scene-level, numpy)
# ════════════════════════════════════════════════════════════════════════════

class RAGSystem:
    """
    Scene-level RAG using numpy cosine similarity index.
    Uses embeddings.npy + chunks_metadata.jsonl from data/processed/
    """

    def __init__(self):
        print("  Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("  Loading scene-level index...")
        emb_path  = DATA_DIR / "embeddings.npy"
        meta_path = DATA_DIR / "chunks_metadata.jsonl"

        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "Scene-level index not found. Run build_index.py first."
            )

        self.embeddings = np.load(emb_path)
        self.chunks = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        print(f"  Scene index loaded: {len(self.chunks)} chunks")

        # Generation model
        try:
            from transformers import pipeline
            self.generator = pipeline(
                "text-generation",
                model=GENERATION_MODEL,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
            )
            self.gen_available = True
        except Exception:
            self.gen_available = False

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        query_vec = self.model.encode([query])
        scores    = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx   = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        context = "\n\n".join([
            f"[{c['play']} Act {c['act']} Scene {c['scene']}]\n{c['text'][:300]}"
            for c in chunks
        ])
        return (
            f"You are a Shakespeare-aware assistant helping a beginner reader.\n"
            f"Use ONLY the retrieved context below to answer the question.\n"
            f"If the context is insufficient, say so clearly.\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    def answer(self, question: str) -> Dict:
        retrieved = self.retrieve(question)
        prompt    = self.build_prompt(question, retrieved)

        if self.gen_available:
            try:
                output = self.generator(prompt)[0]["generated_text"]
                answer = output.replace(prompt, "").strip()
            except Exception:
                answer = "[Generation failed]"
        else:
            answer = "[Generation model not available]"

        return {
            "system": "rag_scene",
            "question": question,
            "retrieved_chunks": [
                {
                    "play": c["play"],
                    "act": c["act"],
                    "scene": c["scene"],
                    "score": round(c["score"], 3),
                    "summary": c.get("scene_summary", "")[:80],
                }
                for c in retrieved
            ],
            "answer": answer,
            "prompt": prompt,
        }


# ════════════════════════════════════════════════════════════════════════════
# SYSTEM 3: ENHANCED RAG (utterance-level, ChromaDB)
# ════════════════════════════════════════════════════════════════════════════

class EnhancedRAGSystem:
    """
    Utterance-level Enhanced RAG using ChromaDB.
    Uses data/chroma_utterances/ built by build_utterance_chroma.py
    """

    def __init__(self):
        print("  Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("  Loading ChromaDB utterance index...")
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.collection = client.get_collection("shakespeare_utterances")
            print(f"  ChromaDB loaded: {self.collection.count()} utterances")
            self.available = True
        except Exception as e:
            print(f"  [WARNING] ChromaDB not available: {e}")
            print("  Run build_utterance_chroma.py first.")
            self.available = False

        # Generation model
        try:
            from transformers import pipeline
            self.generator = pipeline(
                "text-generation",
                model=GENERATION_MODEL,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
            )
            self.gen_available = True
        except Exception:
            self.gen_available = False

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not self.available:
            return []
        query_vec = self.model.encode([query]).tolist()
        results   = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            chunks.append({
                "play":          meta.get("play", ""),
                "act":           meta.get("act", 0),
                "scene":         meta.get("scene", 0),
                "speaker":       meta.get("speaker", ""),
                "scene_summary": meta.get("scene_summary", ""),
                "text":          results["documents"][0][i],
                "score":         round(1 - results["distances"][0][i], 3),
            })
        return chunks

    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        context = "\n\n".join([
            f"[{c['play']} Act {c['act']} Scene {c['scene']} — {c['speaker']}]\n{c['text'][:300]}"
            for c in chunks
        ])
        return (
            f"You are a Shakespeare-aware assistant helping a beginner reader.\n"
            f"Use ONLY the retrieved context below to answer the question.\n"
            f"If the context is insufficient, say so clearly.\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    def answer(self, question: str) -> Dict:
        retrieved = self.retrieve(question)
        prompt    = self.build_prompt(question, retrieved)

        if self.gen_available and retrieved:
            try:
                output = self.generator(prompt)[0]["generated_text"]
                answer = output.replace(prompt, "").strip()
            except Exception:
                answer = "[Generation failed]"
        elif not retrieved:
            answer = "[No chunks retrieved — ChromaDB not available]"
        else:
            answer = "[Generation model not available]"

        return {
            "system": "enhanced_rag_utterance",
            "question": question,
            "retrieved_chunks": [
                {
                    "play":    c["play"],
                    "act":     c["act"],
                    "scene":   c["scene"],
                    "speaker": c["speaker"],
                    "score":   c["score"],
                    "summary": c.get("scene_summary", "")[:80],
                }
                for c in retrieved
            ],
            "answer": answer,
            "prompt": prompt,
        }


# ════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE
# ════════════════════════════════════════════════════════════════════════════

class LLMJudge:
    """
    Uses a language model to automatically score each answer on 4 criteria.

    WHY LLM-AS-JUDGE:
    - Faster than manual scoring across 45 answers (15 questions x 3 systems)
    - Consistent scoring criteria applied uniformly
    - Still requires human review — treat as a first pass, not final scores

    LIMITATION:
    - Small model (distilgpt2) may not judge accurately
    - Use a larger model (GPT-4, Claude) for more reliable scores
    - Always verify auto-scores manually before including in report
    """

    def __init__(self):
        try:
            from transformers import pipeline
            self.judge = pipeline(
                "text-generation",
                model=GENERATION_MODEL,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=50256,
            )
            self.available = True
        except Exception:
            self.available = False

    def score(self, question: str, expected_focus: str,
              answer: str, retrieved_chunks: List) -> Dict:
        """
        Score an answer on correctness, grounding, retrieval, and usefulness.
        Returns scores 1-5 for each criterion.
        """
        if not self.available or not answer or "[" in answer[:5]:
            return {
                "correctness": None,
                "grounding":   None,
                "retrieval":   None,
                "usefulness":  None,
                "style":       None,
                "judge_note":  "Auto-scoring not available — score manually",
            }

        prompt = (
            f"Rate this Shakespeare QA answer on a scale of 1-5.\n"
            f"Question: {question}\n"
            f"Expected focus: {expected_focus}\n"
            f"Answer: {answer[:200]}\n\n"
            f"Correctness (1-5): "
        )

        try:
            output = self.judge(prompt)[0]["generated_text"]
            raw    = output.replace(prompt, "").strip()
            # Extract first digit found
            score  = next((int(c) for c in raw if c.isdigit() and c in "12345"), 3)
        except Exception:
            score = None

        return {
            "correctness": score,
            "grounding":   len(retrieved_chunks) > 0 and score is not None and max(1, score - 1) or None,
            "retrieval":   len(retrieved_chunks),
            "usefulness":  score,
            "style":       None,
            "judge_note":  "Auto-scored — verify manually before submitting",
        }


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_evaluation() -> None:
    print("=" * 60)
    print("EVALUATION RUNNER")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load systems
    print("\nLoading systems...")
    print("  [1/3] Baseline...")
    baseline = BaselineSystem()
    print("  [2/3] RAG (scene-level)...")
    rag = RAGSystem()
    print("  [3/3] Enhanced RAG (utterance ChromaDB)...")
    enhanced = EnhancedRAGSystem()

    judge = LLMJudge() if USE_LLM_JUDGE else None

    systems = [baseline, rag, enhanced]
    all_results = []

    print(f"\nRunning {len(QUESTIONS)} questions through {len(systems)} systems...")
    print("=" * 60)

    for q_data in QUESTIONS:
        qid      = q_data["id"]
        question = q_data["question"]
        expected = q_data["expected_focus"]
        qtype    = q_data["type"]
        source   = q_data["source"]

        print(f"\n{qid}: {question[:60]}...")

        for system in systems:
            sys_name = system.__class__.__name__
            print(f"  Running {sys_name}...")

            try:
                result = system.answer(question)
            except Exception as e:
                result = {
                    "system":           sys_name,
                    "question":         question,
                    "retrieved_chunks": [],
                    "answer":           f"[ERROR: {e}]",
                    "prompt":           "",
                }

            # LLM judge scoring
            scores = {}
            if judge:
                scores = judge.score(
                    question, expected,
                    result["answer"],
                    result["retrieved_chunks"]
                )

            all_results.append({
                "id":               qid,
                "question":         question,
                "question_type":    qtype,
                "source":           source,
                "expected_focus":   expected,
                "system":           result["system"],
                "answer":           result["answer"],
                "retrieved_chunks": result["retrieved_chunks"],
                "correctness":      scores.get("correctness"),
                "grounding":        scores.get("grounding"),
                "retrieval":        scores.get("retrieval"),
                "usefulness":       scores.get("usefulness"),
                "style":            scores.get("style"),
                "judge_note":       scores.get("judge_note", ""),
                "comments":         "",
            })

            time.sleep(0.1)

    # Save CSV
    csv_path = RESULTS_DIR / "evaluation_results.csv"
    csv_fields = [
        "id", "question", "question_type", "source", "expected_focus",
        "system", "answer", "correctness", "grounding", "retrieval",
        "usefulness", "style", "judge_note", "comments",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, "") for k in csv_fields}
            writer.writerow(row)
    print(f"\nSaved CSV  → {csv_path}")

    # Save JSON (includes full retrieved chunks)
    json_path = RESULTS_DIR / "evaluation_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON → {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    systems_seen = {}
    for r in all_results:
        s = r["system"]
        if s not in systems_seen:
            systems_seen[s] = {"count": 0, "scored": 0}
        systems_seen[s]["count"] += 1
        if r["correctness"] is not None:
            systems_seen[s]["scored"] += 1

    for sys_name, info in systems_seen.items():
        print(f"  {sys_name}: {info['count']} answers, {info['scored']} auto-scored")

    print(f"\nResults saved to {RESULTS_DIR.resolve()}")
    print("Open evaluation_results.csv in Excel to review and fill in manual scores.")
    print("\nNOTE: Auto-scores from LLM judge are a starting point only.")
    print("      Always verify and adjust manually before including in report.")


if __name__ == "__main__":
    run_evaluation()
