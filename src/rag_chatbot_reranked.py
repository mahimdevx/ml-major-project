# src/rag_chatbot_reranked.py
# Enhanced RAG pipeline with cross-encoder re-ranking

import ollama
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from real_retriever import retrieve, collection
from sentence_transformers import CrossEncoder

# --- Load re-ranker ---
print("Loading re-ranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Collection size cap ---
MAX_COLLECTION = collection.count()

# --- LLM ---
def generate(prompt):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# --- Re-ranking retriever ---
def retrieve_and_rerank(query, n_retrieve=15, n_return=5):
    # Cap n_return and n_retrieve to collection size
    n_return   = min(n_return,   MAX_COLLECTION)
    n_retrieve = min(n_retrieve, MAX_COLLECTION)

    # Step 1 — broad retrieval from ChromaDB
    docs, metas = retrieve(query, n=n_retrieve)

    # Step 2 — re-score with cross-encoder
    pairs = [[query, doc] for doc in docs]
    raw_scores = reranker.predict(pairs)

    # Handle both float and dict output formats across sentence-transformers versions
    scores = [
        float(s) if not isinstance(s, dict) else float(s["score"])
        for s in raw_scores
    ]

    # Step 3 — sort by re-rank score
    ranked = sorted(zip(scores, docs, metas), reverse=True)

    # Step 4 — diversify: max 2 results per scene
    top_docs   = []
    top_metas  = []
    scene_counts = {}

    for score, doc, meta in ranked:
        scene_key = f"{meta['play']}_{meta['act']}_{meta['scene']}"
        if scene_counts.get(scene_key, 0) < 2:
            top_docs.append(doc)
            top_metas.append(meta)
            scene_counts[scene_key] = scene_counts.get(scene_key, 0) + 1
        if len(top_docs) == n_return:
            break

    # Show re-ranking scores
    print(f"\n[Re-ranker] Scored {n_retrieve} candidates → returning top {n_return} (max 2 per scene)")
    for i, (doc, meta) in enumerate(zip(top_docs, top_metas)):
        print(f"  #{i+1} | {meta.get('speaker', 'Unknown')} | Act {meta['act']} Scene {meta['scene']}")

    return top_docs, top_metas

# --- Display retrieved evidence ---
def display_evidence(docs, metadatas):
    print("\n=== Retrieved Evidence ===")
    for doc, m in zip(docs, metadatas):
        location = m.get('scene_summary', '')[:60]
        speaker  = m.get('speaker') or 'Unknown'
        print(f"\n[{m['play']} | Act {m['act']}, Scene {m['scene']} | {location}]")
        print(f"Speaker: {speaker}")
        print(f'  "{doc[:150]}"')

# --- Build context block ---
def build_context(docs, metas):
    parts = []
    for doc, meta in zip(docs, metas):
        parts.append(
            f"[{meta['play']} Act {meta['act']} Scene {meta['scene']}]\n"
            f"Speaker: {meta.get('speaker') or 'Unknown'}\n"
            f"Text: {doc}\n"
            f"Scene context: {meta['scene_summary']}"
        )
    return "\n\n".join(parts)

# --- Load prompt template ---
def load_prompt(mode):
    paths = {
        "qa":       "prompts/system_prompt.txt",
        "concept":  "prompts/system_prompt.txt",
        "stylised": "prompts/stylised_prompt.txt"
    }
    with open(paths.get(mode, "prompts/system_prompt.txt")) as f:
        return f.read()

# --- Main chat function ---
def chat(query, mode="qa", n_results=5):
    # Step 1 — retrieve and re-rank
    docs, metas = retrieve_and_rerank(query, n_retrieve=n_results * 3, n_return=n_results)

    # Step 2 — always display evidence
    display_evidence(docs, metas)

    # Step 3 — evidence mode stops here
    if mode == "evidence":
        return

    # Step 4 — build prompt
    context = build_context(docs, metas)
    template = load_prompt(mode)
    prompt = template.format(context=context, question=query)

    # Step 5 — generate
    response = generate(prompt)

    # Step 6 — print answer
    if mode == "stylised":
        print("\n[Stylised response — creative, not factual]")
    else:
        print("\n=== Answer ===")
    print(response)

# --- Entry point ---
if __name__ == "__main__":
    print("\n=== RAG Chatbot with Re-ranking ===")
    print("Modes:")
    print("  qa       — question answering")
    print("  concept  — character or concept explanation")
    print("  evidence — show source passages only")
    print("  stylised — Shakespearean style response\n")

    valid_modes = ["qa", "concept", "evidence", "stylised"]

    mode = input("Mode (default: qa): ").strip() or "qa"
    if mode not in valid_modes:
        print(f"Invalid mode '{mode}'. Defaulting to qa.")
        mode = "qa"

    n_results = input("Number of passages to retrieve (default: 5): ").strip()
    n_results = int(n_results) if n_results.isdigit() else 5
    n_results = min(n_results, MAX_COLLECTION)

    query = input("Question: ").strip()
    chat(query, mode, n_results)