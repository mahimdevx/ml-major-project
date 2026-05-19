# src/rag_chatbot.py

import ollama
import os
import sys

# --- LLM ---
def generate(prompt):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# --- Retriever ---
# Import from mock for now — one line change when real index arrives
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from real_retriever import retrieve

# --- Display retrieved evidence ---
def display_evidence(docs, metadatas):
    print("\n=== Retrieved Evidence ===")
    for doc, m in zip(docs, metadatas):
        location = m.get('scene_summary', '')[:60]
        speaker  = m.get('speaker') or 'Unknown'
        print(f"\n[{m['play']} | Act {m['act']}, Scene {m['scene']} | {location}]")
        print(f"Speaker: {speaker}")
        print(f'  "{doc[:150]}"')

# --- Build context block for prompt ---
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

# --- Four interaction modes ---
def chat(query, mode="qa", n_results=5):
    docs, metas = retrieve(query, n=n_results)
    display_evidence(docs, metas)

    if mode == "evidence":
        return

    context = build_context(docs, metas)
    template = load_prompt(mode)
    prompt = template.format(context=context, question=query)
    response = generate(prompt)

    if mode == "stylised":
        print("\n[Stylised response — creative, not factual]")
    else:
        print("\n=== Answer ===")
    print(response)


if __name__ == "__main__":
    print("\nModes:")
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

    query = input("Question: ").strip()
    chat(query, mode, n_results)