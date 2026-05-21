# src/main.py
# Main entry point for the Shakespeare RAG system

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# --- Mode selection helper ---
def select_mode():
    print("\nModes:")
    print("  1 — Question answering")
    print("  2 — Character or concept explanation")
    print("  3 — Show source passages only")
    print("  4 — Shakespearean style response\n")

    mode_map = {
        "1": "qa",
        "2": "concept",
        "3": "evidence",
        "4": "stylised"
    }

    choice = input("Select mode (1/2/3/4, default: 1): ").strip() or "1"
    if choice not in mode_map:
        print(f"Invalid choice '{choice}'. Defaulting to question answering.")
        choice = "1"

    return mode_map[choice]

# --- Passage count helper ---
def select_k():
    n = input("Number of passages to retrieve (default: 5): ").strip()
    return int(n) if n.isdigit() and int(n) > 0 else 5

# ---------------------------------------------------------------

print("\n=== Shakespeare-Aware RAG System ===")
print("CSCI433/933 Assignment 2\n")

# --- Choose system ---
print("Systems:")
print("  1 — Baseline        (no retrieval, raw LLM)")
print("  2 — RAG             (utterance-level retrieval)")
print("  3 — RAG + Reranking (retrieval + cross-encoder reranking)\n")

system = input("Choose system (1/2/3, default: 2): ").strip() or "2"

if system == "1":
    from baseline import answer
    query = input("\nQuestion: ").strip()
    answer(query)

elif system == "3":
    from rag_chatbot_reranked import chat
    mode     = select_mode()
    n_results = select_k()
    query    = input("Question: ").strip()
    chat(query, mode, n_results)

else:
    from rag_chatbot import chat
    mode      = select_mode()
    n_results = select_k()
    query     = input("Question: ").strip()
    chat(query, mode, n_results)