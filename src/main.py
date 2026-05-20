# src/main.py
# Main entry point for the Shakespeare RAG system

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

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
    query = input("Question: ").strip()
    answer(query)

elif system == "3":
    from rag_chatbot_reranked import chat

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

else:
    from rag_chatbot import chat

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