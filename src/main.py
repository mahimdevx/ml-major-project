# src/main.py
# Main entry point for the Shakespeare-Aware RAG System.
#
# Integration role: this file is the single user-facing entry point.
# It wires together the three systems implemented by the group
# (baseline, RAG, RAG+rerank) and the four interaction modes
# (qa, concept, evidence, stylised) without modifying any of the
# underlying implementations.
#
# Two usage modes are supported:
#   1. Interactive (default):
#        python src/main.py
#   2. Non-interactive (useful for the demo and for repeatable runs):
#        python src/main.py --system rag --mode qa --k 8 --query "Who is Hamlet?"
#
# The script changes the working directory to the project root before
# importing the team's modules, so relative paths like
# "prompts/system_prompt.txt" work regardless of where the user runs
# it from.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------
# Path setup -- run from anywhere, behave as if run from project root
# ---------------------------------------------------------------
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
SYSTEM_LABELS = {
    "1": ("baseline", "Baseline        (no retrieval, raw LLM)"),
    "2": ("rag",      "RAG             (utterance-level retrieval)"),
    "3": ("reranked", "RAG + Reranking (retrieval + cross-encoder reranking)"),
}

MODE_LABELS = {
    "1": ("qa",       "Question answering"),
    "2": ("concept",  "Character or concept explanation"),
    "3": ("evidence", "Show source passages only"),
    "4": ("stylised", "Shakespearean style response"),
}

VALID_SYSTEMS = {"baseline", "rag", "reranked"}
VALID_MODES   = {"qa", "concept", "evidence", "stylised"}

# ---------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------
def prompt_system() -> str:
    print("\nSystems:")
    for key, (_, label) in SYSTEM_LABELS.items():
        print(f"  {key} - {label}")
    print()
    choice = input("Choose system (1/2/3, default: 2): ").strip() or "2"
    if choice not in SYSTEM_LABELS:
        print(f"Invalid choice '{choice}'. Defaulting to RAG.")
        choice = "2"
    return SYSTEM_LABELS[choice][0]


def prompt_mode() -> str:
    print("\nModes:")
    for key, (_, label) in MODE_LABELS.items():
        print(f"  {key} - {label}")
    print()
    choice = input("Select mode (1/2/3/4, default: 1): ").strip() or "1"
    if choice not in MODE_LABELS:
        print(f"Invalid choice '{choice}'. Defaulting to question answering.")
        choice = "1"
    return MODE_LABELS[choice][0]


def prompt_k() -> int:
    raw = input("Number of passages to retrieve (default: 5): ").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 5


def prompt_query() -> str:
    query = input("\nQuestion: ").strip()
    if not query:
        print("Empty question. Exiting.")
        sys.exit(0)
    return query

# ---------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------
def run_baseline(query: str) -> None:
    try:
        from baseline import answer
    except Exception as e:
        _fail_with(
            f"Could not load the baseline module: {e}",
            hint="Check that Ollama is installed and the model is pulled: "
                 "`ollama pull gemma3:4b`.",
        )
    answer(query)


def run_rag(query: str, mode: str, k: int) -> None:
    try:
        from rag_chatbot import chat
    except Exception as e:
        _fail_with(
            f"Could not load the standard RAG module: {e}",
            hint="Check that the ChromaDB index exists at "
                 "`data/chroma_utterances/` and Ollama is running.",
        )
    chat(query, mode, k)


def run_reranked(query: str, mode: str, k: int) -> None:
    try:
        from rag_chatbot_reranked import chat
    except Exception as e:
        _fail_with(
            f"Could not load the reranked RAG module: {e}",
            hint="Check that the ChromaDB index exists at "
                 "`data/chroma_utterances/`, "
                 "the cross-encoder model can be downloaded, "
                 "and Ollama is running.",
        )
    chat(query, mode, k)


def _fail_with(message: str, hint: str = "") -> None:
    sys.stderr.write(f"\n[error] {message}\n")
    if hint:
        sys.stderr.write(f"[hint]  {hint}\n")
    sys.exit(1)

# ---------------------------------------------------------------
# Argument parsing for non-interactive use
# ---------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="shakespeare-rag",
        description=(
            "Shakespeare-Aware RAG System (CSCI433/933 Assignment 2). "
            "Run without arguments for an interactive menu, or pass "
            "--system / --mode / --query for a non-interactive single run."
        ),
    )
    p.add_argument(
        "--system",
        choices=sorted(VALID_SYSTEMS),
        help="Which system to use. Omit for interactive prompt.",
    )
    p.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        help="Interaction mode (ignored for baseline). "
             "Omit for interactive prompt.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of passages to retrieve (RAG systems only, default 5).",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="The question to ask. Omit for interactive prompt.",
    )
    return p.parse_args()

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("\n=== Shakespeare-Aware RAG System ===")
    print("CSCI433/933 Assignment 2\n")

    # Resolve each parameter: prefer CLI argument, fall back to prompt.
    system = args.system or prompt_system()

    if system == "baseline":
        query = args.query or prompt_query()
        run_baseline(query)
        return

    mode  = args.mode  if args.mode  in VALID_MODES else prompt_mode()
    k     = args.k     if args.k     and args.k > 0 else prompt_k()
    query = args.query or prompt_query()

    if system == "reranked":
        run_reranked(query, mode, k)
    else:
        run_rag(query, mode, k)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n[aborted]\n")
        sys.exit(130)
