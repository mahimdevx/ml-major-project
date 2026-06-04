# src/main.py
# ===============================================================
# Shakespeare-Aware RAG System -- unified command-line interface.
# CSCI433/933 Assignment 2.
#
# This is the single entry point for the whole system. It wraps the
# three pipelines (baseline / standard RAG / reranked RAG) behind one
# menu-driven session, validates all input, always shows retrieved
# evidence before the answer, clearly labels stylised output, and fails
# gracefully when a dependency (Ollama / ChromaDB) is missing.
#
#   Run interactively:   python src/main.py
#   Run the demo script: python src/main.py --demo
#
# Run from the project root so that prompt and data paths resolve.
# ===============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------
SYSTEMS = {
    "1": ("Baseline",        "no retrieval, raw LLM"),
    "2": ("RAG",             "utterance-level retrieval"),
    "3": ("RAG + Reranking", "retrieval + cross-encoder reranking"),
}

MODES = {
    "1": ("qa",       "Question answering"),
    "2": ("concept",  "Character or concept explanation"),
    "3": ("evidence", "Show source passages only (no generation)"),
    "4": ("stylised", "Shakespearean-style response (creative)"),
}

DEFAULT_SYSTEM = "2"
DEFAULT_MODE = "1"
DEFAULT_K = 5

# A small, fixed script used by --demo so a marker can reproduce a run
# without typing. These are the five instructor-provided questions.
DEMO_QUESTIONS = [
    ("qa",      "Who is Hamlet?"),
    ("concept", "What is the role of Lady Macbeth?"),
    ("qa",      "What is the conflict between the Montagues and the Capulets?"),
    ("qa",      "Why does Macbeth kill Duncan?"),
    ("qa",      "Why does Hamlet delay taking revenge?"),
]


# ---------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------
def banner():
    print("\n" + "=" * 60)
    print("  Shakespeare-Aware RAG System")
    print("  CSCI433/933 Machine Learning -- Assignment 2")
    print("=" * 60)
    print("  Ask about Hamlet, Macbeth, and Romeo and Juliet.")
    print("  Retrieved evidence is always shown before each answer.")
    print("  Type 'q' at any question prompt to quit.\n")


def rule(title=""):
    print("\n" + "-" * 60)
    if title:
        print(title)
        print("-" * 60)


def choose(prompt, options, default):
    """Generic validated single-choice selector."""
    print()
    for key, (name, desc) in options.items():
        marker = " (default)" if key == default else ""
        print(f"  {key} - {name:<16}{marker}")
        print(f"      {desc}")
    raw = input(f"\n{prompt} (default: {default}): ").strip()
    if raw == "":
        return default
    if raw not in options:
        print(f"  Invalid choice '{raw}'. Using default ({default}).")
        return default
    return raw


def select_k():
    raw = input(f"Passages to retrieve (default: {DEFAULT_K}): ").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    if raw:
        print(f"  Invalid value '{raw}'. Using default ({DEFAULT_K}).")
    return DEFAULT_K


# ---------------------------------------------------------------
# Lazy loaders -- only import heavy modules once a system is chosen,
# and report the specific missing dependency rather than crashing.
# ---------------------------------------------------------------
def load_baseline():
    try:
        from baseline import answer
        return answer
    except Exception as e:
        print(f"\n[Baseline unavailable] {e}")
        print("  Baseline needs Ollama running with 'gemma3:4b' pulled:")
        print("    ollama pull gemma3:4b")
        return None


def load_rag(reranked=False):
    module = "rag_chatbot_reranked" if reranked else "rag_chatbot"
    try:
        mod = __import__(module)
        return mod.chat
    except Exception as e:
        print(f"\n[{module} unavailable] {e}")
        print("  The RAG systems need:")
        print("    - the ChromaDB index at data/chroma_utterances/")
        print("    - Ollama running with 'gemma3:4b' pulled")
        print("  Run from the project root so data/ resolves correctly.")
        return None


# ---------------------------------------------------------------
# Session runners
# ---------------------------------------------------------------
def run_baseline_session():
    answer = load_baseline()
    if answer is None:
        return
    rule("Baseline -- no retrieval (answers from the model alone)")
    while True:
        query = input("\nQuestion ('q' to quit): ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue
        answer(query)


def run_rag_session(reranked=False):
    chat = load_rag(reranked=reranked)
    if chat is None:
        return
    label = "RAG + Reranking" if reranked else "RAG"
    rule(f"{label} -- evidence shown before every answer")
    while True:
        mode_key = choose("Select mode", MODES, DEFAULT_MODE)
        mode = MODES[mode_key][0]
        k = select_k()
        query = input("Question ('q' to quit): ").strip()
        if query.lower() in ("q", "quit", "exit"):
            break
        if not query:
            continue
        chat(query, mode, k)
        again = input("\nAnother question? (Y/n): ").strip().lower()
        if again in ("n", "no", "q"):
            break


def run_demo():
    """Reproducible scripted run over the five instructor questions
    through the standard RAG system (k = 5)."""
    banner()
    print("DEMO MODE -- standard RAG, k=5, five instructor questions.\n")
    chat = load_rag(reranked=False)
    if chat is None:
        return
    for i, (mode, question) in enumerate(DEMO_QUESTIONS, 1):
        rule(f"[{i}/{len(DEMO_QUESTIONS)}] mode={mode} | {question}")
        chat(question, mode, DEFAULT_K)
    print("\nDemo complete.\n")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def main():
    if "--demo" in sys.argv:
        run_demo()
        return

    banner()
    system_key = choose("Choose system", SYSTEMS, DEFAULT_SYSTEM)

    if system_key == "1":
        run_baseline_session()
    elif system_key == "3":
        run_rag_session(reranked=True)
    else:
        run_rag_session(reranked=False)

    print("\nGoodbye.\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Goodbye.\n")
