#!/usr/bin/env python3
# src/main.py
# ===============================================================
# Shakespeare-Aware RAG System -- command-line interface.
# CSCI433/933 Assignment 2.
#
# A polished, single entry point for the whole system. It wraps the
# three pipelines (baseline / standard RAG / reranked RAG) behind one
# menu-driven session with:
#   * coloured, boxed panels and formatted evidence cards;
#   * validated input that re-prompts on error (never crashes);
#   * a "thinking" spinner during generation;
#   * graceful handling of every missing dependency or runtime error;
#   * always-on evidence display and explicit stylised-output labelling;
#   * a scripted --demo mode for reproducible marking.
#
# It calls the team's existing pipeline functions unchanged and only
# controls presentation, so system behaviour is identical to the
# individual scripts.
#
#   python src/main.py            interactive session
#   python src/main.py --demo     scripted run of the 5 instructor Qs
#   python src/main.py --no-color plain output (no ANSI)
#
# Run from the project root so prompt/data paths resolve.
# ===============================================================

import os
import sys
import shutil
import threading
import itertools
import time
import textwrap

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------
# Colour + width handling (pure ANSI, degrades gracefully)
# ---------------------------------------------------------------
_USE_COLOR = (
    "--no-color" not in sys.argv
    and sys.stdout.isatty()
    and os.environ.get("TERM") != "dumb"
)
if os.name == "nt":           # enable ANSI on Windows 10+ terminals
    os.system("")


def _code(n):
    return f"\033[{n}m" if _USE_COLOR else ""


class C:
    RESET  = _code(0)
    BOLD   = _code(1)
    DIM    = _code(2)
    HEAD   = _code(96)   # bright cyan
    ACCENT = _code(36)   # cyan
    OK     = _code(92)   # green
    WARN   = _code(93)   # yellow
    ERR    = _code(91)   # red
    EVID   = _code(94)   # blue
    STYLE  = _code(95)   # magenta
    GREY   = _code(90)


def term_width(maxw=88, minw=56):
    w = shutil.get_terminal_size((80, 24)).columns
    return max(minw, min(maxw, w))


# ---------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------
def hr(char="─", color=C.GREY):
    print(f"{color}{char * term_width()}{C.RESET}")


def panel(lines, title="", color=C.ACCENT, pad=1):
    """Draw a rounded box around a list of plain-text lines."""
    w = term_width()
    inner = w - 2 - pad * 2
    wrapped = []
    for ln in lines:
        if ln == "":
            wrapped.append("")
            continue
        for seg in (textwrap.wrap(ln, inner) or [""]):
            wrapped.append(seg)

    if title:
        t = f" {title} "
        dash = max(0, w - 3 - len(t))
        top = (f"{color}╭─{C.RESET}{C.BOLD}{t}{C.RESET}"
               f"{color}{'─' * dash}╮{C.RESET}")
    else:
        top = f"{color}╭{'─' * (w - 2)}╮{C.RESET}"
    print(top)
    sp = " " * pad
    for ln in wrapped:
        fill = " " * (inner - len(ln))
        print(f"{color}│{C.RESET}{sp}{ln}{fill}{sp}{color}│{C.RESET}")
    print(f"{color}╰{'─' * (w - 2)}╯{C.RESET}")


def banner():
    w = term_width()
    print()
    print(f"{C.HEAD}{C.BOLD}{'Shakespeare-Aware RAG System'.center(w)}{C.RESET}")
    print(f"{C.ACCENT}{'CSCI433/933  ·  Assignment 2'.center(w)}{C.RESET}")
    hr("═", C.GREY)
    print(f"{C.DIM}Ask about Hamlet, Macbeth, and Romeo & Juliet. Every answer "
          f"shows its\nsource evidence first. Type {C.RESET}q{C.DIM} at any "
          f"question to quit.{C.RESET}")
    print()


# ---------------------------------------------------------------
# Spinner shown while the model generates
# ---------------------------------------------------------------
class Spinner:
    def __init__(self, text="Thinking"):
        self.text = text
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        if _USE_COLOR:
            self._t = threading.Thread(target=self._spin, daemon=True)
            self._t.start()
        else:
            print(f"{self.text}...")
        return self

    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set():
                break
            print(f"\r{C.ACCENT}{ch}{C.RESET} {C.DIM}{self.text}...{C.RESET} ",
                  end="", flush=True)
            time.sleep(0.08)

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join()
        print("\r" + " " * term_width() + "\r", end="", flush=True)


# ---------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------
SYSTEMS = {
    "1": ("Baseline",        "no retrieval — raw model only"),
    "2": ("RAG",             "utterance-level retrieval"),
    "3": ("RAG + Reranking", "retrieval + cross-encoder reranking"),
}
MODES = {
    "1": ("qa",       "Question answering (events, motivations)"),
    "2": ("concept",  "Character / concept explanation"),
    "3": ("evidence", "Show source passages only (no generation)"),
    "4": ("stylised", "Shakespearean-style reply (creative, <=150 words)"),
}
DEFAULT_SYSTEM, DEFAULT_MODE, DEFAULT_K = "2", "1", 5

DEMO_QUESTIONS = [
    ("qa",      "Who is Hamlet?"),
    ("concept", "What is the role of Lady Macbeth?"),
    ("qa",      "What is the conflict between the Montagues and the Capulets?"),
    ("qa",      "Why does Macbeth kill Duncan?"),
    ("qa",      "Why does Hamlet delay taking revenge?"),
]
QUIT = {"q", "quit", "exit"}


# ---------------------------------------------------------------
# Validated input helpers (re-prompt on error; never crash)
# ---------------------------------------------------------------
def choose(prompt, options, default):
    print(f"\n{C.BOLD}{prompt}{C.RESET}")
    for key, (name, desc) in options.items():
        star = f"{C.OK} (default){C.RESET}" if key == default else ""
        print(f"  {C.ACCENT}{key}{C.RESET}  {C.BOLD}{name}{C.RESET}{star}")
        print(f"      {C.DIM}{desc}{C.RESET}")
    for _ in range(3):
        raw = input(f"{C.ACCENT}>{C.RESET} Choose [{default}]: ").strip()
        if raw == "":
            return default
        if raw.lower() in QUIT:
            return None
        if raw in options:
            return raw
        print(f"  {C.WARN}'{raw}' is not an option — enter "
              f"{', '.join(options)}.{C.RESET}")
    print(f"  {C.DIM}Using default ({default}).{C.RESET}")
    return default


def select_k():
    for _ in range(3):
        raw = input(f"{C.ACCENT}>{C.RESET} Passages to retrieve [{DEFAULT_K}]: "
                    ).strip()
        if raw == "":
            return DEFAULT_K
        if raw.lower() in QUIT:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print(f"  {C.WARN}Enter a positive whole number.{C.RESET}")
    return DEFAULT_K


def ask_query():
    while True:
        q = input(f"\n{C.HEAD}{C.BOLD}Question{C.RESET} "
                  f"{C.DIM}(q to quit){C.RESET}: ").strip()
        if q.lower() in QUIT:
            return None
        if q:
            return q
        print(f"  {C.WARN}Please type a question.{C.RESET}")


# ---------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------
def render_evidence(docs, metas):
    if not docs:
        panel(["No passages were retrieved for this query."],
              title="Retrieved Evidence", color=C.WARN)
        return
    n = len(docs)
    print(f"\n{C.EVID}{C.BOLD}Retrieved Evidence{C.RESET} "
          f"{C.DIM}({n} passage{'s' if n != 1 else ''}, shown before the "
          f"answer){C.RESET}")
    for i, (doc, m) in enumerate(zip(docs, metas), 1):
        play = m.get("play", "?")
        act, scene = m.get("act", "?"), m.get("scene", "?")
        speaker = m.get("speaker") or "Unknown"
        summary = (m.get("scene_summary") or "").strip()
        score = m.get("score")
        head = f"{i}. {play} · Act {act} Scene {scene} · {speaker}"
        if score is not None:
            head += f"   (score {score})"
        lines = [head]
        if summary:
            lines.append(f"   {summary[:90]}")
        quote = " ".join(str(doc).split())
        lines.append(f'   "{quote[:200]}"')
        panel(lines, color=C.EVID, pad=1)


def render_answer(text, mode):
    text = (text or "").strip()
    if mode == "stylised":
        panel([text or "(no response)"],
              title="Stylised response — creative, not factual",
              color=C.STYLE)
    else:
        if not text or len(set(text.split())) <= 2:
            panel([text or "(empty)", "",
                   "Note: the model returned little usable text. Try a higher "
                   "passage count, or the reranked system."],
                  title="Answer", color=C.WARN)
        else:
            panel([text], title="Answer", color=C.OK)


def error_panel(title, detail, fixes):
    lines = [detail, ""] + [f"• {f}" for f in fixes]
    panel(lines, title=title, color=C.ERR)


# ---------------------------------------------------------------
# Pipeline loaders -- import lazily; mirror the team's chat() flow
# exactly (retrieve -> build_context -> load_prompt -> generate).
# ---------------------------------------------------------------
def _prompt_path(name):
    return os.path.join("prompts", name)


def load_baseline():
    from baseline import generate

    def run(query, mode=None, k=None):
        with open(_prompt_path("baseline_prompt.txt")) as f:
            tmpl = f.read()
        with Spinner("Generating (no retrieval)"):
            ans = generate(tmpl.format(question=query))
        return [], [], ans
    return run


def load_rag(reranked=False):
    if reranked:
        from rag_chatbot_reranked import (retrieve_and_rerank, build_context,
                                          load_prompt, generate)

        def run(query, mode, k):
            with Spinner("Retrieving + reranking"):
                docs, metas = retrieve_and_rerank(query, n_retrieve=k * 3,
                                                  n_return=k)
            if mode == "evidence":
                return docs, metas, None
            prompt = load_prompt(mode).format(
                context=build_context(docs, metas), question=query)
            with Spinner("Generating"):
                ans = generate(prompt)
            return docs, metas, ans
        return run

    from real_retriever import retrieve
    from rag_chatbot import build_context, load_prompt, generate

    def run(query, mode, k):
        with Spinner("Retrieving"):
            docs, metas = retrieve(query, n=k)
        if mode == "evidence":
            return docs, metas, None
        prompt = load_prompt(mode).format(
            context=build_context(docs, metas), question=query)
        with Spinner("Generating"):
            ans = generate(prompt)
        return docs, metas, ans
    return run


def get_runner(system_key):
    try:
        if system_key == "1":
            return load_baseline(), False
        return load_rag(reranked=(system_key == "3")), True
    except ImportError as e:
        msg = str(e)
        missing = msg.split("'")[-2] if "'" in msg else msg
        error_panel("Missing dependency",
                    f"Could not import a required module ({missing}).",
                    ["Install requirements:  pip install -r requirements.txt",
                     "RAG systems also need Ollama:  ollama pull gemma3:4b"])
        return None, False
    except Exception as e:
        error_panel("Could not start this system", str(e),
                    ["Run from the project root so data/ and prompts/ resolve.",
                     "Ensure the index exists at data/chroma_utterances/.",
                     "Ensure Ollama is running:  ollama pull gemma3:4b"])
        return None, False


# ---------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------
def run_session(system_key):
    name = SYSTEMS[system_key][0]
    print(f"\n{C.OK}Loading {name}…{C.RESET}")
    runner, needs_mode = get_runner(system_key)
    if runner is None:
        return
    print(f"{C.OK}Ready.{C.RESET}")

    while True:
        mode, k = "qa", DEFAULT_K
        if needs_mode:
            mk = choose("Select an interaction mode", MODES, DEFAULT_MODE)
            if mk is None:
                break
            mode = MODES[mk][0]
            k = select_k()
            if k is None:
                break
        query = ask_query()
        if query is None:
            break

        print(f"\n{C.GREY}{'·' * term_width()}{C.RESET}")
        print(f"{C.BOLD}Q:{C.RESET} {query}")
        try:
            docs, metas, answer = runner(query, mode, k)
        except FileNotFoundError as e:
            error_panel("Prompt file not found", str(e),
                        ["Run from the project root (prompts/ must be visible)."])
            continue
        except Exception as e:
            error_panel("The system could not answer this query", str(e),
                        ["Check that Ollama is running and gemma3:4b is pulled.",
                         "Verify the ChromaDB index loads (see README)."])
            continue

        if needs_mode:
            render_evidence(docs, metas)
        if not (needs_mode and mode == "evidence"):
            render_answer(answer, mode)

        nxt = input(f"\n{C.DIM}Enter for another question, q to quit: "
                    f"{C.RESET}").strip().lower()
        if nxt in QUIT:
            break


# ---------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------
def run_demo():
    banner()
    panel(["Scripted demo — standard RAG, k=5, the five instructor questions.",
           "Each item shows retrieved evidence, then the generated answer."],
          title="Demo Mode", color=C.ACCENT)
    runner, _ = get_runner("2")
    if runner is None:
        return
    for i, (mode, q) in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{C.HEAD}{C.BOLD}[{i}/{len(DEMO_QUESTIONS)}]{C.RESET} "
              f"{C.BOLD}{q}{C.RESET}  {C.DIM}(mode={mode}){C.RESET}")
        try:
            docs, metas, ans = runner(q, mode, DEFAULT_K)
            render_evidence(docs, metas)
            render_answer(ans, mode)
        except Exception as e:
            error_panel("Demo step failed", str(e),
                        ["Ensure Ollama + the ChromaDB index are available."])
    print(f"\n{C.OK}Demo complete.{C.RESET}\n")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python src/main.py [--demo] [--no-color]\n"
              "  --demo      scripted run of the five instructor questions\n"
              "  --no-color  disable ANSI colour")
        return
    if "--demo" in sys.argv:
        run_demo()
        return

    banner()
    while True:
        sk = choose("Choose a system", SYSTEMS, DEFAULT_SYSTEM)
        if sk is None:
            break
        run_session(sk)
        again = input(f"\n{C.DIM}Try another system? (y/N): {C.RESET}"
                      ).strip().lower()
        if again not in ("y", "yes"):
            break
    print(f"\n{C.ACCENT}Goodbye.{C.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{C.DIM}Interrupted. Goodbye.{C.RESET}\n")