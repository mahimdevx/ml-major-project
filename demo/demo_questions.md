# Live Demonstration Script

This is the prepared set of questions for the live demo. Each is paired
with the recommended system, mode, and `k`, plus a brief note on what
the demo is intended to show.

The instructor may request any of these (or their own questions)
during the live demonstration.

---

## Setup check (do this BEFORE the demo starts)

```bash
# 1. Make sure Ollama is running
ollama list   # should show gemma3:4b

# 2. Run the smoke test
bash demo/smoke_test.sh
```

If smoke test passes, the system is ready.

---

## Part 1 -- The four required interaction types

### 1.1 Concept explanation

> Who is Hamlet?

```bash
python src/main.py --system rag --mode concept --k 5 \
    --query "Who is Hamlet?"
```

**What this shows:** The system can introduce a character to a beginner
who has never read the play. The retrieved utterances (from Act 1 Scene 2
and Act 3 Scene 1) anchor the explanation in the actual text.

### 1.2 Contextual question answering

> Why does Macbeth kill Duncan?

```bash
python src/main.py --system rag --mode qa --k 8 \
    --query "Why does Macbeth kill Duncan?"
```

**What this shows:** Multi-cause question, retrieval surfaces the
witches' prophecy (Act 1), Lady Macbeth's pressure (Act 1 Scene 5--7),
and the aftermath (Act 2 Scene 2). Use `k=8` so all three threads
appear.

### 1.3 Evidence-only retrieval

> What does Lady Macbeth say after the murder?

```bash
python src/main.py --system rag --mode evidence --k 5 \
    --query "What does Lady Macbeth say after the murder?"
```

**What this shows:** Retrieval works on its own --- no generation, just
the raw utterances with full play/act/scene/speaker metadata. Useful
for audit and grounding.

### 1.4 Stylised generation (labelled creative)

> Describe Juliet's conflict in Early Modern English

```bash
python src/main.py --system rag --mode stylised --k 5 \
    --query "Describe Juliet's conflict after meeting Romeo"
```

**What this shows:** Creative response in Shakespearean register,
always labelled `[Stylised response --- creative, not factual]`,
followed by a plain English summary line.

---

## Part 2 -- Baseline vs RAG vs Reranked comparison

### 2.1 Same question, all three systems

> Why does Hamlet delay taking revenge?

```bash
# System 1: Baseline (no retrieval)
python src/main.py --system baseline \
    --query "Why does Hamlet delay taking revenge?"

# System 2: Standard RAG
python src/main.py --system rag --mode qa --k 5 \
    --query "Why does Hamlet delay taking revenge?"

# System 3: Reranked RAG
python src/main.py --system reranked --mode qa --k 5 \
    --query "Why does Hamlet delay taking revenge?"
```

**What this shows:** Baseline gives a generic answer with no
evidence. RAG anchors the answer in Hamlet's "To be or not to be"
soliloquy (Act 3 Scene 1). Reranked RAG spreads evidence across
multiple scenes due to the diversity constraint, producing a more
complete picture of his hesitation.

### 2.2 Where retrieval is needed (modern phrasing of archaic content)

> What does "wherefore art thou Romeo" actually mean?

```bash
python src/main.py --system reranked --mode qa --k 5 \
    --query "What does wherefore art thou Romeo actually mean?"
```

**What this shows:** A common beginner misreading. The system
retrieves the balcony scene (Romeo & Juliet 2.2) and explains that
"wherefore" means "why", not "where". Demonstrates archaic--modern
vocabulary bridging.

---

## Part 3 -- Known failure cases (be ready to discuss)

These are documented in the report (Section: Failure Analysis). If the
instructor asks about limitations, walk through one of these honestly.

| Q  | System failure | Why it happens |
|----|----------------|----------------|
| Q02 | RAG retrieves the right scene but the answer is repetitive | Generation model failed to incorporate retrieved context |
| Q03 | Scene-level RAG produces empty answer | Generation collapse, not a retrieval issue |
| Q10 | Enhanced RAG gives a nonsensical answer | Despite correct retrieval, generation degrades on short utterances |

Discuss: retrieval is reliable (5.00/5.00) but generation is the
bottleneck.

---

## Part 4 -- Questions to be ready for

The instructor may also ask one of the official questions:

1. Who is Hamlet?
2. What is the role of Lady Macbeth?
3. What is the conflict between the Montagues and the Capulets?
4. Why does Macbeth kill Duncan?
5. Why does Hamlet delay taking revenge?

Each has been evaluated --- see `results/evaluation_results.csv` for
the recorded answers.

---

## Tips for the demonstration

1. **Always show the retrieved evidence first.** The evidence panel
   prints before the generated answer. Let the instructor see what
   the model was given.

2. **Use `--mode evidence` to defend retrieval quality.** If anyone
   questions whether the right passages are being found, switch to
   evidence mode and show them.

3. **Use `k=8` for instructor questions.** This is the manually-tested
   default for complex multi-scene questions.

4. **If Ollama is slow,** open `ollama list` in another terminal to
   confirm the model is loaded. First generation after a long idle
   may take 30+ seconds.

5. **For stylised responses,** always read out the "creative, not
   factual" label. This is what distinguishes our system from one
   that hallucinates Shakespeare.
