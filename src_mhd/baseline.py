# src/baseline.py
# Baseline system — no retrieval, raw LLM only.
# The LLM answers purely from its own knowledge, no retrieved context.
# This is the comparison point for the improved RAG system.

import ollama

# --- LLM ---
def generate(prompt):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# --- Baseline answer — no retrieval ---
def answer(query):
    with open("prompts/baseline_prompt.txt") as f:
        template = f.read()
    
    prompt = template.format(question=query)
    response = generate(prompt)

    print("\n=== Answer (no retrieval) ===")
    print(response)

# --- Entry point ---
if __name__ == "__main__":
    query = input("Question: ").strip()
    answer(query)