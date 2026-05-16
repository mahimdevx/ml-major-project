"""
RAG pipeline for Shakespeare assistant.
"""


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([
        chunk['text'] for chunk in retrieved_chunks
    ])

    prompt = f"""
You are a Shakespeare-aware assistant.

Use the retrieved context below to answer the user's question.

Requirements:
- Answer clearly for beginners.
- Use grounded evidence.
- Avoid hallucinations.
- If insufficient evidence exists, say so.

Retrieved Context:
{context}

Question:
{query}

Answer:
"""

    return prompt
