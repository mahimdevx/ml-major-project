"""
Retrieval evaluation module.
"""

from retrieval.retriever import ShakespeareRetriever


EVALUATION_QUERIES = [
    "Who is Hamlet?",
    "Why does Macbeth kill Duncan?",
    "Why is Juliet conflicted after meeting Romeo?",
    "What is Lady Macbeth's role?",
    "What causes the feud between Montagues and Capulets?"
]


def evaluate_retrieval(retriever):
    for query in EVALUATION_QUERIES:
        results = retriever.retrieve(query)

        print("=" * 50)
        print(f"Query: {query}")

        for idx, chunk in enumerate(results, 1):
            print(f"\nResult {idx}")
            print(chunk)


if __name__ == "__main__":
    print("Run retrieval evaluation after indexing chunks")
