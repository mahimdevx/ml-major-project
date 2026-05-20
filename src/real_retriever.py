# src/real_retriever.py
import chromadb
import os

# --- Connect to ChromaDB ---
DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "chroma_utterances"
)

print("Loading ChromaDB utterance index...")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("shakespeare_utterances")
print(f"Loaded {collection.count()} utterances.")

# --- Adapter function ---
def retrieve(query, n=8):
    results = collection.query(
        query_texts=[query],
        n_results=n
    )
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas