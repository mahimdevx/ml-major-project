import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chroma_utterances"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

utterance_files = [
    RAW_DIR / "hamlet_utterances.jsonl",
    RAW_DIR / "macbeth_utterances.jsonl",
    RAW_DIR / "romeo_and_juliet_utterances.jsonl",
]

print("Building ChromaDB utterance index...")
print(f"Using output directory: {OUTPUT_DIR}")

client = chromadb.PersistentClient(path=str(OUTPUT_DIR))

if "shakespeare_utterances" in [c.name for c in client.list_collections()]:
    print("Removing existing collection shakespeare_utterances")
    client.delete_collection("shakespeare_utterances")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="shakespeare_utterances",
    embedding_function=embedding_fn,
)

ids = []
documents = []
metadatas = []

for file_path in utterance_files:
    print(f"Reading utterances from {file_path.name}")
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ids.append(item.get("utterance_id") or item.get("source_id") or str(len(ids)))
            documents.append(item.get("text", ""))
            metadatas.append({
                "play": item.get("play", ""),
                "act": item.get("act", 0),
                "scene": item.get("scene", 0),
                "speaker": item.get("speaker", ""),
                "scene_summary": item.get("scene_summary", ""),
            })

print(f"Adding {len(ids)} utterances to ChromaDB collection...")
collection.add(ids=ids, documents=documents, metadatas=metadatas)

print("Index build complete.")
print(f"Collection count: {collection.count()}")
