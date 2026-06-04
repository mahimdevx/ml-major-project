import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_DIR / "clean_chunks.jsonl"
OUTPUT_FILE = DATA_DIR / "embeddings.npy"

print("Building scene-level embeddings index...")
print(f"Reading chunks from: {INPUT_FILE}")

texts = []
with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        texts.append(item.get("text", ""))

print(f"Loaded {len(texts)} chunks")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

np.save(OUTPUT_FILE, embeddings)
print(f"Saved embeddings to {OUTPUT_FILE}")
