"""
Retriever module for Shakespeare RAG system.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class ShakespeareRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks):
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        self.embeddings = self.model.encode(texts)

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])[0]

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]
