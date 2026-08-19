import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, directory, model_name="all-MiniLM-L6-v2"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        texts = [x["text"] for x in chunks]
        vectors = self.model.encode(texts, convert_to_numpy=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=True).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks
        faiss.write_index(self.index, str(self.directory/"documents.faiss"))
        (self.directory/"metadata.json").write_text(
            json.dumps(chunks, indent=2), encoding="utf-8"
        )

    def load(self):
        self.index = faiss.read_index(str(self.directory/"documents.faiss"))
        self.chunks = json.loads((self.directory/"metadata.json").read_text(encoding="utf-8"))

    def search(self, query, top_k=4):
        if self.index is None: self.load()
        vector = self.model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self.chunks[idx])
                item["score"] = float(score)
                results.append(item)
        return results
