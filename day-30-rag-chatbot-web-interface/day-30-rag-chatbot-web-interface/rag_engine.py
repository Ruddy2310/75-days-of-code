import json
from pathlib import Path
import faiss
import requests
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def load(self):
        if self.index is None:
            self.index = faiss.read_index(str(self.index_dir/"documents.faiss"))
            self.chunks = json.loads(
                (self.index_dir/"metadata.json").read_text(encoding="utf-8")
            )

    def retrieve(self, question, top_k=4):
        self.load()
        vector = self.model.encode(
            [question], convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(vector, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self.chunks[idx])
                item["score"] = float(score)
                results.append(item)
        return results

    def answer(self, question, use_llm=False):
        results = self.retrieve(question)
        if not results:
            return "No relevant information found.", []

        context = "\n\n".join(
            f"Source: {r['source']}\n{r['text']}" for r in results
        )

        answer = "Relevant information retrieved:\n\n" + context

        if use_llm:
            prompt = (
                "Answer using only the context below. Do not invent facts.\n\n"
                f"Question: {question}\n\nContext:\n{context}"
            )
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model":"llama3.2:3b","prompt":prompt,"stream":False},
                    timeout=120,
                )
                response.raise_for_status()
                answer = response.json()["response"].strip()
            except Exception as exc:
                answer += f"\n\n[Ollama unavailable: {exc}]"

        sources = [
            {"path": r["source"], "score": round(r["score"], 4)}
            for r in results
        ]
        return answer, sources
