import requests
from .vector_store import VectorStore

class RAGPipeline:
    def __init__(self, index_dir, top_k=4, use_llm=False):
        self.store = VectorStore(index_dir)
        self.top_k = top_k
        self.use_llm = use_llm

    def answer(self, question):
        results = self.store.search(question, self.top_k)
        context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in results)

        if self.use_llm:
            prompt = (
                "Answer using only the context below. If the answer is not present, "
                "say so. Do not invent facts.\n\nQuestion: " + question +
                "\n\nContext:\n" + context
            )
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model":"llama3.2:3b","prompt":prompt,"stream":False},
                    timeout=120
                )
                response.raise_for_status()
                answer = response.json()["response"].strip()
            except Exception as exc:
                answer = "Ollama unavailable; retrieved context:\n\n" + context
                answer += f"\n\n[LLM unavailable: {exc}]"
        else:
            answer = "Relevant retrieved context:\n\n" + context

        sources = list(dict.fromkeys(r["source"] for r in results))
        return answer, sources
