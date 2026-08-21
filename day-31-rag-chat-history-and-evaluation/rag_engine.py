import json
from pathlib import Path
import faiss,requests
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self,index_dir):
        self.dir=Path(index_dir); self.model=SentenceTransformer("all-MiniLM-L6-v2")
        self.index=None; self.chunks=[]

    def load(self):
        if self.index is None:
            self.index=faiss.read_index(str(self.dir/"documents.faiss"))
            self.chunks=json.loads((self.dir/"metadata.json").read_text(encoding="utf-8"))

    def retrieve(self,question,top_k=4):
        self.load()
        v=self.model.encode([question],convert_to_numpy=True,
                            normalize_embeddings=True).astype("float32")
        scores,ids=self.index.search(v,top_k); out=[]
        for score,i in zip(scores[0],ids[0]):
            if i>=0:
                x=dict(self.chunks[i]); x["score"]=float(score); out.append(x)
        return out

    def answer(self,question,use_llm=False):
        results=self.retrieve(question)
        if not results: return "No relevant information found.",[]
        context="\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in results)
        answer="Relevant retrieved context:\n\n"+context
        if use_llm:
            prompt=f"Answer only from this context. Do not invent facts.\nQuestion: {question}\n\nContext:\n{context}"
            try:
                r=requests.post("http://localhost:11434/api/generate",
                    json={"model":"llama3.2:3b","prompt":prompt,"stream":False},timeout=120)
                r.raise_for_status(); answer=r.json()["response"].strip()
            except Exception as e: answer+=f"\n\n[Ollama unavailable: {e}]"
        sources=[{"path":x["source"],"score":round(x["score"],4)} for x in results]
        return answer,sources
