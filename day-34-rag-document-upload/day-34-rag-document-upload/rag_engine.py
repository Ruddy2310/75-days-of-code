import json
from pathlib import Path
import faiss,joblib,requests,numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder

class RAGEngine:
    def __init__(self,index_dir="index"):
        self.dir=Path(index_dir)
        self.embedder=SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.index=None; self.vectorizer=None; self.matrix=None; self.chunks=[]

    def load(self):
        if self.index is not None:return
        self.index=faiss.read_index(str(self.dir/"semantic.faiss"))
        self.vectorizer=joblib.load(self.dir/"tfidf_vectorizer.joblib")
        self.matrix=joblib.load(self.dir/"tfidf_matrix.joblib")
        self.chunks=json.loads((self.dir/"metadata.json").read_text(encoding="utf-8"))

    @staticmethod
    def norm(x):
        x=np.asarray(x,dtype="float32")
        if x.max()==x.min(): return np.zeros_like(x)
        return (x-x.min())/(x.max()-x.min())

    def retrieve(self,query,top_k=4):
        self.load(); n=len(self.chunks)
        q=self.embedder.encode([query],convert_to_numpy=True,
                               normalize_embeddings=True).astype("float32")
        sem,_=self.index.search(q,n); sem=sem[0]
        key=(self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        s=self.norm(sem); k=self.norm(key); hybrid=.65*s+.35*k
        ids=np.argsort(-hybrid)[:max(8,top_k*2)]
        candidates=[]
        for i in ids:
            x=dict(self.chunks[int(i)])
            x["hybrid_score"]=float(hybrid[i]); candidates.append(x)
        scores=self.reranker.predict([(query,x["text"]) for x in candidates])
        for x,score in zip(candidates,scores): x["rerank_score"]=float(score)
        candidates.sort(key=lambda x:x["rerank_score"],reverse=True)
        return candidates[:top_k]

    def answer(self,query,use_llm=False):
        results=self.retrieve(query)
        if not results:return "No relevant information found.",[]
        context="\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in results)
        answer="Retrieved evidence:\n\n"+context
        if use_llm:
            try:
                r=requests.post("http://localhost:11434/api/generate",
                    json={"model":"llama3.2:3b","prompt":
                    "Answer only from the evidence. Do not invent facts.\n"
                    f"Question: {query}\n\nEvidence:\n{context}",
                    "stream":False},timeout=120)
                r.raise_for_status(); answer=r.json()["response"].strip()
            except Exception as e: answer+=f"\n\n[Ollama unavailable: {e}]"
        sources=[{"path":x["source"],"hybrid":round(x["hybrid_score"],4),
                  "rerank":round(x["rerank_score"],4)} for x in results]
        return answer,sources
