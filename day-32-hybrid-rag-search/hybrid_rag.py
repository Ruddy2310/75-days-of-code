import json
from pathlib import Path
import faiss,joblib,requests,numpy as np
from sentence_transformers import SentenceTransformer

class HybridRAG:
    def __init__(self,index_dir="index",semantic_weight=.65):
        self.dir=Path(index_dir); self.semantic_weight=semantic_weight
        self.keyword_weight=1-semantic_weight
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
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
        q=self.model.encode([query],convert_to_numpy=True,
                            normalize_embeddings=True).astype("float32")
        semantic,_=self.index.search(q,n); semantic=semantic[0]
        keyword=(self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        s=self.norm(semantic); k=self.norm(keyword)
        hybrid=self.semantic_weight*s+self.keyword_weight*k
        ranked=np.argsort(-hybrid)[:top_k]
        out=[]
        for i in ranked:
            x=dict(self.chunks[int(i)])
            x.update(semantic_score=round(float(s[i]),4),
                     keyword_score=round(float(k[i]),4),
                     hybrid_score=round(float(hybrid[i]),4))
            out.append(x)
        return out

    def answer(self,query,use_llm=False):
        results=self.retrieve(query)
        if not results:return "No relevant information found.",[]
        context="\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in results)
        answer="Hybrid retrieval results:\n\n"+context
        if use_llm:
            try:
                r=requests.post("http://localhost:11434/api/generate",
                    json={"model":"llama3.2:3b","prompt":
                    f"Answer only from this context. Do not invent facts.\nQuestion: {query}\n\nContext:\n{context}",
                    "stream":False},timeout=120)
                r.raise_for_status(); answer=r.json()["response"].strip()
            except Exception as e: answer+=f"\n\n[Ollama unavailable: {e}]"
        sources=[{"path":x["source"],"hybrid":x["hybrid_score"],
                  "semantic":x["semantic_score"],"keyword":x["keyword_score"]} for x in results]
        return answer,sources
