import json,re
from pathlib import Path
import faiss,joblib,requests,numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder

class RetrievalCache:
    def __init__(self,max_size=100):
        self.max_size=max_size; self.data={}; self.hits=0; self.misses=0
    def key(self,q): return re.sub(r"\s+"," ",q.strip().lower())
    def get(self,q):
        k=self.key(q)
        if k in self.data: self.hits+=1; return self.data[k]
        self.misses+=1; return None
    def put(self,q,value):
        k=self.key(q)
        if len(self.data)>=self.max_size: del self.data[next(iter(self.data))]
        self.data[k]=value
    def stats(self):
        total=self.hits+self.misses
        return {"entries":len(self.data),"hits":self.hits,"misses":self.misses,
                "hit_rate":round(self.hits/total,4) if total else 0}

class ConversationalRAG:
    def __init__(self,index_dir="index"):
        self.dir=Path(index_dir)
        self.embedder=SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cache=RetrievalCache()
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

    def rewrite_query(self,message,history):
        if not history:return message
        recent=[x.get("content","") for x in history[-4:] if x.get("content")]
        pronouns={"it","this","that","they","them","these","those","its"}
        words=set(message.lower().replace("?","").replace(".","").split())
        if words & pronouns or len(message.split())<6:
            return " ".join(recent)+" Follow-up: "+message
        return message

    def retrieve(self,query,top_k=4):
        cached=self.cache.get(query)
        if cached is not None:return cached
        self.load(); n=len(self.chunks)
        q=self.embedder.encode([query],convert_to_numpy=True,
                               normalize_embeddings=True).astype("float32")
        semantic,_=self.index.search(q,n); semantic=semantic[0]
        keyword=(self.matrix@self.vectorizer.transform([query]).T).toarray().ravel()
        hybrid=.65*self.norm(semantic)+.35*self.norm(keyword)
        ids=np.argsort(-hybrid)[:max(8,top_k*2)]
        candidates=[]
        for i in ids:
            x=dict(self.chunks[int(i)]); x["hybrid_score"]=float(hybrid[i]); candidates.append(x)
        scores=self.reranker.predict([(query,x["text"]) for x in candidates])
        for x,s in zip(candidates,scores): x["rerank_score"]=float(s)
        candidates.sort(key=lambda x:x["rerank_score"],reverse=True)
        result=candidates[:top_k]; self.cache.put(query,result); return result

    def answer(self,message,history=None,use_llm=False):
        history=history or []
        retrieval_query=self.rewrite_query(message,history)
        results=self.retrieve(retrieval_query)
        context="\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in results)
        answer="Retrieved evidence:\n\n"+context
        if use_llm:
            prompt=("You are a grounded RAG assistant. Answer only from the supplied "
                    "evidence. If evidence is insufficient, say so.\n\n"
                    f"Question: {message}\n\nEvidence:\n{context}")
            try:
                r=requests.post("http://localhost:11434/api/generate",
                    json={"model":"llama3.2:3b","prompt":prompt,"stream":False},timeout=120)
                r.raise_for_status(); answer=r.json()["response"].strip()
            except Exception as e: answer+=f"\n\n[Ollama unavailable: {e}]"
        sources=[{"path":x["source"],"hybrid":round(x["hybrid_score"],4),
                  "rerank":round(x["rerank_score"],4)} for x in results]
        return {"answer":answer,"retrieval_query":retrieval_query,
                "sources":sources,"cache":self.cache.stats()}
