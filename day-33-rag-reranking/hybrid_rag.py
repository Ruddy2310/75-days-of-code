import json,joblib,faiss,requests,numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from reranker import Reranker
class RAGEngine:
    def __init__(self,index_dir='index'):
        self.d=Path(index_dir); self.embed=SentenceTransformer('all-MiniLM-L6-v2'); self.reranker=Reranker(); self.index=None
    def load(self):
        if self.index is not None:return
        self.index=faiss.read_index(str(self.d/'semantic.faiss')); self.vec=joblib.load(self.d/'tfidf_vectorizer.joblib'); self.mat=joblib.load(self.d/'tfidf_matrix.joblib'); self.chunks=json.loads((self.d/'metadata.json').read_text())
    def norm(self,x):
        x=np.asarray(x,dtype='float32'); return np.zeros_like(x) if x.max()==x.min() else (x-x.min())/(x.max()-x.min())
    def retrieve(self,q,top_k=4):
        self.load(); n=len(self.chunks); v=self.embed.encode([q],convert_to_numpy=True,normalize_embeddings=True).astype('float32'); sem=self.index.search(v,n)[0][0]; key=(self.mat@self.vec.transform([q]).T).toarray().ravel(); s=self.norm(sem); k=self.norm(key); h=.65*s+.35*k; ids=np.argsort(-h)[:max(8,top_k*2)]; cand=[]
        for i in ids:
            x=dict(self.chunks[int(i)]); x['hybrid_score']=float(h[i]); cand.append(x)
        return self.reranker.rerank(q,cand,top_k)
    def answer(self,q,use_llm=False):
        r=self.retrieve(q); context='\n\n'.join(f"Source: {x['source']}\n{x['text']}" for x in r); answer='Reranked evidence:\n\n'+context
        if use_llm:
            try:
                z=requests.post('http://localhost:11434/api/generate',json={'model':'llama3.2:3b','prompt':f'Answer only from this evidence.\nQuestion: {q}\n\nEvidence:\n{context}','stream':False},timeout=120); z.raise_for_status(); answer=z.json()['response'].strip()
            except Exception as e: answer+=f'\n\n[Ollama unavailable: {e}]'
        return answer,[{'path':x['source'],'hybrid':round(x['hybrid_score'],4),'rerank':round(x['rerank_score'],4)} for x in r]
