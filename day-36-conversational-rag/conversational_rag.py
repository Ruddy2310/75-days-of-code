import json
from pathlib import Path
import faiss,joblib,requests,numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
class ConversationalRAG:
 def __init__(self,index_dir="index"):
  self.dir=Path(index_dir);self.embedder=SentenceTransformer("all-MiniLM-L6-v2");self.reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2");self.index=None;self.vectorizer=None;self.matrix=None;self.chunks=[]
 def load(self):
  if self.index is not None:return
  self.index=faiss.read_index(str(self.dir/"semantic.faiss"));self.vectorizer=joblib.load(self.dir/"tfidf_vectorizer.joblib");self.matrix=joblib.load(self.dir/"tfidf_matrix.joblib");self.chunks=json.loads((self.dir/"metadata.json").read_text(encoding="utf-8"))
 @staticmethod
 def norm(x):
  x=np.asarray(x,dtype="float32")
  return np.zeros_like(x) if x.max()==x.min() else (x-x.min())/(x.max()-x.min())
 def rewrite_query(self,message,history):
  if not history:return message
  recent=" ".join(x.get("content","") for x in history[-4:])
  pronouns={"it","this","that","they","them","these","those","he","she","its"}
  words=set(message.lower().replace("?","").split())
  return f"{recent} Follow-up question: {message}" if words&pronouns or len(message.split())<6 else message
 def retrieve(self,query,top_k=4):
  self.load();n=len(self.chunks);q=self.embedder.encode([query],convert_to_numpy=True,normalize_embeddings=True).astype("float32")
  sem,_=self.index.search(q,n);sem=sem[0];key=(self.matrix@self.vectorizer.transform([query]).T).toarray().ravel();hybrid=.65*self.norm(sem)+.35*self.norm(key);ids=np.argsort(-hybrid)[:max(8,top_k*2)];c=[]
  for i in ids:x=dict(self.chunks[int(i)]);x["hybrid_score"]=float(hybrid[i]);c.append(x)
  scores=self.reranker.predict([(query,x["text"]) for x in c])
  for x,s in zip(c,scores):x["rerank_score"]=float(s)
  c.sort(key=lambda x:x["rerank_score"],reverse=True);return c[:top_k]
 def chat(self,message,history=None,use_llm=False):
  history=history or []; standalone=self.rewrite_query(message,history);results=self.retrieve(standalone);context="\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in results);answer="Retrieved evidence:\n\n"+context
  if use_llm:
   prompt=("You are a grounded RAG assistant. Answer only from the supplied evidence. If evidence does not answer the question, say so.\n\n"+f"Conversation: {history[-6:]}\nCurrent question: {message}\nEvidence:\n{context}")
   try:
    r=requests.post("http://localhost:11434/api/generate",json={"model":"llama3.2:3b","prompt":prompt,"stream":False},timeout=120);r.raise_for_status();answer=r.json()["response"].strip()
   except Exception as e:answer+=f"\n\n[Ollama unavailable: {e}]"
  sources=[{"path":x["source"],"hybrid":round(x["hybrid_score"],4),"rerank":round(x["rerank_score"],4)} for x in results]
  return {"answer":answer,"retrieval_query":standalone,"sources":sources}
