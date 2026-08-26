from pathlib import Path
import json, faiss, joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
DATA=Path("data/documents"); OUT=Path("index")
def load():
 d=[]
 for p in sorted(DATA.rglob("*")):
  if p.is_file() and p.suffix.lower() in {".txt",".md"}:
   t=p.read_text(encoding="utf-8",errors="ignore").strip()
   if t:d.append({"source":str(p).replace("\\","/"),"text":t})
 return d
def chunks(docs,size=700,overlap=120):
 out=[]
 for d in docs:
  start=cid=0
  while start<len(d["text"]):
   end=min(start+size,len(d["text"])); text=d["text"][start:end].strip()
   if text:out.append({"source":d["source"],"chunk_id":cid,"text":text})
   if end==len(d["text"]):break
   start=max(end-overlap,start+1);cid+=1
 return out
def build():
 c=chunks(load())
 if not c:raise SystemExit("No documents found.")
 texts=[x["text"] for x in c]; model=SentenceTransformer("all-MiniLM-L6-v2")
 v=model.encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True).astype("float32")
 tf=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2)); m=tf.fit_transform(texts)
 OUT.mkdir(exist_ok=True); idx=faiss.IndexFlatIP(v.shape[1]);idx.add(v)
 faiss.write_index(idx,str(OUT/"semantic.faiss"));joblib.dump(tf,OUT/"tfidf_vectorizer.joblib");joblib.dump(m,OUT/"tfidf_matrix.joblib")
 (OUT/"metadata.json").write_text(json.dumps(c,indent=2),encoding="utf-8");print(f"Indexed {len(c)} chunks.")
if __name__=="__main__":build()
