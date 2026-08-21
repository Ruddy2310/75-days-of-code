from pathlib import Path
import json,faiss
from sentence_transformers import SentenceTransformer

DATA=Path("data/documents"); OUT=Path("index")

def load():
    docs=[]
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt",".md"}:
            t=p.read_text(encoding="utf-8",errors="ignore").strip()
            if t: docs.append({"source":str(p).replace("\\","/"),"text":t})
    return docs

def chunks(docs,size=700,overlap=120):
    out=[]
    for d in docs:
        start=0; cid=0
        while start<len(d["text"]):
            end=min(start+size,len(d["text"]))
            text=d["text"][start:end].strip()
            if text: out.append({"source":d["source"],"chunk_id":cid,"text":text})
            if end==len(d["text"]): break
            start=max(end-overlap,start+1); cid+=1
    return out

docs=load(); data=chunks(docs)
if not data: raise SystemExit("No documents found.")
model=SentenceTransformer("all-MiniLM-L6-v2")
v=model.encode([x["text"] for x in data],convert_to_numpy=True,
               normalize_embeddings=True,show_progress_bar=True).astype("float32")
OUT.mkdir(exist_ok=True)
idx=faiss.IndexFlatIP(v.shape[1]); idx.add(v)
faiss.write_index(idx,str(OUT/"documents.faiss"))
(OUT/"metadata.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
print(f"Indexed {len(docs)} documents and {len(data)} chunks.")
