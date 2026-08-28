from pathlib import Path
import json, faiss, joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA=Path("data/documents"); OUT=Path("index")

def load_documents():
    docs=[]
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt",".md"}:
            text=p.read_text(encoding="utf-8",errors="ignore").strip()
            if text: docs.append({"source":str(p).replace("\\","/"),"text":text})
    return docs

def chunk_documents(docs,size=700,overlap=120):
    chunks=[]
    for doc in docs:
        start=cid=0
        while start<len(doc["text"]):
            end=min(start+size,len(doc["text"]))
            text=doc["text"][start:end].strip()
            if text: chunks.append({"source":doc["source"],"chunk_id":cid,"text":text})
            if end==len(doc["text"]): break
            start=max(end-overlap,start+1); cid+=1
    return chunks

def build():
    chunks=chunk_documents(load_documents())
    if not chunks: raise SystemExit("No TXT or Markdown documents found.")
    texts=[x["text"] for x in chunks]
    model=SentenceTransformer("all-MiniLM-L6-v2")
    vectors=model.encode(texts,convert_to_numpy=True,normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")
    tfidf=TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2))
    matrix=tfidf.fit_transform(texts)
    OUT.mkdir(exist_ok=True)
    index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    faiss.write_index(index,str(OUT/"semantic.faiss"))
    joblib.dump(tfidf,OUT/"tfidf_vectorizer.joblib")
    joblib.dump(matrix,OUT/"tfidf_matrix.joblib")
    (OUT/"metadata.json").write_text(json.dumps(chunks,indent=2),encoding="utf-8")
    print(f"Indexed {len(chunks)} chunks.")

if __name__=="__main__": build()
