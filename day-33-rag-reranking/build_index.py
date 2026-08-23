from pathlib import Path
import json, faiss, joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
DATA=Path('data/documents'); OUT=Path('index')
docs=[]
for p in sorted(DATA.rglob('*')):
    if p.is_file() and p.suffix.lower() in {'.txt','.md'}:
        t=p.read_text(encoding='utf-8',errors='ignore').strip()
        if t: docs.append({'source':str(p).replace('\\','/'),'text':t})
chunks=[]
for d in docs:
    start=cid=0
    while start<len(d['text']):
        end=min(start+700,len(d['text'])); t=d['text'][start:end].strip()
        if t: chunks.append({'source':d['source'],'chunk_id':cid,'text':t})
        if end==len(d['text']): break
        start=max(end-120,start+1); cid+=1
texts=[x['text'] for x in chunks]
model=SentenceTransformer('all-MiniLM-L6-v2')
v=model.encode(texts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True).astype('float32')
tfidf=TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2)); matrix=tfidf.fit_transform(texts)
OUT.mkdir(exist_ok=True); idx=faiss.IndexFlatIP(v.shape[1]); idx.add(v)
faiss.write_index(idx,str(OUT/'semantic.faiss')); joblib.dump(tfidf,OUT/'tfidf_vectorizer.joblib'); joblib.dump(matrix,OUT/'tfidf_matrix.joblib')
(OUT/'metadata.json').write_text(json.dumps(chunks,indent=2),encoding='utf-8')
print(f'Indexed {len(chunks)} chunks.')
