from sentence_transformers import CrossEncoder
class Reranker:
    def __init__(self): self.model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    def rerank(self,query,candidates,top_k=4):
        scores=self.model.predict([(query,x['text']) for x in candidates])
        out=[]
        for x,s in zip(candidates,scores):
            y=dict(x); y['rerank_score']=float(s); out.append(y)
        return sorted(out,key=lambda x:x['rerank_score'],reverse=True)[:top_k]
