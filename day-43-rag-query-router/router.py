import re
ROUTES={'comparison':({'compare','comparison','difference','versus','vs','better'},'hybrid','Use semantic + keyword retrieval for both concepts.'),'technical':({'code','python','api','algorithm','implementation','function','programming'},'technical','Prefer precise technical and implementation-focused chunks.'),'factual':({'what','when','where','who','define','definition','meaning'},'semantic','Prefer semantic retrieval for concise factual answers.'),'general':(set(),'semantic','Use the default semantic retrieval strategy.')}
def route_query(query):
    tokens=set(re.findall(r'[a-zA-Z]+',query.lower())); scores={k:len(tokens&v[0]) for k,v in ROUTES.items()}; best=max(scores,key=scores.get)
    if scores[best]==0: best='general'
    conf=min(1.0,.5+scores[best]*.15) if best!='general' else .5; strategy,desc=ROUTES[best][1:]
    return {'query':query,'route':best,'strategy':strategy,'confidence':round(conf,2),'description':desc,'route_scores':scores}
