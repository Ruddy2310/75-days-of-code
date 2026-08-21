import json
from rag_engine import RAGEngine

engine=RAGEngine("index")
cases=json.loads(open("data/evaluation_questions.json",encoding="utf-8").read())
hits=0
for c in cases:
    results=engine.retrieve(c["question"],4)
    sources=[x["source"] for x in results]
    ok=any(c["expected_source"] in s for s in sources)
    hits+=ok
    print(("PASS" if ok else "MISS"), c["question"])
    print(" expected:",c["expected_source"])
    print(" retrieved:",sources)
print(f"\nRetrieval source hit rate: {hits/len(cases):.2%}")
