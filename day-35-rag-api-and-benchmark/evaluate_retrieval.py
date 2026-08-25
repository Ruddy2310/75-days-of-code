import json
from pathlib import Path
from rag_engine import RAGEngine

cases=json.loads(Path("data/evaluation_questions.json").read_text(encoding="utf-8"))
engine=RAGEngine("index")
hit1=hit3=0
rr=[]

for case in cases:
    results=engine.retrieve(case["question"],top_k=3)
    sources=[Path(x["source"]).name for x in results]
    expected=case["expected_source"]
    rank=next((i for i,s in enumerate(sources,1) if s==expected),None)
    hit1 += rank==1
    hit3 += rank is not None
    rr.append(1/rank if rank else 0)
    print(f"Q: {case['question']}")
    print(f"Expected: {expected}")
    print(f"Retrieved: {sources}")
    print(f"Rank: {rank}\n")

total=len(cases)
print("="*45)
print(f"Hit@1: {hit1/total:.2%}")
print(f"Hit@3: {hit3/total:.2%}")
print(f"MRR:    {sum(rr)/total:.3f}")
