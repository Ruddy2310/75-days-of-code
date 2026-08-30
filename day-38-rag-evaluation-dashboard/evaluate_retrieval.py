import json
from pathlib import Path
from rag_engine import RAG

def evaluate():
    cases=json.loads(Path("data/evaluation_questions.json").read_text(encoding="utf-8"))
    engine=RAG("index"); hit1=hit3=0; rr=[]; details=[]
    for case in cases:
        results=engine.retrieve(case["question"],3)
        names=[Path(x["source"]).name for x in results]
        rank=next((i for i,s in enumerate(names,1) if s==case["expected_source"]),None)
        hit1 += rank==1
        hit3 += rank is not None and rank<=3
        rr.append(1/rank if rank else 0)
        details.append({"question":case["question"],"expected":case["expected_source"],
                        "retrieved":names,"rank":rank})
    n=len(cases)
    return {"total_questions":n,"hit_at_1":hit1/n if n else 0,
            "hit_at_3":hit3/n if n else 0,"mrr":sum(rr)/n if n else 0,"details":details}

if __name__=="__main__":
    r=evaluate()
    print(f"Hit@1: {r['hit_at_1']:.2%}")
    print(f"Hit@3: {r['hit_at_3']:.2%}")
    print(f"MRR: {r['mrr']:.3f}")
    for x in r["details"]: print(x)
