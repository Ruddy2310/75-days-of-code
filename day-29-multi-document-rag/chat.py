import argparse
from src.rag_pipeline import RAGPipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", action="store_true")
    p.add_argument("--top-k", type=int, default=4)
    args = p.parse_args()

    rag = RAGPipeline("index", top_k=args.top_k, use_llm=args.llm)
    print("Day 29 — Multi-Document RAG Chatbot")
    print("Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}: break
        if not q: continue
        answer, sources = rag.answer(q)
        print("\nAssistant:", answer)
        print("\nSources:")
        for s in sources: print(" -", s)

if __name__ == "__main__":
    main()
