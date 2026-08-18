"""Day 28 - CLI RAG chatbot."""

from __future__ import annotations

import argparse
import requests

from rag_engine import RAGEngine


def ask_ollama(
    question: str,
    context: str,
    model: str = "llama3.2:3b",
    host: str = "http://localhost:11434",
) -> str | None:
    """Generate a grounded answer with a local Ollama model."""
    prompt = f"""You are a helpful RAG assistant.
Answer the user's question using ONLY the supplied context.
If the context does not contain the answer, say that the information was
not found in the knowledge base. Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}
"""

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip() or None
    except (requests.RequestException, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG chatbot")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for generation")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print("🔎 Building RAG knowledge base...")
    engine = RAGEngine()
    engine.build_index()

    print(f"✅ Indexed {len(engine.chunks)} document chunks.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break

        if not question:
            continue

        results = engine.retrieve(question, top_k=args.top_k)
        context = "\n\n".join(
            f"[Source: {chunk.source}, chunk {chunk.chunk_id}]\n{chunk.text}"
            for chunk, _ in results
        )

        answer = None
        if args.ollama:
            answer = ask_ollama(question, context, model=args.model)

        if answer is None:
            answer = engine.fallback_answer(question, results)

        print(f"\nAssistant: {answer}\n")
        print("Sources:")
        for i, (chunk, score) in enumerate(results, start=1):
            print(f"  [{i}] {chunk.source} | similarity={score:.3f}")
        print()


if __name__ == "__main__":
    main()
