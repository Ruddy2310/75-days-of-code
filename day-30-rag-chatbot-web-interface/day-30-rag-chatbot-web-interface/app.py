from flask import Flask, render_template, request
from rag_engine import RAGEngine

app = Flask(__name__)
engine = RAGEngine("index")

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    sources = []
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        use_llm = request.form.get("use_llm") == "on"
        if question:
            answer, sources = engine.answer(question, use_llm)

    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        question=question,
    )

if __name__ == "__main__":
    app.run(debug=True)
