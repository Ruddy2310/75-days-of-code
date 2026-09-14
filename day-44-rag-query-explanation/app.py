from flask import Flask, jsonify, render_template, request
from explainer import explain_query

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/explain")
def api_explain():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    return jsonify(explain_query(query))

@app.get("/api/examples")
def examples():
    return jsonify([
        "What is retrieval augmented generation?",
        "Compare vector search and keyword search",
        "How do I implement FAISS in Python?",
        "Tell me about embeddings"
    ])

if __name__ == "__main__":
    app.run(debug=True)
