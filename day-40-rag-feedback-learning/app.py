from flask import Flask, jsonify, render_template, request
from feedback_store import FeedbackStore

app = Flask(__name__)
store = FeedbackStore()

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/feedback")
def get_feedback():
    return jsonify(store.all())

@app.post("/api/feedback")
def add_feedback():
    data = request.get_json(silent=True) or {}
    if not data.get("query") or not data.get("answer"):
        return jsonify({"error": "query and answer are required"}), 400
    score = data.get("quality_score")
    try:
        if score is not None:
            score = float(score)
            if not 0 <= score <= 1:
                raise ValueError
        record = store.add(data["query"], data["answer"], data.get("rating", ""), score, data.get("sources", []))
    except ValueError:
        return jsonify({"error": "rating must be up/down and score must be 0..1"}), 400
    return jsonify(record), 201

@app.get("/api/stats")
def stats():
    return jsonify(store.stats())

if __name__ == "__main__":
    app.run(debug=True)
