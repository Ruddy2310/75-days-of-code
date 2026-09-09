import json
from flask import Flask, jsonify, render_template, request
from evaluator import AnswerQualityEvaluator

app = Flask(__name__)
evaluator = AnswerQualityEvaluator()

with open("data/evaluation_dataset.json", "r", encoding="utf-8") as f:
    DATASET = json.load(f)


def evaluate_dataset(dataset):
    results = [
        evaluator.evaluate(
            item["question"],
            item["answer"],
            item["reference_answer"],
        )
        for item in dataset
    ]
    average = (
        sum(item["quality_score"] for item in results) / len(results)
        if results else 0.0
    )
    return results, round(average, 4)


@app.route("/")
def home():
    results, average = evaluate_dataset(DATASET)
    return render_template(
        "index.html",
        results=results,
        average=average,
        count=len(results),
    )


@app.post("/api/evaluate")
def api_evaluate():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")
    answer = payload.get("answer", "")
    reference = payload.get("reference_answer", "")

    if not question or not answer or not reference:
        return jsonify({
            "error": "question, answer and reference_answer are required"
        }), 400

    return jsonify(evaluator.evaluate(question, answer, reference))


@app.get("/api/evaluate-dataset")
def api_dataset():
    results, average = evaluate_dataset(DATASET)
    return jsonify({"average_quality_score": average, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
