from flask import Flask,jsonify,render_template
import json
from pathlib import Path
from insights import build_insights
app=Flask(__name__); DATA=Path("data/feedback.json")
def load():
 try:return json.loads(DATA.read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError):return []
@app.get("/")
def home():return render_template("index.html")
@app.get("/api/feedback")
def feedback():return jsonify(load())
@app.get("/api/insights")
def insights():return jsonify(build_insights(load()))
if __name__=="__main__":app.run(debug=True)
