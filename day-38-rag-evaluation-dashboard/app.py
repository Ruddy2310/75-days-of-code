from flask import Flask,render_template,request,session,redirect,url_for,jsonify
from rag_engine import RAG
from evaluate_retrieval import evaluate

app=Flask(__name__); app.secret_key="day38-development-secret"; engine=RAG("index")

@app.route("/",methods=["GET","POST"])
def home():
    result=None
    if request.method=="POST":
        message=request.form.get("message","").strip()
        if message:
            history=session.get("history",[])
            result=engine.answer(message,history,request.form.get("use_llm")=="on")
            history += [{"role":"user","content":message},{"role":"assistant","content":result["answer"]}]
            session["history"]=history[-12:]
    return render_template("index.html",history=session.get("history",[]),result=result,cache=engine.cache.stats())

@app.post("/clear")
def clear(): session.pop("history",None); return redirect(url_for("home"))

@app.post("/api/chat")
def chat():
    data=request.get_json(silent=True) or {}; message=str(data.get("message","")).strip()
    if not message:return jsonify({"error":"message is required"}),400
    return jsonify(engine.answer(message,data.get("history",[]),bool(data.get("use_llm",False))))

@app.get("/api/evaluate")
def api_evaluate(): return jsonify(evaluate())

@app.get("/api/cache")
def api_cache(): return jsonify(engine.cache.stats())

@app.get("/api/health")
def health(): return jsonify({"status":"ok","service":"rag-evaluation-dashboard","day":38})

if __name__=="__main__": app.run(debug=True)
