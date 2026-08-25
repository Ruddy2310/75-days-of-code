from flask import Flask,render_template,request,session,redirect,url_for,jsonify
from rag_engine import RAGEngine

app=Flask(__name__)
app.secret_key="day35-development-secret"
engine=RAGEngine("index")

@app.route("/",methods=["GET","POST"])
def home():
    answer=None;sources=[];question=""
    if request.method=="POST":
        question=request.form.get("question","").strip()
        if question:
            answer,sources=engine.answer(question,request.form.get("use_llm")=="on")
            h=session.get("history",[])
            h.append({"question":question,"answer":answer,"sources":sources})
            session["history"]=h[-10:]
    return render_template("index.html",answer=answer,sources=sources,
                           question=question,history=session.get("history",[]))

@app.post("/clear")
def clear():
    session.pop("history",None); return redirect(url_for("home"))

@app.post("/api/ask")
def api_ask():
    data=request.get_json(silent=True) or {}
    question=str(data.get("question","")).strip()
    if not question:return jsonify({"error":"question is required"}),400
    answer,sources=engine.answer(question,bool(data.get("use_llm",False)))
    return jsonify({"question":question,"answer":answer,"sources":sources})

@app.get("/api/documents")
def api_documents():
    engine.load()
    docs=sorted({x["source"] for x in engine.chunks})
    return jsonify({"documents":docs})

if __name__=="__main__": app.run(debug=True)
