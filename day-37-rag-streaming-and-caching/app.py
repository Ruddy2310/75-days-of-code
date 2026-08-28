import json,time
from flask import Flask,render_template,request,session,redirect,url_for,jsonify,Response
from rag_engine import ConversationalRAG

app=Flask(__name__)
app.secret_key="day37-development-secret"
engine=ConversationalRAG("index")

@app.route("/",methods=["GET","POST"])
def home():
    result=None
    if request.method=="POST":
        message=request.form.get("message","").strip()
        if message:
            history=session.get("history",[])
            result=engine.answer(message,history,request.form.get("use_llm")=="on")
            history += [{"role":"user","content":message},
                        {"role":"assistant","content":result["answer"]}]
            session["history"]=history[-12:]
    return render_template("index.html",history=session.get("history",[]),
                           result=result,cache=engine.cache.stats())

@app.post("/clear")
def clear():
    session.pop("history",None); return redirect(url_for("home"))

@app.post("/api/chat")
def api_chat():
    data=request.get_json(silent=True) or {}
    message=str(data.get("message","")).strip()
    if not message:return jsonify({"error":"message is required"}),400
    return jsonify(engine.answer(message,data.get("history",[]),bool(data.get("use_llm",False))))

@app.post("/api/chat/stream")
def stream():
    data=request.get_json(silent=True) or {}
    message=str(data.get("message","")).strip()
    if not message:return jsonify({"error":"message is required"}),400
    result=engine.answer(message,data.get("history",[]),bool(data.get("use_llm",False)))
    def generate():
        yield "event: metadata\n"
        yield "data: "+json.dumps({"retrieval_query":result["retrieval_query"],
                                   "sources":result["sources"],"cache":result["cache"]})+"\n\n"
        for word in result["answer"].split():
            yield "event: token\n"
            yield "data: "+json.dumps({"text":word+" "})+"\n\n"
            time.sleep(.015)
        yield "event: done\n\n"
    return Response(generate(),mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache"})

@app.get("/api/cache")
def cache(): return jsonify(engine.cache.stats())

@app.get("/api/health")
def health(): return jsonify({"status":"ok","service":"streaming-conversational-rag","day":37})

if __name__=="__main__": app.run(debug=True)
