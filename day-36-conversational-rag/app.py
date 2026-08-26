from flask import Flask,render_template,request,session,redirect,url_for,jsonify
from conversational_rag import ConversationalRAG
app=Flask(__name__);app.secret_key="day36-development-secret";engine=ConversationalRAG("index")
@app.route("/",methods=["GET","POST"])
def home():
 result=None
 if request.method=="POST":
  msg=request.form.get("message","").strip()
  if msg:
   h=session.get("history",[]);result=engine.chat(msg,h,request.form.get("use_llm")=="on");h += [{"role":"user","content":msg},{"role":"assistant","content":result["answer"]}];session["history"]=h[-12:]
 return render_template("index.html",history=session.get("history",[]),result=result)
@app.post("/clear")
def clear():session.pop("history",None);return redirect(url_for("home"))
@app.post("/api/chat")
def api_chat():
 d=request.get_json(silent=True) or {};msg=str(d.get("message","")).strip()
 if not msg:return jsonify({"error":"message is required"}),400
 return jsonify(engine.chat(msg,d.get("history",[]),bool(d.get("use_llm",False))))
@app.get("/api/health")
def health():return jsonify({"status":"ok","service":"conversational-rag","day":36})
if __name__=="__main__":app.run(debug=True)
