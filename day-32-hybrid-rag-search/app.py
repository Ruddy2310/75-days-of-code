from flask import Flask,render_template,request,session,redirect,url_for
from hybrid_rag import HybridRAG
app=Flask(__name__); app.secret_key="day32-development-secret"
engine=HybridRAG("index")

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

if __name__=="__main__": app.run(debug=True)
