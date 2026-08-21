from flask import Flask, render_template, request, session, redirect, url_for
from rag_engine import RAGEngine

app=Flask(__name__)
app.secret_key="day31-development-secret"
engine=RAGEngine("index")

@app.route("/",methods=["GET","POST"])
def home():
    answer=None; sources=[]; question=""
    history=session.get("history",[])
    if request.method=="POST":
        question=request.form.get("question","").strip()
        if question:
            answer,sources=engine.answer(question,request.form.get("use_llm")=="on")
            history.append({"question":question,"answer":answer,"sources":sources})
            session["history"]=history[-10:]
    return render_template("index.html",history=session.get("history",[]),
                           answer=answer,sources=sources,question=question)

@app.post("/clear")
def clear():
    session.pop("history",None)
    return redirect(url_for("home"))

if __name__=="__main__":
    app.run(debug=True)
