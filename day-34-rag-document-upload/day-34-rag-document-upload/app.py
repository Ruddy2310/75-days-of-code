from pathlib import Path
from flask import Flask,render_template,request,session,redirect,url_for,flash
from werkzeug.utils import secure_filename
from build_index import build
from rag_engine import RAGEngine

app=Flask(__name__)
app.secret_key="day34-development-secret"
app.config["MAX_CONTENT_LENGTH"]=2*1024*1024
UPLOAD_DIR=Path("data/documents")
ALLOWED={".txt",".md"}
engine=None

def get_engine():
    global engine
    if engine is None: engine=RAGEngine("index")
    return engine

@app.route("/",methods=["GET","POST"])
def home():
    answer=None;sources=[];question=""
    if request.method=="POST":
        question=request.form.get("question","").strip()
        if question:
            try:
                answer,sources=get_engine().answer(question,request.form.get("use_llm")=="on")
                h=session.get("history",[])
                h.append({"question":question,"answer":answer,"sources":sources})
                session["history"]=h[-10:]
            except FileNotFoundError:
                flash("Upload a document first to build the index.")
    documents=sorted(p.name for p in UPLOAD_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in ALLOWED) if UPLOAD_DIR.exists() else []
    return render_template("index.html",answer=answer,sources=sources,question=question,
                           history=session.get("history",[]),documents=documents)

@app.post("/upload")
def upload():
    file=request.files.get("document")
    if not file or not file.filename:
        flash("Choose a TXT or Markdown document."); return redirect(url_for("home"))
    filename=secure_filename(file.filename)
    if not filename or Path(filename).suffix.lower() not in ALLOWED:
        flash("Only .txt and .md files are supported."); return redirect(url_for("home"))
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    file.save(UPLOAD_DIR/filename)
    try:
        build()
        global engine; engine=RAGEngine("index")
        flash(f"{filename} uploaded and indexed successfully.")
    except Exception as e: flash(f"Indexing failed: {e}")
    return redirect(url_for("home"))

@app.post("/clear")
def clear():
    session.pop("history",None); return redirect(url_for("home"))

@app.errorhandler(413)
def too_large(_):
    flash("File is too large. Maximum size is 2 MB."); return redirect(url_for("home"))

if __name__=="__main__": app.run(debug=True)
