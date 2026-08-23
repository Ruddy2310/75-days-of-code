from flask import Flask,render_template,request,session,redirect
from hybrid_rag import RAGEngine
app=Flask(__name__); app.secret_key='day33-development-secret'; engine=RAGEngine('index')
@app.route('/',methods=['GET','POST'])
def home():
    answer=None; sources=[]; q=''
    if request.method=='POST':
        q=request.form.get('question','').strip()
        if q:
            answer,sources=engine.answer(q,request.form.get('use_llm')=='on'); h=session.get('history',[]); h.append({'question':q,'answer':answer,'sources':sources}); session['history']=h[-10:]
    return render_template('index.html',answer=answer,sources=sources,question=q,history=session.get('history',[]))
@app.post('/clear')
def clear(): session.pop('history',None); return redirect('/')
if __name__=='__main__': app.run(debug=True)
