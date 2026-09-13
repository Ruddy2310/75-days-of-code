from flask import Flask,jsonify,render_template,request
from router import route_query
app=Flask(__name__)
@app.get('/')
def home(): return render_template('index.html')
@app.post('/api/route')
def api_route():
    q=str((request.get_json(silent=True) or {}).get('query','')).strip()
    return (jsonify({'error':'query is required'}),400) if not q else jsonify(route_query(q))
@app.get('/api/examples')
def examples(): return jsonify(['What is retrieval augmented generation?','Compare vector search and keyword search','How do I implement FAISS in Python?','Explain embeddings'])
if __name__=='__main__': app.run(debug=True)
