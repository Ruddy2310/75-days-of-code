# Day 28 — RAG Chatbot (Incremental Build #3)

Part of my **75 Days of Code Challenge**.

## 🎯 Goal

Build the third incremental version of a Retrieval-Augmented Generation (RAG) chatbot.

This version adds:

- Document loading from a local `data/` folder
- Text chunking with overlap
- Semantic embeddings using `sentence-transformers`
- Vector search using FAISS
- Top-k relevant context retrieval
- Optional LLM answer generation through Ollama
- A simple command-line chat interface
- Source references for retrieved chunks

## 🧠 How RAG Works

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Vector Store
   ↓
User Question
   ↓
Semantic Retrieval
   ↓
Relevant Context
   ↓
LLM / Local Answer Generator
   ↓
Final Answer + Sources
```

## 📁 Project Structure

```text
day-28-rag-chatbot/
├── app.py
├── rag_engine.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── ai_notes.txt
```

## ⚙️ Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run

```bash
python app.py
```

The first run downloads the embedding model.

### Optional: use Ollama for generated answers

Install Ollama separately and make sure a local model is available, for example:

```bash
ollama pull llama3.2:3b
```

Then run:

```bash
python app.py --ollama
```

If Ollama is not available, the project automatically falls back to a retrieval-based answer that shows the most relevant context.

## 💬 Example

```text
You: What is artificial intelligence?

Assistant:
Artificial intelligence is the field of computer science focused on
building systems that can perform tasks associated with human intelligence.

Sources:
[1] ai_notes.txt
[2] ai_notes.txt
```

## 🛠️ Technologies

- Python
- Sentence Transformers
- FAISS
- NumPy
- Ollama (optional)

## 📌 Day 28 Learning

Today I learned how a RAG pipeline connects document retrieval with an answer-generation layer. The important idea is that the model does not have to rely only on its trained knowledge: relevant information can first be retrieved from an external knowledge base and then supplied as context.

## 🚀 Next Step — Day 29

Improve retrieval quality, add conversation memory, and experiment with better prompt engineering and evaluation.

---
**75 Days of Code Challenge | Day 28**
