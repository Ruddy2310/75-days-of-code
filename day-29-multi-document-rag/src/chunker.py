def chunk_documents(documents, chunk_size=700, overlap=120):
    chunks = []
    for doc in documents:
        start = 0
        chunk_id = 0
        text = doc["text"]
        while start < len(text):
            end = min(start + chunk_size, len(text))
            part = text[start:end].strip()
            if part:
                chunks.append({"text": part, "source": doc["source"], "chunk_id": chunk_id})
            if end == len(text): break
            start = max(end - overlap, start + 1)
            chunk_id += 1
    return chunks
