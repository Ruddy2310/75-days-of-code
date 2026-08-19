from pathlib import Path

def load_documents(directory):
    docs = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append({"source": str(path).replace("\\", "/"), "text": text})
    return docs
