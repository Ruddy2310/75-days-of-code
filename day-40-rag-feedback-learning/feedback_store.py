import json
from datetime import datetime, timezone
from pathlib import Path

class FeedbackStore:
    def __init__(self, path="data/feedback.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, records):
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def add(self, query, answer, rating, quality_score=None, sources=None):
        rating = rating.strip().lower()
        if rating not in {"up", "down"}:
            raise ValueError("rating must be up or down")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query.strip(),
            "answer": answer.strip(),
            "rating": rating,
            "quality_score": quality_score,
            "sources": sources or []
        }
        records = self._read()
        records.append(record)
        self._write(records)
        return record

    def all(self):
        return list(reversed(self._read()))

    def stats(self):
        records = self._read()
        total = len(records)
        positive = sum(r.get("rating") == "up" for r in records)
        scores = [float(r["quality_score"]) for r in records if r.get("quality_score") is not None]
        return {
            "total": total,
            "positive": positive,
            "negative": total - positive,
            "positive_rate": round(positive * 100 / total, 1) if total else 0,
            "average_quality_score": round(sum(scores) / len(scores), 3) if scores else None
        }
