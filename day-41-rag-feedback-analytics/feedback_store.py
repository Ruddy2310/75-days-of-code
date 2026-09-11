import json
from collections import Counter
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

    def all(self):
        return list(reversed(self._read()))

    def add(self, query, answer, rating, quality_score=None):
        rating = rating.strip().lower()
        if rating not in {"up", "down"}:
            raise ValueError("rating must be up or down")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query.strip(),
            "answer": answer.strip(),
            "rating": rating,
            "quality_score": quality_score
        }
        records = self._read()
        records.append(record)
        self._write(records)
        return record

    def analytics(self):
        records = self._read()
        total = len(records)
        helpful = sum(r.get("rating") == "up" for r in records)
        scores = [float(r["quality_score"]) for r in records
                  if r.get("quality_score") is not None]

        daily = Counter()
        for r in records:
            try:
                day = datetime.fromisoformat(r["timestamp"]).date().isoformat()
                daily[day] += 1
            except (ValueError, TypeError):
                pass

        low = [
            {"query": r.get("query", ""), "quality_score": r.get("quality_score")}
            for r in reversed(records) if r.get("rating") == "down"
        ][:10]

        return {
            "total_feedback": total,
            "helpful": helpful,
            "not_helpful": total - helpful,
            "helpful_rate": round(helpful * 100 / total, 1) if total else 0,
            "average_quality": round(sum(scores) / len(scores), 3) if scores else None,
            "daily_feedback": dict(sorted(daily.items())),
            "low_rated_queries": low
        }
