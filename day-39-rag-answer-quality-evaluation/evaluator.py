import re
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class AnswerQualityEvaluator:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _tokens(text):
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def semantic_similarity(self, answer, reference):
        vectors = self.model.encode([answer, reference], normalize_embeddings=True)
        return float(np.dot(vectors[0], vectors[1]))

    def keyword_coverage(self, answer, reference):
        ref_tokens = self._tokens(reference)
        ans_tokens = self._tokens(answer)
        if not ref_tokens:
            return 1.0
        return len(ref_tokens & ans_tokens) / len(ref_tokens)

    @staticmethod
    def completeness(answer):
        text = answer.strip()
        if not text:
            return 0.0
        score = 0.4
        if len(text.split()) >= 8:
            score += 0.3
        if any(p in text for p in [".", "!", "?"]):
            score += 0.2
        if len(text.split()) >= 20:
            score += 0.1
        return min(score, 1.0)

    def evaluate(self, question, answer, reference):
        semantic = self.semantic_similarity(answer, reference)
        keyword = self.keyword_coverage(answer, reference)
        complete = self.completeness(answer)

        # Keep all metrics in [0, 1].
        semantic = max(0.0, min(1.0, semantic))
        score = (0.60 * semantic) + (0.25 * keyword) + (0.15 * complete)

        return {
            "question": question,
            "answer": answer,
            "reference": reference,
            "semantic_similarity": round(semantic, 4),
            "keyword_coverage": round(keyword, 4),
            "completeness": round(complete, 4),
            "quality_score": round(score, 4),
        }
