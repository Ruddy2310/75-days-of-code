import re

ROUTES = {
    "comparison": {
        "keywords": {"compare", "comparison", "difference", "versus", "vs", "better"},
        "strategy": "hybrid",
        "reason": "Comparison questions benefit from evidence about multiple concepts."
    },
    "technical": {
        "keywords": {"code", "python", "api", "algorithm", "implementation", "function", "programming"},
        "strategy": "technical",
        "reason": "Technical questions need precise implementation-focused context."
    },
    "factual": {
        "keywords": {"what", "when", "where", "who", "define", "definition", "meaning"},
        "strategy": "semantic",
        "reason": "Factual questions are well suited to semantic retrieval."
    },
    "general": {
        "keywords": set(),
        "strategy": "semantic",
        "reason": "No strong intent signal was detected, so the default semantic strategy is used."
    }
}

def explain_query(query):
    tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
    scores = {}
    matched = {}

    for name, config in ROUTES.items():
        hits = sorted(tokens & config["keywords"])
        matched[name] = hits
        scores[name] = len(hits)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "general"

    confidence = 0.50 if best == "general" else min(1.0, 0.50 + scores[best] * 0.15)
    config = ROUTES[best]

    trace = [
        f"Tokenized query into {len(tokens)} unique terms.",
        f"Detected {scores[best]} matching intent keyword(s) for '{best}'.",
        f"Selected '{best}' as the strongest route.",
        f"Mapped route '{best}' to the '{config['strategy']}' retrieval strategy."
    ]

    return {
        "query": query,
        "route": best,
        "strategy": config["strategy"],
        "confidence": round(confidence, 2),
        "reason": config["reason"],
        "matched_keywords": matched[best],
        "route_scores": scores,
        "decision_trace": trace
    }
