import re
from collections import Counter
STOPWORDS={"the","a","an","is","are","was","were","what","why","how","can","to","of","and","in","for","on","with","this","that","it","from","as","do","does","be","use","using","about"}
def build_insights(records):
 total=len(records); helpful=sum(r.get("rating")=="up" for r in records)
 scores=[float(r["quality_score"]) for r in records if r.get("quality_score") is not None]
 buckets=Counter()
 for s in scores: buckets["low" if s<.50 else "medium" if s<.75 else "high"]+=1
 weak=[r for r in records if r.get("rating")=="down" or (r.get("quality_score") is not None and float(r["quality_score"])<.60)]
 words=Counter()
 for r in weak:
  words.update(w for w in re.findall(r"[a-zA-Z]{3,}",r.get("query","").lower()) if w not in STOPWORDS)
 rec=[]
 if total and helpful/total<.75: rec.append("Review retrieval quality because the helpful rate is below 75%.")
 if buckets["low"]: rec.append("Inspect low-quality answers and add better source documents or chunks.")
 if words: rec.append("Prioritize frequently weak topics: "+", ".join(w for w,_ in words.most_common(5))+".")
 if not rec: rec.append("Feedback signals look healthy; continue collecting evaluations.")
 return {"total_feedback":total,"helpful":helpful,"not_helpful":total-helpful,"helpful_rate":round(helpful*100/total,1) if total else 0,"average_quality":round(sum(scores)/len(scores),3) if scores else None,"quality_buckets":dict(buckets),"weak_query_count":len(weak),"weak_topics":words.most_common(8),"recommendations":rec}
