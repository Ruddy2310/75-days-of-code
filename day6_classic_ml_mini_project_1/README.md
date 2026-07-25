# Day 4 — Content-Based Movie Recommender System

Part of my [#75DaysOfCode](https://github.com/Ruddy2310/75-days-of-code) challenge.

## What it does
Given a movie title, recommends the top N most similar movies based on:
- Genres
- Keywords
- Cast
- Director
- Overview (plot summary)

## How it works
1. Combine the text features above into a single "soup" string per movie.
2. Vectorize the soup using **TF-IDF**.
3. Compute pairwise **cosine similarity** between all movies.
4. For a given title, return the movies with the highest similarity score.

## Example output
```
Because you watched: The Dark Knight
                  title  similarity
  The Dark Knight Rises       0.365
          Batman Begins       0.307
         Batman Returns       0.255
```

## Tech stack
- Python
- pandas
- scikit-learn (TfidfVectorizer, cosine_similarity)

## Run it
```bash
pip install pandas scikit-learn
python recommender.py
```

## Dataset
~4800 movies with metadata (genres, cast, crew, keywords, overview).

## Why this matters
This is a scaled-down version of the recommendation logic I'll need for
**EmoTunes** (my emotion-aware music recommender project) — swapping
"movie similarity" for "song similarity based on detected mood" is the
next step.

## Next steps
- [ ] Add popularity/vote-weighted re-ranking
- [ ] Try embeddings (e.g. sentence-transformers) instead of TF-IDF
- [ ] Wrap in a simple Streamlit UI
