"""
Day 4 - #75DaysOfCode
Content-Based Movie Recommender System

Given a movie title, recommends similar movies based on combined
content features: genres, keywords, cast, director, and overview.

Approach: TF-IDF vectorization + Cosine Similarity
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="movies.csv"):
    df = pd.read_csv(path)
    # Keep only the columns we need for content-based filtering
    features = ["genres", "keywords", "cast", "director", "overview", "title"]
    df = df[features].copy()
    for col in ["genres", "keywords", "cast", "director", "overview"]:
        df[col] = df[col].fillna("")
    return df


def create_soup(df):
    """Combine relevant text features into a single string per movie."""
    def combine(row):
        return f"{row['genres']} {row['keywords']} {row['cast']} {row['director']} {row['overview']}"

    df["soup"] = df.apply(combine, axis=1)
    return df


def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def recommend(title, df, cosine_sim, top_n=5):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title.lower())]
    if matches.empty:
        return f"No movie found matching '{title}'. Try another title."

    idx = matches.index[0]
    matched_title = df.loc[idx, "title"]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 3) for i in sim_scores]

    result = df.loc[movie_indices, ["title"]].copy()
    result["similarity"] = scores
    result = result.reset_index(drop=True)
    return matched_title, result


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    df = create_soup(df)

    print("Building similarity matrix (TF-IDF + cosine similarity)...")
    cosine_sim = build_similarity_matrix(df)

    demo_titles = ["The Dark Knight", "Avatar", "Inception"]

    for t in demo_titles:
        result = recommend(t, df, cosine_sim, top_n=5)
        if isinstance(result, str):
            print(result)
            continue
        matched_title, recs = result
        print(f"\nBecause you watched: {matched_title}")
        print(recs.to_string(index=False))

    print("\nDone. Try recommend('Your Favorite Movie', df, cosine_sim) in a notebook/REPL.")
