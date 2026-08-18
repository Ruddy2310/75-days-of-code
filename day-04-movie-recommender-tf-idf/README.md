# Day 4 — Content-Based Movie Recommender

Part of my #75DaysOfCode challenge.

## What it does
Given a movie title, recommends the top N most similar movies based on genres, keywords, cast, director, and plot overview.

## How it works
1. Combine text features into a single 'soup' string per movie.
2. Vectorize using TF-IDF.
3. Compute pairwise cosine similarity between all movies.
4. Return the movies with the highest similarity score.
