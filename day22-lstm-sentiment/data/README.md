# Data

The project is intentionally runnable without downloading a dataset.

## Optional real IMDB dataset

Add an IMDB CSV named:

```text
IMDB Dataset.csv
```

inside this directory.

Required columns:

- `review`
- `sentiment`

Example:

```csv
review,sentiment
"This movie was amazing and emotional","positive"
"I disliked the movie and found it boring","negative"
```

The CSV is ignored by Git so large datasets are not accidentally committed.
