import re
import random
from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


POSITIVE = [
    "amazing", "excellent", "fantastic", "wonderful", "great",
    "brilliant", "love", "loved", "enjoyed", "best", "beautiful",
    "funny", "powerful", "interesting", "perfect", "superb"
]

NEGATIVE = [
    "awful", "terrible", "boring", "bad", "worst", "hate",
    "hated", "disliked", "poor", "weak", "dull", "annoying",
    "predictable", "confusing", "disappointing", "waste"
]


def tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def make_fallback_dataset(n=1000, seed=42):
    rng = random.Random(seed)
    rows = []

    positive_templates = [
        "The movie was {word} and I really enjoyed it.",
        "I {word} this film because the story was wonderful.",
        "What an {word} movie with a beautiful performance.",
        "The actors were great and the movie was {word}.",
    ]
    negative_templates = [
        "The movie was {word} and I really disliked it.",
        "I {word} this film because the story was terrible.",
        "What a {word} movie with a weak performance.",
        "The actors were bad and the movie was {word}.",
    ]

    for _ in range(n // 2):
        rows.append((rng.choice(positive_templates).format(word=rng.choice(POSITIVE)), 1))
        rows.append((rng.choice(negative_templates).format(word=rng.choice(NEGATIVE)), 0))

    rng.shuffle(rows)
    return rows


def load_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'review' and 'sentiment' columns.")

    rows = []
    for _, row in df.dropna(subset=["review", "sentiment"]).iterrows():
        label = 1 if str(row["sentiment"]).strip().lower() in {"positive", "1", "pos"} else 0
        rows.append((str(row["review"]), label))
    return rows


def build_vocab(rows, max_vocab=10000, min_freq=1):
    counter = Counter()
    for text, _ in rows:
        counter.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab:
            break
        vocab[word] = len(vocab)
    return vocab


@dataclass
class TextDataset(Dataset):
    rows: list
    vocab: dict
    max_len: int = 100

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        tokens = tokenize(text)[:self.max_len]
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        if not ids:
            ids = [self.vocab["<UNK>"]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, lengths, labels
