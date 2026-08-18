"""
dataset.py
----------
Data pipeline for the LSTM sentiment classifier.

Two sources are supported:
1. A synthetic-but-realistic IMDB-style review generator (default, offline,
   deterministic with a seed) — lets the whole project run end-to-end with
   zero external downloads.
2. The real IMDB dataset via torchtext / HuggingFace `datasets`, enabled by
   flipping `USE_REAL_IMDB = True` below (one-line change, see README).

Both paths funnel into the same Vocab + collate_fn so the model code never
needs to know which one is active.
"""

import random
import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Flip this to True to use the real IMDB dataset instead of the synthetic
# generator below (requires `pip install datasets` and network access).
USE_REAL_IMDB = False

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

# ---------------------------------------------------------------------------
# Synthetic-but-realistic review generator
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = [
    "amazing", "brilliant", "fantastic", "wonderful", "touching", "hilarious",
    "gripping", "beautiful", "outstanding", "compelling", "charming", "superb",
    "delightful", "masterful", "captivating", "heartwarming", "excellent",
    "unforgettable", "engaging", "clever",
]
_NEGATIVE_WORDS = [
    "boring", "terrible", "awful", "dull", "predictable", "tedious",
    "disappointing", "clunky", "forgettable", "wooden", "flat", "annoying",
    "pointless", "overlong", "lazy", "incoherent", "shallow", "cheesy",
    "unwatchable", "bland",
]
_NEUTRAL_FILLER = [
    "movie", "film", "story", "plot", "director", "cast", "acting", "scene",
    "script", "ending", "character", "performance", "soundtrack", "pacing",
    "dialogue", "sequel", "budget", "cinematography", "audience", "review",
    "the", "a", "was", "is", "this", "that", "really", "very", "so", "quite",
    "i", "we", "it", "and", "but", "with", "for", "of", "in", "to",
]

_TEMPLATES = [
    "the {adj1} {noun} had a {adj2} {noun2} and I {verb} it",
    "honestly this {noun} was {adj1}, the {noun2} felt {adj2}",
    "what a {adj1} {noun}, the {noun2} was {adj2} too",
    "I {verb} the {noun} because the {noun2} was {adj1} and {adj2}",
    "{adj1} {noun} overall, though the {noun2} felt a bit {adj2}",
]


def _generate_review(rng: random.Random, label: int) -> str:
    pool = _POSITIVE_WORDS if label == 1 else _NEGATIVE_WORDS
    template = rng.choice(_TEMPLATES)
    sentence = template.format(
        adj1=rng.choice(pool),
        adj2=rng.choice(pool),
        noun=rng.choice(_NEUTRAL_FILLER),
        noun2=rng.choice(_NEUTRAL_FILLER),
        verb=rng.choice(["loved", "enjoyed", "hated", "disliked"]),
    )
    # Bulk it out with filler words so sequence lengths vary, like real reviews.
    extra = rng.sample(_NEUTRAL_FILLER, k=rng.randint(3, 10))
    words = sentence.split() + extra
    rng.shuffle(words)
    return " ".join(words)


def generate_synthetic_dataset(
    n_samples: int = 4000, seed: int = 42
) -> List[Tuple[str, int]]:
    """Returns a list of (review_text, label) with label 1=positive, 0=negative."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        label = rng.randint(0, 1)
        data.append((_generate_review(rng, label), label))
    rng.shuffle(data)
    return data


def load_real_imdb() -> List[Tuple[str, int]]:
    """Loads the real IMDB dataset via HuggingFace `datasets` (train split)."""
    from datasets import load_dataset  # local import: optional dependency

    ds = load_dataset("imdb", split="train")
    return [(row["text"], row["label"]) for row in ds]


def load_raw_dataset(n_samples: int = 4000, seed: int = 42) -> List[Tuple[str, int]]:
    if USE_REAL_IMDB:
        return load_real_imdb()
    return generate_synthetic_dataset(n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Tokenization + vocabulary
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, texts: List[str], min_freq: int = 1, max_size: int = 20000):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))

        self.itos = [PAD_TOKEN, UNK_TOKEN]
        for word, freq in counter.most_common(max_size):
            if freq >= min_freq:
                self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        unk = self.stoi[UNK_TOKEN]
        return [self.stoi.get(tok, unk) for tok in tokenize(text)]


# ---------------------------------------------------------------------------
# PyTorch Dataset + collate function
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], vocab: Vocab, max_len: int = 200):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        ids = self.vocab.encode(text)[: self.max_len]
        if len(ids) == 0:
            ids = [self.vocab.stoi[UNK_TOKEN]]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, lengths, labels


def train_val_split(samples: List[Tuple[str, int]], val_frac: float = 0.15, seed: int = 42):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]
