from src.dataset import make_fallback_dataset, build_vocab, TextDataset, collate_batch
from src.model import LSTMSentimentClassifier
from torch.utils.data import DataLoader


rows = make_fallback_dataset(20)
vocab = build_vocab(rows)
dataset = TextDataset(rows, vocab)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)

x, lengths, y = next(iter(loader))
model = LSTMSentimentClassifier(len(vocab))
logits = model(x, lengths)

assert logits.shape == y.shape
print("Smoke test passed.")
