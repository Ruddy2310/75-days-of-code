import torch
from torch import nn


class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64,
                 num_layers=1, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        last_hidden = hidden[-1]
        logits = self.fc(self.dropout(last_hidden)).squeeze(1)
        return logits
