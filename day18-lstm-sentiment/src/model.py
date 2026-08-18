"""
model.py
--------
A small, readable LSTM classifier for binary sentiment analysis.

Architecture:
    Embedding -> (bi-directional, multi-layer) LSTM -> mean+max pool over
    time -> Linear -> single logit (sigmoid applied via BCEWithLogitsLoss).

Packed sequences are used so padding tokens never influence the LSTM's
hidden state.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        pooled_dim = hidden_dim * directions * 2  # mean-pool + max-pool concat

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)   lengths: (batch,)
        embedded = self.dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        # outputs: (batch, seq_len, hidden_dim * directions)

        mask = (input_ids != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_outputs = outputs * mask

        # Mean pool (ignoring padding).
        summed = masked_outputs.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        mean_pooled = summed / counts

        # Max pool (ignoring padding via -inf fill).
        max_pooled = masked_outputs.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        max_pooled = torch.nan_to_num(max_pooled, neginf=0.0)

        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits
