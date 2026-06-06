from __future__ import annotations

import torch
import torch.nn as nn


class TransformerSequenceClassifier(nn.Module):
    """
    Transformer encoder sequence classifier.

    Input:
        x: [batch_size, sequence_length, num_features]

    Output:
        logits: [batch_size, num_classes]
    """

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        num_classes: int = 3,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.sequence_length = sequence_length
        self.d_model = d_model

        self.input_projection = nn.Linear(
            num_features,
            d_model,
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.positional_embedding[:, : x.shape[1], :]

        encoded = self.encoder(x)

        # Use the final token because the target belongs to the
        # final candle in the sequence.
        final_token = encoded[:, -1, :]

        logits = self.classifier(final_token)

        return logits