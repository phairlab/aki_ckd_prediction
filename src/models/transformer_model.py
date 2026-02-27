"""
Transformer architectures for tabular data.

TabularTransformer  — small model (64-dim, 4 heads, 2 layers)
LargeTabularTransformer — larger model with bottleneck (128-dim, 8 heads, 3 layers)
"""

import torch.nn as nn


class TabularTransformer(nn.Module):
    """Small transformer for tabular classification.

    64-dim embedding, 4 attention heads, 2 encoder layers, 0.1 dropout.
    """

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.embedding_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1

        self.input_embedding = nn.Linear(input_dim, self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_embedding(x)        # [batch, embedding_dim]
        x = x.unsqueeze(1)                 # [batch, 1, embedding_dim]
        x = self.transformer_encoder(x)    # [batch, 1, embedding_dim]
        x = x.squeeze(1)                   # [batch, embedding_dim]
        return self.classifier(x)          # [batch, num_classes]


class LargeTabularTransformer(nn.Module):
    """Larger transformer with bottleneck for high-dimensional tabular data.

    128-dim embedding, 256-dim bottleneck, 8 heads, 3 layers, 0.2 dropout, GELU.
    """

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.embedding_dim = 128
        self.bottleneck_dim = 256
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.2

        # Dimensionality reduction
        self.feature_reduction = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
            nn.Linear(self.bottleneck_dim, self.embedding_dim),
        )

        self.pre_transformer_norm = nn.LayerNorm(self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(self.embedding_dim // 4, num_classes),
        )

    def forward(self, x):
        x = self.feature_reduction(x)      # [batch, embedding_dim]
        x = self.pre_transformer_norm(x)
        x = x.unsqueeze(1)                 # [batch, 1, embedding_dim]
        x = self.transformer_encoder(x)    # [batch, 1, embedding_dim]
        x = x.squeeze(1)                   # [batch, embedding_dim]
        return self.classifier(x)          # [batch, num_classes]
