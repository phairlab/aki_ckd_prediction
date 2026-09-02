"""
Transformer architectures for tabular data.

Three architectures are available.

RowTokenTransformer (formerly TabularTransformer)
    The architecture used in the original submission. The whole feature vector
    is projected to a single embedding, unsqueezed to sequence length 1, and
    passed through a TransformerEncoder.

    IMPORTANT, and worth stating plainly in the manuscript: with a sequence of
    length 1, self-attention is a no-op. Softmax over a single key returns a
    weight of exactly 1, so the attention sublayer reduces to the value
    projection and the block collapses to a position-wise feedforward network
    with residual connections and layer norm. This model is an MLP. It is kept
    because it is what the submitted results were computed with, and dropping
    it would make the resubmission non-comparable -- but it should not be
    described as exploiting self-attention over clinical variables.

FeatureTokenTransformer
    A genuine tabular transformer in the FT-Transformer style. Each of the d
    input features is embedded as its own token (per-feature weight and bias),
    a learned [CLS] token is prepended, and self-attention runs over the
    resulting length-(d+1) sequence. Attention therefore models interactions
    between individual clinical variables, which is the mechanism the
    manuscript claims to be testing. Classification reads the [CLS] token.

LargeTabularTransformer
    The original wide/bottleneck variant, also sequence length 1 and therefore
    also an MLP. Retained for backwards compatibility.

Reference for the feature-tokenised design:
    Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
    NeurIPS 2021.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sequence-length-1 architectures (functionally MLPs)
# ---------------------------------------------------------------------------

class RowTokenTransformer(nn.Module):
    """Original submitted architecture: one token per patient row.

    Defaults reproduce the originally submitted model exactly
    (64-dim, 4 heads, 2 layers, dropout 0.1).
    """

    def __init__(self, input_dim, num_classes=2, embedding_dim=64, num_heads=4,
                 num_layers=2, dropout=0.1, ff_multiplier=4, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_embedding = nn.Linear(input_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * ff_multiplier,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_embedding(x)        # [batch, embedding_dim]
        x = x.unsqueeze(1)                 # [batch, 1, embedding_dim]
        x = self.transformer_encoder(x)    # attention over one token: identity-weighted
        x = x.squeeze(1)                   # [batch, embedding_dim]
        return self.classifier(x)


# Backwards-compatible alias. Existing checkpoints and imports keep working.
TabularTransformer = RowTokenTransformer


class LargeTabularTransformer(nn.Module):
    """Wider variant with a bottleneck. Also sequence length 1 (see module docstring)."""

    def __init__(self, input_dim, num_classes=2, embedding_dim=128, bottleneck_dim=256,
                 num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.feature_reduction = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embedding_dim),
        )
        self.pre_transformer_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embedding_dim // 4, num_classes),
        )

    def forward(self, x):
        x = self.feature_reduction(x)
        x = self.pre_transformer_norm(x)
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Genuine feature-tokenised transformer
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Embed each scalar feature as its own token.

    Token j for a patient is  x_j * W_j + b_j,  with W_j and b_j learned per
    feature, giving a [batch, n_features, d_model] sequence. A learned [CLS]
    token is prepended, so the sequence attention operates over is length
    n_features + 1.
    """

    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        # Uniform(-1/sqrt(d), 1/sqrt(d)), as in the FT-Transformer reference
        bound = 1.0 / math.sqrt(d_model)
        for p in (self.weight, self.bias, self.cls_token):
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x):                       # x: [batch, n_features]
        tokens = x.unsqueeze(-1) * self.weight + self.bias   # [batch, n_feat, d_model]
        cls = self.cls_token.expand(x.shape[0], -1, -1)      # [batch, 1, d_model]
        return torch.cat([cls, tokens], dim=1)               # [batch, n_feat+1, d_model]


class FeatureTokenTransformer(nn.Module):
    """FT-Transformer style model: self-attention across individual features.

    Unlike RowTokenTransformer, attention here is meaningful -- the sequence has
    one position per clinical variable, so the attention weights describe
    interactions between variables. This is the architecture that actually
    tests the manuscript's stated hypothesis about complex models capturing
    variable interactions.

    Memory scales with n_features^2 per attention layer, so d_model should be
    kept modest when the input has 100+ columns.
    """

    def __init__(self, input_dim, num_classes=2, d_model=64, num_heads=8,
                 num_layers=3, dropout=0.1, ff_multiplier=2, activation="gelu"):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.tokenizer = FeatureTokenizer(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ff_multiplier),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,          # pre-norm: more stable on small tabular data
        )
        # enable_nested_tensor=False: nested tensors are incompatible with
        # norm_first, and torch warns about it on every construction.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)              # [batch, n_feat+1, d_model]
        encoded = self.transformer_encoder(tokens)
        return self.head(encoded[:, 0])         # read the [CLS] position


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "row_token": RowTokenTransformer,      # original submission (an MLP)
    "small": RowTokenTransformer,          # legacy alias for model_size="small"
    "large": LargeTabularTransformer,      # legacy alias for model_size="large"
    "feature_token": FeatureTokenTransformer,
}


def build_transformer(input_dim, architecture="row_token", num_classes=2, **kwargs):
    """Construct a transformer by name, passing through architecture kwargs.

    Unknown kwargs are dropped rather than raising, so one hyperparameter dict
    can be shared across architectures with different knobs.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from {sorted(ARCHITECTURES)}"
        )
    cls = ARCHITECTURES[architecture]

    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(input_dim=input_dim, num_classes=num_classes, **filtered)


def count_parameters(model):
    """Trainable parameter count, for reporting model capacity in the methods."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
