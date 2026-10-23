import torch
import torch.nn as nn
import torchvision.models as models
import math

# Supported frozen backbones. "tokens" is the number of feature vectors per image,
# "grid" the spatial layout of those tokens (used for attention heatmaps) and
# "prefix" the number of non-spatial tokens in front (CLIP's CLS token).
ENCODERS = {
    "resnet50": {"feature_dim": 2048, "tokens": 49, "grid": 7, "prefix": 0},
    "clip-vit-b16": {"feature_dim": 768, "tokens": 197, "grid": 14, "prefix": 1,
                     "hf_name": "openai/clip-vit-base-patch16"},
    "clip-vit-b32": {"feature_dim": 768, "tokens": 50, "grid": 7, "prefix": 1,
                     "hf_name": "openai/clip-vit-base-patch32"},
}


class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, num_tokens=49, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class CNNEncoder(nn.Module):
    """
    Frozen backbone (ResNet50 or CLIP ViT) followed by a trainable head:
    projection -> learned positional encoding -> 2 transformer refinement layers.
    `extract_features` runs only the backbone (its output can be cached to disk),
    `encode_features` runs only the head.
    """

    def __init__(self, embed_dim=512, dropout=0.1, pretrained=True, fine_tune=False, encoder_type="resnet50",
                 load_backbone=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        spec = ENCODERS[encoder_type]
        self.grid_size = spec["grid"]
        self.num_prefix_tokens = spec["prefix"]

        backbone = None
        if not load_backbone:
            pass  # training on cached features: only the head is needed
        elif encoder_type == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            # We need the spatial features, so we remove the last two layers (avgpool and fc)
            self.resnet = nn.Sequential(*list(resnet.children())[:-2])
            backbone = self.resnet
        else:
            from transformers import CLIPVisionModel
            # CLIP weights always come from the Hub; they are not stored in our checkpoints
            self.clip = CLIPVisionModel.from_pretrained(spec["hf_name"])
            backbone = self.clip

        # Freeze or fine-tune backbone
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = fine_tune

        # Projection from the backbone's feature size to the decoder's embed_dim
        self.cnn_projection = nn.Linear(spec["feature_dim"], embed_dim)

        # Learned positional encoding for the backbone's token grid
        self.pos_encoding = PositionalEncoding2D(embed_dim, num_tokens=spec["tokens"])

        # Secondary Transformer Encoder refinement layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads=8, dim_feedforward=2048, dropout=dropout)
            for _ in range(2)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def extract_features(self, images):
        """images: (B, 3, 224, 224) -> backbone tokens (B, T, feature_dim)."""
        if self.encoder_type == "resnet50":
            # ResNet50 output: (B, 2048, 7, 7) -> (B, 49, 2048)
            features = self.resnet(images)
            return features.flatten(2).transpose(1, 2)
        return self.clip(pixel_values=images).last_hidden_state

    def encode_features(self, features):
        """Backbone tokens (B, T, feature_dim) -> refined memory (B, T, embed_dim)."""
        projected = self.pos_encoding(self.cnn_projection(features))
        for layer in self.transformer_layers:
            projected = layer(projected)
        return self.final_norm(projected)

    def forward(self, images):
        return self.encode_features(self.extract_features(images))

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, captions, encoder_out):
        embeddings = self.embedding(captions)
        embeddings = self.pos_encoder(embeddings)

        tgt_mask = self._generate_square_subsequent_mask(captions.size(1)).to(captions.device)

        decoder_out = self.transformer_decoder(
            tgt=embeddings,
            memory=encoder_out,
            tgt_mask=tgt_mask
        )

        predictions = self.fc_out(decoder_out)
        return predictions

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, pretrained=True, encoder_type="resnet50",
                 load_backbone=True):
        super().__init__()

        self.encoder = CNNEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
            pretrained=pretrained,
            encoder_type=encoder_type,
            load_backbone=load_backbone,
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, images, captions):
        encoder_out = self.encoder(images)
        predictions = self.decoder(captions, encoder_out)
        return predictions

    def trainable_state_dict(self):
        """State dict without frozen backbone weights (those are reloaded from their source)."""
        frozen = ("encoder.resnet.", "encoder.clip.")
        return {k: v for k, v in self.state_dict().items() if not k.startswith(frozen)}
