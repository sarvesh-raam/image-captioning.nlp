import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, max_h=7, max_w=7, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_h * max_w, embed_dim))

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

class HybridEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_transformer_layers=2, num_heads=8,
                 dim_feedforward=2048, dropout=0.1, pretrained=True, fine_tune_cnn=False):
        super().__init__()
        self.embed_dim = embed_dim

        # ResNet50 CNN (Pretrained on ImageNet)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        modules = list(resnet.children())[:-2] # Remove global pool and FC
        self.resnet = nn.Sequential(*modules)
        self.resnet_dim = 2048
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        for param in self.resnet.parameters():
            param.requires_grad = fine_tune_cnn

        # Projection to Transformer embedding dimension
        self.cnn_projection = nn.Linear(self.resnet_dim, embed_dim)

        # 2D Positional encoding for spatial features
        self.pos_encoding = PositionalEncoding2D(embed_dim, 7, 7, dropout)

        # Transformer layers for global visual context
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.cnn_projection.weight)
        nn.init.zeros_(self.cnn_projection.bias)

    def forward(self, images):
        batch_size = images.size(0)
        # Extract features: (B, 2048, 7, 7)
        cnn_features = self.resnet(images)
        cnn_features = self.adaptive_pool(cnn_features)
        
        # Reshape to sequence: (B, 49, 2048)
        cnn_features = cnn_features.permute(0, 2, 3, 1)
        cnn_features = cnn_features.reshape(batch_size, -1, self.resnet_dim)
        
        # Project and encode
        features = self.cnn_projection(cnn_features)
        features = self.pos_encoding(features)

        # Transformer encoding
        for transformer_layer in self.transformer_layers:
            features = transformer_layer(features)

        features = self.final_norm(features)
        return features

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
                 dim_feedforward=2048, dropout=0.1, use_hybrid_encoder=True,
                 encoder_transformer_layers=2):
        super().__init__()

        self.encoder = HybridEncoder(
            embed_dim=embed_dim, 
            num_transformer_layers=encoder_transformer_layers,
            num_heads=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            pretrained=True
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
