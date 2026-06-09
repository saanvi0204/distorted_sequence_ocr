"""
model.py — CNN backbone and full OCR model (CNN → BiLSTM → Attention → CTC).
"""

import torch
import torch.nn as nn

from .config import CFG


class CNNBackbone(nn.Module):
    """
    Five-block convolutional feature extractor.

    Input:  (B, 1, H, W)  — single-channel grayscale
    Output: (B, 512, H', W')

    Height is pooled aggressively (H → H/16) while width is preserved
    (W stays constant) so that the resulting feature map columns map
    directly to character positions in the image.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 64 channels, H/2 × W/2
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2 — 128 channels, H/4 × W/4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3 — 256 channels, H/4 × W/4  (no width pool)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # height-only pool → H/8

            # Block 4 — 512 channels, H/16 × W/4  (no width pool)
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),   # height-only pool → H/16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class OCRModel(nn.Module):
    """
    Full CRNN-style OCR model with self-attention.

    Pipeline
    --------
    1. CNNBackbone       — spatial feature extraction
    2. AdaptiveAvgPool   — collapse height dimension to 1
    3. Permute           — reshape to sequence (B, T, C) where T = feature map width
    4. BiLSTM            — sequential context in both directions
    5. MultiheadAttention — attend over all timesteps; output added residually to LSTM
    6. Linear classifier  — character logits at each timestep
    7. Log-softmax + permute → (T, B, vocab_size) for CTCLoss

    Args:
        vocab_size: Total number of output classes including the CTC blank token.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.cnn = CNNBackbone()

        # Collapse height to 1 while keeping width (temporal dimension) intact
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=CFG.HIDDEN_SIZE,
            num_layers=CFG.NUM_LSTM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=CFG.DROPOUT,
        )

        # embed_dim = HIDDEN_SIZE * 2 because the LSTM is bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=CFG.HIDDEN_SIZE * 2,
            num_heads=CFG.ATTN_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.classifier = nn.Linear(CFG.HIDDEN_SIZE * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, H, W) → (B, 512, H', W')
        x = self.cnn(x)
        # (B, 512, H', W') → (B, 512, 1, W') → (B, 512, W')
        x = self.pool(x).squeeze(2)
        # (B, 512, W') → (B, W', 512)  —  W' becomes the time axis T
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # Residual connection: attention output is added to LSTM output
        # to stabilise training and preserve sequential information.
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = lstm_out + attn_out

        logits    = self.classifier(x)                    # (B, T, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2)                 # (T, B, vocab_size) for CTCLoss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
