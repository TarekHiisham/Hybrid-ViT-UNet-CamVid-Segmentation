"""
AutoScene – Model Definitions
==============================
Contains:
  • ViTransformerEncoder  – custom ViT-based encoder with skip features
  • ViTUNetDecoder        – U-Net decoder with skip connections
  • ViTUNet               – Full segmentation network (encoder + decoder)

Training config used:
    img_size   = 384
    patch_size = 6
    embed_dim  = 96
    n_heads    = 6
    num_classes= 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Encoder building blocks
# ──────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2)          # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)     # (B, n_patches, embed_dim)
        return x


class PosEmbedding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, x):
        return x + self.pos_encoding


class DownSample(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 2 * embed_dim,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim * 2),
            nn.GELU()
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)           # (B, 2C, H/2, W/2)
        x = x.flatten(2).transpose(1, 2)
        return x


class EncoderStage(nn.Module):
    def __init__(self, embed_dim, n_heads, num_layers):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.stage = nn.TransformerEncoder(
            self.layer, num_layers=num_layers,
            enable_nested_tensor=False
        )

    def forward(self, x):
        return self.stage(x)


class ViTransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size=6, in_channels=3,
                 embed_dim=96, n_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_embed   = PosEmbedding(self.n_patches, embed_dim)

        self.stages = nn.Sequential(
            # Stage 1
            EncoderStage(embed_dim, n_heads, num_layers=6),
            DownSample(embed_dim),
            # Stage 2
            EncoderStage(2 * embed_dim, n_heads, num_layers=4),
            DownSample(2 * embed_dim),
            # Stage 3
            EncoderStage(4 * embed_dim, n_heads, num_layers=2),
            DownSample(4 * embed_dim),
        )

    def forward(self, x):
        feats = []
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        for module in self.stages:
            x = module(x)
            if isinstance(module, EncoderStage):
                B, N, C = x.shape
                H = W = int(N ** 0.5)
                feats.append(x.transpose(1, 2).reshape(B, C, H, W))

        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x, feats


# ──────────────────────────────────────────────
# Decoder building blocks
# ──────────────────────────────────────────────

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class ViTUNetDecoder(nn.Module):
    def __init__(self, in_dim, output_size):
        super().__init__()
        self.output_size = output_size

        self.stages = nn.Sequential(
            # Stage 1
            UpSample(in_dim,       in_dim // 2),
            nn.Conv2d(in_dim,      in_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(),
            # Stage 2
            UpSample(in_dim // 2,  in_dim // 4),
            nn.Conv2d(in_dim // 2, in_dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU(),
            # Stage 3
            UpSample(in_dim // 4,  in_dim // 8),
            nn.Conv2d(in_dim // 4, in_dim // 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 8),
            nn.ReLU(),
            # Head
            nn.Conv2d(in_dim // 8, 32, kernel_size=1),
        )

    def forward(self, x, feats):
        skips    = list(reversed(feats))
        skip_idx = 0
        for module in self.stages:
            x = module(x)
            if isinstance(module, UpSample):
                x = torch.cat([x, skips[skip_idx]], dim=1)
                skip_idx += 1
        # Interpolate back to original/display size
        x = F.interpolate(x, size=self.output_size,
                          mode='bilinear', align_corners=False)
        return x


# ──────────────────────────────────────────────
# Full network
# ──────────────────────────────────────────────

class ViTUNet(nn.Module):
    """
    ViT Encoder + U-Net Decoder segmentation network.

    The network always resizes input to (img_size × img_size) internally,
    then the decoder's final interpolation is set to that same size.
    The app is responsible for resizing back to the ORIGINAL image dimensions
    AFTER calling this model (see inference.py).
    """

    def __init__(self, img_size=384, in_channels=3,
                 embed_dim=96, patch_size=6, n_heads=6):
        super().__init__()
        self.img_size = img_size
        self.encoder  = ViTransformerEncoder(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim, n_heads=n_heads
        )
        self.decoder  = ViTUNetDecoder(
            in_dim=embed_dim * 8,
            output_size=(img_size, img_size)
        )

    def forward(self, x):
        enc_out, feats = self.encoder(x)
        logits = self.decoder(enc_out, feats)   # (B, 32, img_size, img_size)
        return logits
