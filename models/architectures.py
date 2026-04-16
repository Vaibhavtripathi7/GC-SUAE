"""
Model architectures: five ablation baselines plus the proposed GC-SUAE.

  1. Unimodal2DCNN  - 2D-CNN autoencoder, IIRS only, global-pooled latent
  2. Unimodal3DCNN  - 3D-CNN spectral-spatial autoencoder, IIRS only
  3. EarlyFusionAE  - channel-stack all modalities before encoding
  4. LateFusionAE   - encode separately, concatenate, project
  5. TRIAD          - pooled-vector MultiheadAttention fusion
  6. GC_SUAE        - spatial cross-attention + multi-head LMM decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# Shared building blocks

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """2D residual block for spatial feature extraction."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SpectralAngleMapperLoss(nn.Module):
    """
    Spectral Angle Mapper loss - measures angular distance between
    reconstructed and true spectra. Invariant to illumination scaling.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W) or (B, C)
        p = pred.reshape(pred.size(0), pred.size(1), -1)
        t = target.reshape(target.size(0), target.size(1), -1)
        dot = (p * t).sum(dim=1)
        norm_p = torch.norm(p, dim=1).clamp(min=1e-8)
        norm_t = torch.norm(t, dim=1).clamp(min=1e-8)
        cos = (dot / (norm_p * norm_t)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos).mean()
