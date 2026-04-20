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


# Baseline 1: Unimodal 2D-CNN autoencoder

class Unimodal2DCNN(nn.Module):
    """
    Simplest baseline: standard 2D-CNN autoencoder treating spectral bands
    as channels. Global average pooling collapses spatial information.
    """
    def __init__(self, n_bands: int = 86, latent_dim: int = 64, patch_size: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(n_bands, 128), nn.MaxPool2d(2),
            ConvBnRelu(128, 64),     nn.MaxPool2d(2),
            ConvBnRelu(64, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_enc = nn.Linear(64, latent_dim)
        ps4 = patch_size // 4
        self.fc_dec = nn.Linear(latent_dim, 64 * ps4 * ps4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_bands, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._ps4 = ps4

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["iirs"]
        z = self.fc_enc(self.encoder(x).flatten(1))
        recon = self.decoder(self.fc_dec(z).view(z.size(0), 64, self._ps4, self._ps4))
        return recon, z


# Baseline 2: Unimodal 3D-CNN autoencoder

class Unimodal3DCNN(nn.Module):
    """
    3D-CNN that processes spectral dimension with depth-wise convolutions,
    then collapses to spatial features. Better spectral locality than 2D-CNN
    but still unimodal.
    """
    def __init__(self, n_bands: int = 86, latent_dim: int = 64, patch_size: int = 64):
        super().__init__()
        # Spectral feature extraction
        self.spectral_enc = nn.Sequential(
            nn.Conv3d(1, 16, (7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 1, 1)),
            nn.Conv3d(16, 32, (7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 1, 1)),
        )
        # Collapse spectral dimension to fixed size 32 via adaptive pooling
        # This avoids any ceiling/floor division issues with arbitrary n_bands
        self.spectral_pool = nn.AdaptiveAvgPool3d((32, None, None))
        spectral_out = 32
        self.spatial_enc = nn.Sequential(
            ConvBnRelu(32 * spectral_out, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_enc = nn.Linear(256, latent_dim)

        ps4 = patch_size // 4
        self.fc_dec = nn.Linear(latent_dim, 64 * ps4 * ps4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_bands, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._ps4 = ps4
        self._spectral_out = spectral_out

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["iirs"].unsqueeze(1)           # (B, 1, C, H, W)
        x = self.spectral_enc(x)                 # (B, 32, C', H, W) - C' varies
        x = self.spectral_pool(x)                # (B, 32, 32, H, W) - fixed spectral dim
        B = x.size(0)
        x = x.view(B, -1, x.size(3), x.size(4)) # (B, 32*32, H, W)
        z = self.fc_enc(self.spatial_enc(x).flatten(1))
        recon = self.decoder(self.fc_dec(z).view(z.size(0), 64, self._ps4, self._ps4))
        return recon, z


# Baseline 3: Early-fusion autoencoder

class EarlyFusionAE(nn.Module):
    """
    All modalities stacked channel-wise before encoding. Simple but loses
    modality-specific characteristics; spectral bands get diluted by DEM/FeO.
    """
    def __init__(self, n_bands: int = 86, latent_dim: int = 64, patch_size: int = 64):
        super().__init__()
        in_ch = n_bands + 2  # IIRS + DEM + FeO
        self.encoder = nn.Sequential(
            ConvBnRelu(in_ch, 128), nn.MaxPool2d(2),
            ConvBnRelu(128, 64),    nn.AdaptiveAvgPool2d(1),
        )
        self.fc_enc = nn.Linear(64, latent_dim)
        ps4 = patch_size // 4
        self.fc_dec = nn.Linear(latent_dim, 64 * ps4 * ps4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_bands, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._ps4 = ps4

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([batch["iirs"], batch["dem"], batch["feo"]], dim=1)
        z = self.fc_enc(self.encoder(x).flatten(1))
        recon = self.decoder(self.fc_dec(z).view(z.size(0), 64, self._ps4, self._ps4))
        return recon, z


# Baseline 4: Late-fusion autoencoder

class LateFusionAE(nn.Module):
    """
    Each modality encoded independently, feature vectors concatenated and
    projected. No inter-modal attention; context modalities don't guide
    spectral encoding.
    """
    def __init__(self, n_bands: int = 86, latent_dim: int = 64, patch_size: int = 64):
        super().__init__()
        self.iirs_enc = nn.Sequential(
            ConvBnRelu(n_bands, 128), nn.MaxPool2d(2),
            ConvBnRelu(128, 64),      nn.AdaptiveAvgPool2d(1),
        )
        self.aux_enc = nn.Sequential(
            ConvBnRelu(1, 16), nn.MaxPool2d(2),
            ConvBnRelu(16, 32), nn.AdaptiveAvgPool2d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        ps4 = patch_size // 4
        self.fc_dec = nn.Linear(latent_dim, 64 * ps4 * ps4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_bands, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._ps4 = ps4

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        v_iirs = self.iirs_enc(batch["iirs"]).flatten(1)
        v_dem  = self.aux_enc(batch["dem"]).flatten(1)
        v_feo  = self.aux_enc(batch["feo"]).flatten(1)
        z = self.fusion(torch.cat([v_iirs, v_dem, v_feo], dim=1))
        recon = self.decoder(self.fc_dec(z).view(z.size(0), 64, self._ps4, self._ps4))
        return recon, z


# Baseline 5: TRIAD (pooled-vector attention)

class TRIAD(nn.Module):
    """
    Original TRIAD architecture from the prior codebase.
    Cross-modal MultiheadAttention on globally-pooled 1D feature vectors.
    Included as Baseline 5 to show improvement from spatial attention.
    """
    def __init__(self, n_bands: int = 86, latent_dim: int = 64, patch_size: int = 64):
        super().__init__()
        self.iirs_enc = nn.Sequential(
            ConvBnRelu(n_bands, 128), nn.MaxPool2d(2),
            ConvBnRelu(128, 64), nn.AdaptiveAvgPool2d(1),
        )
        self.iirs_proj = nn.Linear(64, 256)
        self.aux_enc = nn.Sequential(
            ConvBnRelu(1, 16), nn.AdaptiveAvgPool2d(1),
        )
        self.aux_proj = nn.Linear(16, 256)
        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.latent_proj = nn.Linear(256, latent_dim)

        ps4 = patch_size // 4
        self.fc_dec = nn.Linear(latent_dim, 64 * ps4 * ps4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_bands, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._ps4 = ps4

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.iirs_proj(self.iirs_enc(batch["iirs"]).flatten(1)).unsqueeze(1)
        k_dem = self.aux_proj(self.aux_enc(batch["dem"]).flatten(1)).unsqueeze(1)
        k_feo = self.aux_proj(self.aux_enc(batch["feo"]).flatten(1)).unsqueeze(1)
        kv = torch.cat([k_dem, k_feo], dim=1)
        fused, _ = self.attention(q, kv, kv)
        z = self.latent_proj(fused.squeeze(1))
        recon = self.decoder(self.fc_dec(z).view(z.size(0), 64, self._ps4, self._ps4))
        return recon, z


# Proposed model: GC-SUAE (Geologically-Constrained Spectral Unmixing Autoencoder)

class SpatialSpectralEncoder(nn.Module):
    """
    Encodes IIRS hyperspectral data into a spatial feature map (h×w×C),
    preserving spatial structure via residual blocks rather than pooling.
    """
    def __init__(self, n_bands: int = 86, out_channels: int = 256):
        super().__init__()
        self.stem = ConvBnRelu(n_bands, 128, k=1, p=0)  # 1×1 spectral mixing
        self.stage1 = nn.Sequential(
            ConvBnRelu(128, 128),
            ResidualBlock(128),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            ConvBnRelu(128, out_channels),
            ResidualBlock(out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, C, H/4, W/4) feature map."""
        return self.stage2(self.stage1(self.stem(x)))


class TerrainEncoder(nn.Module):
    """
    Encodes terrain modality into a spatial feature map (H/4 × W/4 × C).

    Input: 3 channels - [normalized DEM elevation, Sobel slope magnitude,
           Sobel aspect angle] - precomputed in dataset.py as physics-derived
           features. The encoder learns to extract mineralogically-relevant
           terrain patterns from these gradient-based inputs.
    """
    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            ConvBnRelu(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            ConvBnRelu(64, out_channels),
        )

    def forward(self, dem: torch.Tensor, slope: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        """Returns (B, C, H/4, W/4) terrain feature map."""
        x = torch.cat([dem, slope, aspect], dim=1)
        return self.encoder(x)


class GeochemicalEncoder(nn.Module):
    """Encodes FeO abundance map to spatial feature map."""
    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(1, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            ConvBnRelu(32, out_channels),
            ResidualBlock(out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, feo: torch.Tensor) -> torch.Tensor:
        return self.encoder(feo)
