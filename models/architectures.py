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


class SpatialCrossAttention(nn.Module):
    """
    Spatial cross-attention between the IIRS feature map (query) and an
    auxiliary modality map (DEM or FeO context), over full H×W feature maps
    rather than pooled vectors. Each query position attends over all context
    positions: softmax(QK^T / sqrt(d_head)) V, with a residual back to the query
    so IIRS features are preserved.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.q_proj   = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.k_proj   = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.v_proj   = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.out_proj = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.norm     = nn.GroupNorm(min(8, d_model // 16), d_model)

    def forward(
        self,
        query:   torch.Tensor,  # (B, C, H, W) - IIRS spatial features
        context: torch.Tensor,  # (B, C, H, W) - auxiliary modality features
    ) -> torch.Tensor:
        B, C, H, W = query.shape

        q = self.q_proj(query)    # (B, C, H, W)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Split heads: (B, C, H, W) → (B*n_heads, HW, d_head)
        def to_heads(x):
            return (x.view(B, self.n_heads, self.d_head, H * W)
                     .permute(0, 1, 3, 2)            # (B, h, HW, d_head)
                     .reshape(B * self.n_heads, H * W, self.d_head))

        q_h = to_heads(q)
        k_h = to_heads(k)
        v_h = to_heads(v)

        # Scaled dot-product attention
        attn = torch.softmax(
            torch.bmm(q_h, k_h.transpose(1, 2)) * self.scale, dim=-1
        )                                             # (B*h, HW, HW)
        out_h = torch.bmm(attn, v_h)                 # (B*h, HW, d_head)

        # Merge heads back to spatial map
        out = (out_h.reshape(B, self.n_heads, H * W, self.d_head)
                    .permute(0, 1, 3, 2)              # (B, h, d_head, HW)
                    .reshape(B, C, H, W))

        # Residual + norm
        return self.norm(self.out_proj(out) + query)


class LMMDecoder(nn.Module):
    """
    Physics-Constrained Decoder implementing the Linear Mixing Model (LMM).

    The LMM states: r = E · a
    where:
        r ∈ [0,1]^{n_bands}  - observed reflectance at a pixel
        E ∈ [0,1]^{K×n_bands} - endmember spectral matrix (K minerals)
        a ∈ Δ^{K-1}          - abundance simplex (ANC + ASC)

    Constraints:
        ANC: a_k ≥ 0  for all k  (Abundance Non-Negativity)
        ASC: Σ_k a_k = 1         (Abundance Sum-to-One)
        E bounded to [0,1]       (physical reflectance range)

    Key design decisions:
        - softplus(·) for ANC: avoids dead-neuron problem of ReLU, smooth gradient
        - ASC via L1 normalization: exact, differentiable
        - E = sigmoid(raw_E): keeps endmembers in [0,1] without extra loss terms
        - NO sigmoid on final output: r = E·a is already in [0,1] when E∈[0,1], a∈Δ
    """
    def __init__(
        self,
        latent_dim: int = 128,
        n_minerals: int = 6,
        n_bands: int = 86,
        patch_size: int = 64,
    ):
        super().__init__()
        self.n_minerals = n_minerals
        self.n_bands    = n_bands
        ps4 = patch_size // 4
        self._ps4 = ps4

        # Latent → spatial abundance map
        self.fc_expand = nn.Linear(latent_dim, 256 * ps4 * ps4)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, n_minerals, 4, stride=2, padding=1),
        )

        # Raw (unconstrained) endmember parameter; E = sigmoid(raw_E) ensures [0,1]
        # Initialize to approximate flat spectrum (sigmoid(0) = 0.5)
        self.raw_endmember_matrix = nn.Parameter(
            torch.zeros(n_minerals, n_bands)
        )

    @property
    def endmember_matrix(self) -> torch.Tensor:
        """Constrained endmember matrix E ∈ [0,1]^{K × n_bands}."""
        return torch.sigmoid(self.raw_endmember_matrix)

    def get_abundances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns per-pixel mineral abundances: (B, n_minerals, H, W)
        satisfying ANC (softplus ≥ 0) and ASC (L1 normalized to sum = 1).
        """
        x   = self.fc_expand(z).view(z.size(0), 256, self._ps4, self._ps4)
        raw = self.upsample(x)                              # (B, K, H, W)
        anc = F.softplus(raw)                               # smooth ANC
        asc = anc / (anc.sum(dim=1, keepdim=True) + 1e-8)  # exact ASC
        return asc

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon:      (B, n_bands, H, W)  - reconstructed reflectance via LMM
            abundances: (B, n_minerals, H, W) - interpretable mineral fractions
            E:          (n_minerals, n_bands) - constrained endmember matrix
        """
        abundances = self.get_abundances(z)          # (B, K, H, W)
        E          = self.endmember_matrix            # (K, n_bands)  ∈ [0,1]

        # r = E^T · a  at each pixel (linear mixing - no sigmoid needed)
        B, K, H, W = abundances.shape
        a_flat     = abundances.permute(0, 2, 3, 1).reshape(-1, K)   # (B·H·W, K)
        recon_flat = torch.mm(a_flat, E)                               # (B·H·W, n_bands)
        recon      = recon_flat.view(B, H, W, self.n_bands).permute(0, 3, 1, 2)
        # recon is in [0,1]: linear combination of [0,1] spectra with weights in [0,1] summing to 1
        return recon, abundances, E


class AuxiliaryDecoder(nn.Module):
    """
    Auxiliary decoder heads for FeO prediction and DEM gradient reconstruction.
    Provides additional supervised signal during training.
    """
    def __init__(self, latent_dim: int = 128, patch_size: int = 64):
        super().__init__()
        ps4 = patch_size // 4

        self.feo_head = nn.Sequential(
            nn.Linear(latent_dim, 64 * ps4 * ps4),
            nn.Unflatten(1, (64, ps4, ps4)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.dem_head = nn.Sequential(
            nn.Linear(latent_dim, 64 * ps4 * ps4),
            nn.Unflatten(1, (64, ps4, ps4)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.feo_head(z), self.dem_head(z)
