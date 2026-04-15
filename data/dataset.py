"""
Multimodal lunar patch dataset: Chandrayaan-2 IIRS reflectance with TMC-2 DEM
(plus Sobel slope/aspect) and Clementine FeO, using per-modality percentile
normalization and strided patch extraction.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import spectral.io.envi as envi
import torch
from scipy.ndimage import sobel
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class LunarMultimodalDataset(Dataset):
    """
    Multimodal patch dataset for Chandrayaan-2 IIRS + TMC-2 DEM + Clementine FeO.

    Each sample is a dict:
        {
            'iirs':   (n_bands, patch_size, patch_size)  float32 reflectance
            'dem':    (1, patch_size, patch_size)         float32 normalized elevation
            'feo':    (1, patch_size, patch_size)         float32 normalized FeO wt%
            'slope':  (1, patch_size, patch_size)         float32 terrain slope magnitude
            'aspect': (1, patch_size, patch_size)         float32 terrain aspect
            'feo_mean':  scalar float  - patch-mean FeO for TAGCL pair construction
            'slope_mean': scalar float - patch-mean slope for TAGCL
            'patch_idx': int           - unique patch index for pairing
        }
    """

    def __init__(
        self,
        iirs_hdr: str,
        iirs_qub: str,
        dem_path: str,
        feo_path: str,
        band_start: int = 0,
        band_end: int = 86,
        patch_size: int = 64,
        stride: int = 32,
        normalize: bool = True,
        subsample_frac: float = 0.1,  # fraction used for global stats
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.band_start = band_start
        self.band_end = band_end
        self.n_bands = band_end - band_start

        # Load IIRS (memory-mapped for large files)
        print("[Dataset] Opening IIRS memory map...")
        self._iirs_lib = envi.open(iirs_hdr, image=iirs_qub)
        self._iirs_mmap = self._iirs_lib.open_memmap(writable=False)
        self.H, self.W, self.B = self._iirs_mmap.shape
        print(f"[Dataset] IIRS shape: {self.H}×{self.W}×{self.B} bands")

        # Load DEM
        with rasterio.open(dem_path) as src:
            self.dem_data = src.read(1).astype(np.float32)
        print(f"[Dataset] DEM shape: {self.dem_data.shape}")

        # Compute terrain slope and aspect from DEM
        # Sobel gradient as fixed physics-based feature (not learned)
        sobel_x = sobel(self.dem_data, axis=1)
        sobel_y = sobel(self.dem_data, axis=0)
        self.slope_data = np.hypot(sobel_x, sobel_y).astype(np.float32)
        self.aspect_data = np.arctan2(sobel_y, sobel_x).astype(np.float32)

        # Load FeO geochemical map
        with rasterio.open(feo_path) as src:
            self.feo_data = src.read(1).astype(np.float32)
        print(f"[Dataset] FeO shape: {self.feo_data.shape}")

        # Compute valid patch indices
        self.indices: List[Tuple[int, int]] = []
        for r in range(0, self.H - patch_size, stride):
            for c in range(0, self.W - patch_size, stride):
                self.indices.append((r, c))
        print(f"[Dataset] Total patches: {len(self.indices)}")

        # Global normalization stats (computed on subsample)
        if normalize:
            self._compute_normalization_stats(subsample_frac)
        else:
            # Identity transform
            self.iirs_min, self.iirs_max = 0.0, 1.0
            self.dem_min, self.dem_max = 0.0, 1.0
            self.feo_min, self.feo_max = 0.0, 1.0
            self.slope_min, self.slope_max = 0.0, 1.0

    def _compute_normalization_stats(self, frac: float):
        """
        Compute global normalization statistics using percentile clipping
        (2nd-98th percentile) to avoid outlier sensitivity.
        """
        print("[Dataset] Computing normalization statistics (this may take a moment)...")

        step = max(1, int(1.0 / frac))
        iirs_sub = np.nan_to_num(
            self._iirs_mmap[::step, ::step, self.band_start:self.band_end]
        ).astype(np.float32)
        self.iirs_min = float(np.percentile(iirs_sub, 2))
        self.iirs_max = float(np.percentile(iirs_sub, 98))

        dem_valid = np.nan_to_num(self.dem_data)
        self.dem_min = float(np.percentile(dem_valid, 2))
        self.dem_max = float(np.percentile(dem_valid, 98))

        feo_valid = np.nan_to_num(self.feo_data)
        self.feo_min = float(np.percentile(feo_valid, 2))
        self.feo_max = float(np.percentile(feo_valid, 98))

        slope_valid = np.nan_to_num(self.slope_data)
        self.slope_min = float(np.percentile(slope_valid, 2))
        self.slope_max = float(np.percentile(slope_valid, 98))

        print(
            f"[Dataset] IIRS: [{self.iirs_min:.4f}, {self.iirs_max:.4f}] | "
            f"DEM: [{self.dem_min:.2f}, {self.dem_max:.2f}] | "
            f"FeO: [{self.feo_min:.4f}, {self.feo_max:.4f}]"
        )

    def _normalize(self, x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax > vmin:
            return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        return np.zeros_like(x)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r, c = self.indices[idx]
        ps = self.patch_size

        # IIRS patch
        iirs_raw = np.array(
            self._iirs_mmap[r:r+ps, c:c+ps, self.band_start:self.band_end],
            dtype=np.float32
        )
        iirs_raw = np.nan_to_num(iirs_raw).transpose(2, 0, 1)  # (B, H, W)
        iirs = self._normalize(iirs_raw, self.iirs_min, self.iirs_max)

        # DEM patch
        dem_raw = np.nan_to_num(self.dem_data[r:r+ps, c:c+ps])
        dem = self._normalize(dem_raw, self.dem_min, self.dem_max)[None]  # (1,H,W)

        # FeO patch
        feo_raw = np.nan_to_num(self.feo_data[r:r+ps, c:c+ps])
        feo = self._normalize(feo_raw, self.feo_min, self.feo_max)[None]  # (1,H,W)

        # Slope + Aspect patches
        slope_raw = np.nan_to_num(self.slope_data[r:r+ps, c:c+ps])
        slope = self._normalize(slope_raw, self.slope_min, self.slope_max)[None]

        aspect_raw = np.nan_to_num(self.aspect_data[r:r+ps, c:c+ps])
        # aspect is in [-pi, pi]; normalize to [0, 1]
        aspect = ((aspect_raw + np.pi) / (2 * np.pi)).astype(np.float32)[None]

        # Scalar stats for TAGCL pair construction
        # Use raw (physical) units for proximity thresholds
        feo_mean = float(np.nanmean(feo_raw))
        slope_mean = float(np.nanmean(slope_raw))

        return {
            "iirs":       torch.from_numpy(iirs),
            "dem":        torch.from_numpy(dem),
            "feo":        torch.from_numpy(feo.astype(np.float32)),
            "slope":      torch.from_numpy(slope.astype(np.float32)),
            "aspect":     torch.from_numpy(aspect),
            "feo_mean":   torch.tensor(feo_mean, dtype=torch.float32),
            "slope_mean": torch.tensor(slope_mean, dtype=torch.float32),
            "patch_idx":  torch.tensor(idx, dtype=torch.long),
        }

    @property
    def spatial_grid(self) -> Tuple[int, int]:
        """Returns (n_rows, n_cols) of the patch grid for spatial map assembly."""
        n_rows = (self.H - self.patch_size) // self.stride + 1
        n_cols = (self.W - self.patch_size) // self.stride + 1
        return n_rows, n_cols


class EndmemberLibrary:
    """
    Loads and manages the RELAB/USGS reference endmember spectra for
    6 dominant lunar minerals, resampled to IIRS band positions.

    Minerals:
        0 - low-Ca pyroxene (orthopyroxene)
        1 - high-Ca pyroxene (clinopyroxene)
        2 - olivine
        3 - plagioclase feldspar
        4 - ilmenite
        5 - Mg-spinel
    """

    MINERAL_NAMES = [
        "Low-Ca Pyroxene",
        "High-Ca Pyroxene",
        "Olivine",
        "Plagioclase",
        "Ilmenite",
        "Mg-Spinel",
    ]

    def __init__(self, endmember_path: str, n_bands: int = 86):
        """
        Args:
            endmember_path: Path to .npy file of shape (6, n_bands_original)
                            containing RELAB spectra resampled to IIRS wavelengths.
            n_bands: Number of IIRS bands used (should match dataset).
        """
        if not os.path.exists(endmember_path):
            raise FileNotFoundError(
                f"Endmember library not found at {endmember_path}. "
                "See scripts/prepare_endmembers.py to generate from RELAB data."
            )
        data = np.load(endmember_path)  # (6, n_bands)
        assert data.shape[0] == 6, f"Expected 6 endmembers, got {data.shape[0]}"

        # L2-normalize each endmember spectrum
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        self.spectra = (data / (norms + 1e-8)).astype(np.float32)  # (6, n_bands)
        self.n_minerals = 6

    def as_tensor(self) -> torch.Tensor:
        """Returns endmember matrix as tensor (6, n_bands)."""
        return torch.from_numpy(self.spectra)

    def identify(
        self, cluster_spectra: np.ndarray, sam_threshold: float = 0.15
    ) -> List[Optional[int]]:
        """
        Identifies each cluster's dominant mineral via SAM distance to endmembers.

        Args:
            cluster_spectra: (n_clusters, n_bands) mean spectra per cluster.
            sam_threshold: SAM distance (radians) cutoff. Above this = unidentified.

        Returns:
            List of mineral indices (or None if unidentified) for each cluster.
        """
        n_clusters = cluster_spectra.shape[0]
        assignments = []
        for i in range(n_clusters):
            spec = cluster_spectra[i]
            spec_norm = spec / (np.linalg.norm(spec) + 1e-8)
            sams = []
            for em in self.spectra:
                cos = np.clip(np.dot(spec_norm, em), -1.0, 1.0)
                sams.append(np.arccos(cos))
            best_idx = int(np.argmin(sams))
            assignments.append(best_idx if sams[best_idx] < sam_threshold else None)
        return assignments
