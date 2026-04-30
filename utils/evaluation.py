"""
Evaluation: clustering metrics, reconstruction error, endmember-based mineral
identification, spatial mineral maps, and the result figures.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader

from data.dataset import EndmemberLibrary


MINERAL_COLORS = {
    "Low-Ca Pyroxene":  "#E24B4A",
    "High-Ca Pyroxene": "#EF9F27",
    "Olivine":          "#639922",
    "Plagioclase":      "#378ADD",
    "Ilmenite":         "#7F77DD",
    "Mg-Spinel":        "#D4537E",
    "Unidentified":     "#888780",
}


@torch.no_grad()
def extract_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract latent vectors, FeO means, and patch indices from a dataloader.

    Returns:
        latents:     (N, latent_dim) numpy array
        feo_means:   (N,) numpy array
        slope_means: (N,) numpy array
    """
    model.eval()
    latents, feo_means, slope_means = [], [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        if hasattr(model, "encode"):
            z, _ = model.encode(batch)
        else:
            _, z = model(batch)

        latents.append(z.cpu().numpy())
        feo_means.append(batch["feo_mean"].cpu().numpy())
        slope_means.append(batch["slope_mean"].cpu().numpy())

    return (
        np.vstack(latents),
        np.concatenate(feo_means),
        np.concatenate(slope_means),
    )


def cluster_latents(
    latents: np.ndarray,
    n_clusters: int = 10,
    n_init: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    K-Means++ clustering on latent space.

    Returns:
        labels:  (N,) cluster assignments
        metrics: dict of clustering quality metrics
    """
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed, init="k-means++")
    labels = km.fit_predict(latents)

    sil = silhouette_score(latents, labels)
    db  = davies_bouldin_score(latents, labels)

    metrics = {
        "silhouette_score":    round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "n_clusters":          n_clusters,
    }
    return labels, km.cluster_centers_, metrics


def compute_cluster_spectra(
    model: torch.nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    n_clusters: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute mean spectral signature for each cluster by averaging
    IIRS patches assigned to each cluster.

    Returns: (n_clusters, n_bands) mean spectra
    """
    from models.architectures import SpectralAngleMapperLoss

    model.eval()
    cluster_sums   = np.zeros((n_clusters, loader.dataset.n_bands))
    cluster_counts = np.zeros(n_clusters)
    idx = 0

    with torch.no_grad():
        for batch in loader:
            B = batch["iirs"].shape[0]
            iirs_np = batch["iirs"].numpy()  # (B, n_bands, H, W)
            for b in range(B):
                if idx < len(labels):
                    # Mean spectrum of this patch
                    patch_mean = iirs_np[b].mean(axis=(1, 2))  # (n_bands,)
                    cluster_sums[labels[idx]]   += patch_mean
                    cluster_counts[labels[idx]] += 1
                    idx += 1

    # Avoid division by zero
    counts = cluster_counts[:, None]
    counts[counts == 0] = 1
    return cluster_sums / counts


def compute_sam_reconstruction(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute pixel-wise SAM error on the validation set."""
    from models.architectures import SpectralAngleMapperLoss
    model.eval()
    sam_fn = SpectralAngleMapperLoss().to(device)

    all_sams, all_mses = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if hasattr(model, "encode"):
                output = model(batch)
                recon = output["recon_iirs"]
            else:
                recon, _ = model(batch)

            iirs = batch["iirs"]
            all_sams.append(sam_fn(recon, iirs).item())
            all_mses.append(torch.nn.functional.mse_loss(recon, iirs).item())

    return {
        "mean_sam_rad": round(float(np.mean(all_sams)), 5),
        "mean_mse":     round(float(np.mean(all_mses)), 6),
        "mean_sam_deg": round(float(np.degrees(np.mean(all_sams))), 3),
    }


def build_mineral_map(
    labels: np.ndarray,
    mineral_assignments: List[Optional[int]],
    spatial_grid: Tuple[int, int],
    mineral_names: List[str],
) -> np.ndarray:
    """
    Build a 2D spatial mineral map from flat cluster labels.

    Returns: (n_rows, n_cols) array of mineral indices (-1 = unidentified)
    """
    n_rows, n_cols = spatial_grid
    n_valid = n_rows * n_cols

    mineral_labels = np.full(len(labels), -1, dtype=int)
    for cluster_id, mineral_id in enumerate(mineral_assignments):
        if mineral_id is not None:
            mineral_labels[labels == cluster_id] = mineral_id

    mineral_map = mineral_labels[:n_valid].reshape(n_rows, n_cols)
    return mineral_map
