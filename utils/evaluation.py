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


# Publication-quality figure generation

def figure1_main_results(
    results: Dict[str, dict],
    cluster_spectra: np.ndarray,
    feo_array: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    output_path: str,
    dpi: int = 300,
):
    """
    Figure 1: Main quantitative results (4-panel).
    Panel A: Ablation bar chart (Silhouette Score)
    Panel B: FeO abundance per mineral cluster (box plots)
    Panel C: t-SNE latent space projection
    Panel D: Mean spectral signatures per cluster
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Ablation comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    sil_scores  = [results[m]["silhouette_score"] for m in model_names]
    colors = ["#aaaaaa"] * (len(model_names) - 1) + ["#2ecc71"]
    bars = ax.bar(model_names, sil_scores, color=colors, edgecolor="black", width=0.55)
    ax.set_title("A. Silhouette Score - Model Ablation", fontweight="bold")
    ax.set_ylabel("Silhouette Score (higher = better)")
    ax.set_ylim(0, max(sil_scores) + 0.08)
    for bar, val in zip(bars, sil_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    # Panel B: FeO per cluster
    ax = axes[0, 1]
    sns.boxplot(x=labels, y=feo_array, ax=ax, palette="tab10")
    ax.set_title("B. FeO Abundance by Mineral Cluster", fontweight="bold")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("FeO (normalized)")

    # Panel C: t-SNE projection
    ax = axes[1, 0]
    # Use PCA to 50D first, then t-SNE for stability
    try:
        from sklearn.decomposition import PCA
        pca_50 = PCA(n_components=min(50, cluster_spectra.shape[0] - 1))
        # Note: latent matrix not available here; placeholder
        ax.set_title("C. t-SNE Latent Space Projection", fontweight="bold")
        ax.text(0.5, 0.5, "Run with full latent matrix\n(see evaluate.py)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
    except Exception:
        pass

    # Panel D: Mean spectral signatures
    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(min(n_clusters, cluster_spectra.shape[0])):
        ax.plot(cluster_spectra[i], label=f"Cluster {i}", color=colors[i], linewidth=1.8)
    ax.set_title("D. Mean Spectral Signature per Cluster", fontweight="bold")
    ax.set_xlabel("IIRS Band Index (800-2500 nm)")
    ax.set_ylabel("Normalized Reflectance")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved Figure 1 → {output_path}")


def figure2_mineral_map(
    mineral_map: np.ndarray,
    dem_data: np.ndarray,
    feo_data: np.ndarray,
    mineral_assignments: List[Optional[int]],
    mineral_names: List[str],
    output_path: str,
    dpi: int = 300,
):
    """
    Figure 2: Spatial mineral map (3-panel).
    Panel A: DEM topography
    Panel B: Clementine FeO geochemistry
    Panel C: GC-SUAE mineral map overlaid on DEM
    """
    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    # Panel A: DEM
    axes[0].imshow(dem_data, cmap="terrain", interpolation="bilinear")
    axes[0].set_title("A. TMC-2 DEM (Topography)", fontweight="bold", fontsize=13)
    axes[0].axis("off")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="terrain"), ax=axes[0], shrink=0.6, pad=0.02
    )
    cbar.set_label("Relative Elevation", fontweight="bold")

    # Panel B: FeO map
    axes[1].imshow(feo_data, cmap="inferno", interpolation="bilinear")
    axes[1].set_title("B. Clementine FeO Abundance", fontweight="bold", fontsize=13)
    axes[1].axis("off")
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(cmap="inferno"), ax=axes[1], shrink=0.6, pad=0.02
    )
    cbar2.set_label("FeO Wt % (normalized)", fontweight="bold")

    # Panel C: Mineral map
    n_minerals = len(mineral_names) + 1  # +1 for unidentified
    cmap_colors = [MINERAL_COLORS.get(m, "#888780") for m in mineral_names]
    cmap_colors.append(MINERAL_COLORS["Unidentified"])
    mineral_cmap = mcolors.ListedColormap(cmap_colors)

    # Map mineral_map values (-1 = unidentified → last color)
    display_map = mineral_map.copy()
    display_map[display_map == -1] = len(mineral_names)

    axes[2].imshow(dem_data, cmap="gray", alpha=0.4, interpolation="bilinear")
    im = axes[2].imshow(
        display_map, cmap=mineral_cmap,
        alpha=0.65, interpolation="nearest",
        vmin=0, vmax=n_minerals - 1,
    )
    axes[2].set_title("C. GC-SUAE Mineral Map (overlaid on topography)",
                       fontweight="bold", fontsize=13)
    axes[2].axis("off")

    # Custom legend
    legend_elements = [
        matplotlib.patches.Patch(facecolor=MINERAL_COLORS.get(m, "#888780"), label=m)
        for m in list(mineral_names) + ["Unidentified"]
    ]
    axes[2].legend(handles=legend_elements, loc="lower right",
                   fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved Figure 2 → {output_path}")


def figure3_ablation_table_figure(
    results: Dict[str, dict],
    output_path: str,
    dpi: int = 300,
):
    """
    Figure 3: Comprehensive ablation heatmap table.
    Shows all metrics across all model variants as a styled heatmap.
    """
    import pandas as pd

    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model":              model_name,
            "Silhouette ↑":       metrics.get("silhouette_score", 0),
            "Davies-Bouldin ↓":   metrics.get("davies_bouldin_index", 0),
            "Mean SAM (°) ↓":     metrics.get("mean_sam_deg", 0),
            "Mineral ID Acc. ↑":  metrics.get("mineral_id_accuracy", 0),
        })

    df = pd.DataFrame(rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Render as styled table
    table = ax.table(
        cellText=df.round(4).values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.0)

    # Highlight proposed model (last row)
    for j in range(len(df.columns)):
        cell = table[len(df), j + 1]
        cell.set_facecolor("#d4f1d4")
        cell.set_text_props(fontweight="bold")

    ax.set_title("Table 1. Comprehensive Ablation Results", fontweight="bold",
                 fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved Figure 3 → {output_path}")


def print_results_table(results: Dict[str, dict]):
    """Pretty-print results to stdout."""
    header = f"{'Model':<22} {'Silhouette ↑':>14} {'DB Index ↓':>12} {'SAM (°) ↓':>10} {'MinID Acc ↑':>12}"
    print("\n" + "=" * 74)
    print(header)
    print("-" * 74)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m.get('silhouette_score', 0):>14.4f} "
            f"{m.get('davies_bouldin_index', 0):>12.4f} "
            f"{m.get('mean_sam_deg', 0):>10.3f} "
            f"{m.get('mineral_id_accuracy', 0)*100:>11.1f}%"
        )
    print("=" * 74 + "\n")
