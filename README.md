# LunarSpecNet: Physics-Informed Multimodal Spectral Unmixing for Chandrayaan-2 IIRS Data

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

> **LunarSpecNet** is a physics-informed multimodal deep learning framework for unsupervised lunar mineral mapping using Chandrayaan-2 IIRS hyperspectral reflectance data fused with TMC-2 topographic and Clementine geochemical modalities. It introduces a **Geologically-Constrained Spectral Unmixing Autoencoder (GC-SUAE)** that enforces mineralogical priors directly in the latent space via a novel terrain-aware contrastive loss, producing interpretable mineral abundance maps anchored to RELAB/USGS endmember spectra.

---

## Key Contributions

1. **GC-SUAE Architecture** - A spatially-aware multimodal encoder with deformable cross-attention between IIRS spatial feature maps and DEM/FeO modalities (operating on H×W feature tensors, not pooled vectors), preserving spatial mineralogical context through the bottleneck.

2. **Terrain-Aware Geochemical Contrastive Loss (TAGCL)** - A domain-specific contrastive objective where positive/negative pairs are constructed from joint geochemical-topographic proximity (FeO abundance + slope similarity), not augmentation. This regularizes the latent space to reflect physical mineralogical relationships - not learned by any prior lunar hyperspectral work.

3. **Physics-Constrained Decoder with Hapke Linearization** - The decoder enforces non-negativity and sum-to-one constraints on abundance fractions (Linear Mixing Model), making outputs physically interpretable as fractional mineral abundances rather than opaque cluster IDs.

4. **Endmember-Anchored Mineral Identification** - Clusters are mapped to six RELAB endmember spectra (low-Ca pyroxene, high-Ca pyroxene, olivine, plagioclase, ilmenite, Mg-spinel) via SAM matching, producing a geologically labeled mineral map - not a segmentation map.

5. **Comprehensive Ablation** - 5-model ablation (Unimodal 2D-CNN, Unimodal 3D-CNN, Early-Fusion, Late-Fusion, TRIAD-base) plus the proposed GC-SUAE, with Silhouette Score, Davies-Bouldin Index, SAM reconstruction error, and endmember identification accuracy.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        GC-SUAE                              │
│                                                             │
│  IIRS (B×256×H×W) ──► Spectral-Spatial Encoder             │
│                              │                              │
│                         F_iirs (B×C×h×w)                   │
│                              │                              │
│  DEM  (B×1×H×W)  ──► Terrain Encoder ──► Sobel Gradient   │
│                              │                              │
│                         F_dem  (B×C×h×w)                   │
│                              │                              │
│  FeO  (B×1×H×W)  ──► Geochemical Encoder                  │
│                              │                              │
│                         F_feo  (B×C×h×w)                   │
│                                                             │
│  ┌──────── Deformable Cross-Attention Fusion ────────────┐  │
│  │  IIRS queries DEM+FeO spatial features                │  │
│  │  Output: fused spatial feature map F_fused            │  │
│  └────────────────────────────────────────────────────── ┘  │
│                              │                              │
│                    Latent Bottleneck Z                      │
│                              │                              │
│  ┌──────────── Multi-Head Decoder ───────────────────────┐  │
│  │  Head 1: IIRS Reconstruction (MSE + SAM)              │  │
│  │  Head 2: FeO Abundance Prediction (semi-supervised)   │  │
│  │  Head 3: DEM Gradient Reconstruction                  │  │
│  └────────────────────────────────────────────────────── ┘  │
│                                                             │
│  Loss: L_mse + λ_sam·L_sam + λ_geo·L_tagcl + λ_dem·L_dem  │
│         + λ_feo·L_feo + λ_lmm·L_lmm                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Modality | Source | Resolution | Description |
|---|---|---|---|
| IIRS Hyperspectral | Chandrayaan-2 IIRS Level-2 | ~80m, 256 bands (800-5000nm) | Calibrated reflectance |
| DEM | Chandrayaan-2 TMC-2 DTM | ~32m | Digital terrain model |
| FeO Geochemistry | Clementine UVVIS | ~1km | Iron oxide abundance proxy |
| Endmembers | RELAB/USGS Spectral Library | - | 6 mineral reference spectra |

---

## Installation

```bash
git clone https://github.com/Vaibhavtripathi7/LunarSpecNet.git
cd LunarSpecNet
pip install -e ".[dev]"
```

---

## Usage

### Training

```bash
python scripts/train.py --config configs/gcsuae_default.yaml
```

### Evaluation + Mineral Mapping

```bash
python scripts/evaluate.py --checkpoint outputs/gcsuae_best.pth --output_dir results/
```

### Full Ablation Study

```bash
python scripts/run_ablation.py --config configs/ablation.yaml
```

---

## Results

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Mean SAM ↓ | Mineral ID Acc. ↑ |
|---|---|---|---|---|
| Unimodal 2D-CNN | - | - | - | - |
| Unimodal 3D-CNN | - | - | - | - |
| Early-Fusion | - | - | - | - |
| Late-Fusion | - | - | - | - |
| TRIAD (baseline) | - | - | - | - |
| **GC-SUAE (ours)** | **-** | **-** | **-** | **-** |

*Fill in after running experiments.*

---

## Citation

```bibtex
@article{tripathi2026lunarspecnet,
  title={LunarSpecNet: Physics-Informed Multimodal Spectral Unmixing for Chandrayaan-2 IIRS Hyperspectral Data},
  author={Tripathi, Vaibhav},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
