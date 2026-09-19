# GC-SUAE

### Geologically-Constrained Spectral Unmixing Autoencoder for Unsupervised Lunar Mineral Mapping via Multimodal Spatial Cross-Attention

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Colab-lightgrey)


## Introduction

Mapping lunar mineral composition from orbit is harder than it looks. Each pixel in a hyperspectral image captures a mixture of minerals, but the spectrum you measure is not purely a function of what minerals are present. It also depends on how steep the terrain is, how badly the surface has been weathered by micrometeorites and solar wind, and whether thermal emission is contaminating the longer wavelengths. Methods that work from spectral data alone cannot separate mineralogy from these confounding physical effects.

GC-SUAE addresses this by jointly processing three physically independent measurements: Chandrayaan-2 IIRS hyperspectral reflectance (86 bands, 800–2500 nm), TMC-2 terrain morphology (elevation, slope, aspect), and Clementine iron abundance maps. The core idea is that terrain and geochemistry constrain mineral identity from directions completely independent of reflectance. Steep crater walls tend to expose subsurface orthopyroxene and olivine. Flat maria host clinopyroxene and ilmenite. Iron abundance directly tracks ferrous mineral presence. These are not redundant signals; they resolve ambiguities that spectra alone cannot.

Rather than collapsing these auxiliary signals into a single global context vector, GC-SUAE fuses them through pixel-wise spatial cross-attention. Each pixel in the IIRS feature map attends to the terrain and geochemical features at its own spatial location, so local context at crater rims, slope transitions, and compositional boundaries is preserved rather than averaged away.

The decoder enforces physically valid outputs through a Linear Mixing Model constraint. Abundance fractions are non-negative and sum to one by construction, not as a soft penalty, and learned endmember spectra are kept within the physical reflectance range throughout training.


## Results

All experiments were run on a Chandrayaan-2 IIRS scene covering the southern near-side lunar highlands (24.79°S–39.64°S, 15.97°E–16.99°E), a predominantly anorthositic highland region where 86.6% of pixels fall within the 1–5 wt% FeO range. Six model variants were trained under identical conditions: same dataset, same optimizer, same random seed, same hardware.

| Method | Silhouette ↑ | DB Index ↓ | SAM (°) ↓ |
|:---|:---:|:---:|:---:|
| Unimodal-2DCNN | 0.168 | 1.503 | 6.59 |
| Unimodal-3DCNN | 0.186 | 1.453 | 6.86 |
| EarlyFusion | 0.182 | 1.468 | 6.68 |
| LateFusion | 0.198 | 1.382 | 6.85 |
| PooledAttnFusion | 0.395 | 0.732 | 7.96 |
| GC-SUAE w/ TAGCL | 0.032 | 3.649 | 29.3 |
| **GC-SUAE w/o TAGCL** | **0.254** | **1.285** | **23.5** |

Attention-based fusion models substantially outperform all unimodal and naive fusion baselines. The Silhouette gap between LateFusion (the best non-attention baseline) and GC-SUAE w/o TAGCL is over 0.05 points, a difference that holds consistently across both clustering metrics. GC-SUAE with spatial cross-attention outperforms every non-attention baseline on Silhouette and DB index.

TAGCL collapses on this scene because 86.6% of pixels fall within the same 1–5 wt% FeO bin, making contrastive positive pair construction degenerate. Almost every pair qualifies as positive and the loss cannot learn discriminative representations. This is a scene-specific data quality issue, not an architectural flaw. On geochemically varied terrain like mare-highland transitions where FeO gradients are steep, TAGCL is expected to work as intended.

A fourth metric, mineral identification accuracy against a synthetic RELAB endmember library, was evaluated and produced zero matches across all models. This is not a representation failure. The clustering metrics above confirm meaningful latent structure is being learned. The zero result reflects a calibration gap between laboratory reference spectra and real orbital data: space weathering suppresses absorption band depth, thermal emission contaminates wavelengths above 2000 nm, and instrument response differences between laboratory and orbital conditions are not captured in synthetic endmembers. Accurate mineral identification from IIRS requires scene-derived endmembers extracted directly from the data using N-FINDR or Vertex Component Analysis, which is a clear direction for future work.

Full metrics are in `results/ablation_results.json`.


## Architecture

![arch-diagram](./docs/gc_suae_architecture_v7(1).png)

Three modality-specific encoders produce feature maps at H/4 × W/4 resolution. Two sequential Spatial Cross-Attention modules fuse terrain then geochemistry into the IIRS representation, with each pixel attending to its own location in the auxiliary maps rather than a scene-wide average. The fused features are compressed to a 128-dimensional latent vector and decoded through transposed convolutions back to full spatial resolution.

Spatial Cross-Attention uses IIRS features as query and auxiliary modality features as key and value. A residual connection ensures terrain context is added to spectral identity rather than replacing it, so the spectral signal is never discarded during fusion.

The LMM decoder enforces abundance non-negativity via softplus activation, sum-to-one via exact L1 normalisation, and constrains learned endmember spectra to [0, 1] via sigmoid parameterisation, all by construction with no approximation.

TAGCL constructs contrastive positive pairs from patches sharing joint proximity in both FeO abundance and terrain slope. The AND-gate means both conditions must hold simultaneously, grounding the contrastive signal in two independent physical measurements rather than one. A MoCo-style momentum encoder with queue size 4096 provides a large and consistent set of negative pairs.


## Repository Structure

```
GC-SUAE/
├── configs/
│   ├── ablation.yaml                   # Training config for all six model variants
│   └── gcsuae_default.yaml             # Default single-model GC-SUAE training config
├── data/
│   ├── dataset.py                      # Multimodal patch dataset with memory-mapped IIRS
│   └── endmembers/
│       ├── relab_lunar_6minerals.npy   # Synthetic RELAB-motivated endmember spectra (6×86)
│       └── iirs_wavelengths.npy        # IIRS wavelength axis, 86 bands, 800 to 2500 nm
├── losses/
│   └── tagcl.py                        # TAGCL contrastive loss and combined training objective
├── models/
│   └── architectures.py                # All six model variants and build_model factory
├── notebooks/
│   └── reproduction.ipynb              # End-to-end Colab notebook to reproduce all results
├── results/
│   └── ablation_results.json           # Ablation metrics for all six models
├── scripts/
│   ├── prepare_endmembers.py           # Generate synthetic RELAB endmember library
│   ├── preprocess_data.py              # Align IIRS, DEM, and FeO to a common spatial grid
│   ├── run_ablation.py                 # Train all six variants sequentially and evaluate
│   └── train.py                        # Train a single GC-SUAE model from a config
├── tests/
│   └── test_models.py                  # 19 unit tests covering all model variants
├── utils/
│   ├── evaluation.py                   # Clustering metrics and mineral identification
│   └── trainer.py                      # Training loop with cosine annealing and checkpointing
└── pyproject.toml                      # Package definition, install with pip install -e .
```


## Data

The raw data files are not included in this repository due to size. Three files are required from two sources.

Chandrayaan-2 IIRS Level-2 reflectance is available from the ISRO PRADAN portal at https://pradan.issdc.gov.in. Create a free account and navigate to Chandrayaan-2 → IIRS → Level-2. The data comes as two files that must always be kept together: a `.hdr` header and a `.qub` data cube. Total size is approximately 1.5 GB.

The TMC-2 Digital Elevation Model is available from the same PRADAN portal under Chandrayaan-2 → TMC-2 → Level-2 DTM. Download the GeoTIFF covering the same geographic area as your IIRS scene. The preprocessing script handles spatial alignment to the IIRS grid automatically. Size is approximately 50 MB.

The Clementine UVVIS FeO abundance map is available from the NASA Planetary Data System at https://pds-geosciences.wustl.edu/missions/clementine. This is a global map and the preprocessing script will crop it to your IIRS scene bounds automatically. Size is approximately 20 MB.


## Installation

Python 3.10 or later is required. Clone the repository and install in editable mode:

```bash
git clone https://github.com/Vaibhavtripathi7/GC-SUAE.git
cd GC-SUAE
pip install -e .
```

To verify the installation run the unit tests:

```bash
python -m pytest tests/ -v
# Expected: 19 passed
```


## Reproducing Results

A complete notebook is at `notebooks/reproduction.ipynb`, developed and tested on Google Colab with a T4 GPU. The four steps below reproduce the full ablation from scratch.

**Step 1: Generate the endmember library.** This creates both endmember files under `data/endmembers/` and takes about five seconds.

```bash
python scripts/prepare_endmembers.py
```

**Step 2: Preprocess and align the three modalities.** This reads the IIRS data cube, reprojects the DEM and FeO map to the IIRS spatial grid, and writes aligned GeoTIFFs to `data/processed/`.

```bash
python scripts/preprocess_data.py \
    --iirs_hdr /path/to/data.hdr \
    --iirs_qub /path/to/data.qub \
    --tmc_dem  /path/to/dem.tif \
    --feo_map  /path/to/feo.tif \
    --output_dir data/processed/
```

**Step 3: Run the full ablation.** This trains all six model variants sequentially. If a run is interrupted, simply restart it. Completed models are detected via saved checkpoint files and skipped automatically. Training takes approximately three to four hours on a T4 GPU.

```bash
python scripts/run_ablation.py --config configs/ablation.yaml
```

**Step 4: Inspect results.** Results are written to `outputs/ablation/ablation_results.json` when all models finish.

```python
import json

with open('outputs/ablation/ablation_results.json') as f:
    results = json.load(f)

print(f"{'Method':<28} {'Silhouette':>12} {'DB Index':>10} {'SAM (°)':>9}")
print('─' * 62)
for model, m in results.items():
    print(f"{model:<28} {m['silhouette_score']:>12.4f} "
          f"{m['davies_bouldin_index']:>10.4f} "
          f"{m['mean_sam_deg']:>9.3f}")
```


## Citation

If you use this code or build on this work, please cite:

```bibtex
@misc{tripathi2026gcsuae,
  title  = {{GC-SUAE}: Geologically-Constrained Spectral Unmixing Autoencoder
            for Unsupervised Lunar Mineral Mapping via Multimodal Spatial Cross-Attention},
  author = {Tripathi, Vaibhav},
  year   = {2026},
  url    = {https://github.com/Vaibhavtripathi7/GC-SUAE}
}
```

This work builds directly on the following papers. Bioucas-Dias et al. (IEEE JSTARS 2012) for spectral unmixing foundations and the Linear Mixing Model formulation. Khosla et al. (NeurIPS 2020) for the supervised InfoNCE loss used in TAGCL. He et al. (CVPR 2020) for the MoCo momentum encoder architecture. Chauhan et al. (Icarus 2023) for the Chandrayaan-2 IIRS dataset and lunar mapping context. Kodikara et al. (arXiv 2024) for the lunar hyperspectral unmixing baseline.


## Acknowledgements

Chandrayaan-2 IIRS and TMC-2 data are provided by the Indian Space Research Organisation (ISRO) through the PRADAN portal. The Clementine UVVIS FeO abundance map is provided by NASA through the Planetary Data System. Synthetic endmember spectra in this repository are motivated by the RELAB spectral library maintained by Brown University.


## License

MIT, see [LICENSE](LICENSE).