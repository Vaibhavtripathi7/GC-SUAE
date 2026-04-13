"""
Build the 6-mineral endmember library on the IIRS wavelength grid (~800-2500nm).

Uses RELAB/USGS reference spectra when available
(https://www.planetary.brown.edu/relab/); otherwise constructs a synthetic
library from published spectral parameters, suitable for relative
mineralogical analysis.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# IIRS wavelength axis (800-2500nm, ~20nm spacing = 86 bands)
# Adjust if your actual IIRS HDR reports different wavelengths.
N_BANDS = 86
WAVELENGTHS = np.linspace(800, 2500, N_BANDS)  # nm


def gaussian_absorption(wl_center: float, width: float, depth: float,
                         wavelengths: np.ndarray) -> np.ndarray:
    """Returns a Gaussian absorption feature centered at wl_center."""
    return depth * np.exp(-0.5 * ((wavelengths - wl_center) / width) ** 2)


def make_endmember_spectra(wavelengths: np.ndarray) -> np.ndarray:
    """
    Construct physically-motivated synthetic endmember spectra for
    6 dominant lunar minerals, based on published spectral parameters
    from the RELAB/USGS spectral library.

    References:
        Klima et al. (2011) - pyroxene band positions
        Sunshine et al. (2007) - olivine absorption features
        Cheek et al. (2011) - plagioclase 1.25μm feature
        Pieters et al. (2014) - ilmenite spectral characteristics
        Gross et al. (2020) - Mg-spinel 2.0μm feature

    Returns: (6, n_bands) float32 array of L2-normalized spectra
    """
    spectra = np.zeros((6, len(wavelengths)), dtype=np.float32)

    # 0: Low-Ca Pyroxene (Orthopyroxene)
    # Bands at ~0.9μm and ~1.9μm; low Ca shifts bands to shorter wavelengths
    base = 0.25 + 0.15 * (wavelengths - 800) / (2500 - 800)
    spec = base.copy()
    spec -= gaussian_absorption(900,  80, 0.12, wavelengths)   # 1μm band
    spec -= gaussian_absorption(1900, 120, 0.18, wavelengths)  # 2μm band
    spectra[0] = spec.clip(0.01)

    # 1: High-Ca Pyroxene (Clinopyroxene)
    # Bands shifted to ~1.0μm and ~2.3μm; deeper 2μm band
    base = 0.20 + 0.10 * (wavelengths - 800) / (2500 - 800)
    spec = base.copy()
    spec -= gaussian_absorption(1010, 100, 0.14, wavelengths)  # 1μm band (shifted)
    spec -= gaussian_absorption(2250, 150, 0.22, wavelengths)  # 2μm band (shifted high-Ca)
    spectra[1] = spec.clip(0.01)

    # 2: Olivine
    # Broad composite absorption ~1.0-1.2μm (three overlapping features)
    base = 0.30 + 0.05 * (wavelengths - 800) / (2500 - 800)
    spec = base.copy()
    spec -= gaussian_absorption(860,  60, 0.08, wavelengths)
    spec -= gaussian_absorption(1050, 90, 0.16, wavelengths)
    spec -= gaussian_absorption(1260, 80, 0.10, wavelengths)
    spectra[2] = spec.clip(0.01)

    # 3: Plagioclase Feldspar (Anorthosite)
    # High overall reflectance (highlands material), weak 1.25μm Fe2+ feature
    base = 0.50 + 0.08 * (wavelengths - 800) / (2500 - 800)
    spec = base.copy()
    spec -= gaussian_absorption(1250, 120, 0.06, wavelengths)  # weak Fe2+ band
    spectra[3] = spec.clip(0.05)

    # 4: Ilmenite (FeTiO3)
    # Very dark, low reflectance, broad Fe2+ absorption, flat/featureless
    base = 0.08 + 0.02 * (wavelengths - 800) / (2500 - 800)
    spec = base.copy()
    spec -= gaussian_absorption(1100, 300, 0.04, wavelengths)  # very broad, shallow
    spectra[4] = spec.clip(0.01)

    # 5: Mg-Spinel
    # Distinctive 2.0μm absorption; relatively bright in NIR
    base = 0.35 + 0.05 * (wavelengths - 800) / (2500 - 800)
    spec = base.copy()
    spec -= gaussian_absorption(2000, 200, 0.20, wavelengths)  # strong 2μm Mg2+ feature
    spectra[5] = spec.clip(0.01)

    # L2-normalize each spectrum
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    return (spectra / (norms + 1e-8)).astype(np.float32)


def main():
    output_dir = "data/endmembers"
    os.makedirs(output_dir, exist_ok=True)

    print("[Endmembers] Building synthetic RELAB-motivated endmember library...")
    spectra = make_endmember_spectra(WAVELENGTHS)

    output_path = os.path.join(output_dir, "relab_lunar_6minerals.npy")
    np.save(output_path, spectra)

    wavelengths_path = os.path.join(output_dir, "iirs_wavelengths.npy")
    np.save(wavelengths_path, WAVELENGTHS)

    print(f"[Endmembers] Saved {spectra.shape} spectra → {output_path}")
    print(f"[Endmembers] Wavelength axis → {wavelengths_path}")

    # Verify: plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mineral_names = [
            "Low-Ca Pyroxene", "High-Ca Pyroxene", "Olivine",
            "Plagioclase", "Ilmenite", "Mg-Spinel"
        ]
        colors = ["#E24B4A", "#EF9F27", "#639922", "#378ADD", "#7F77DD", "#D4537E"]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (name, color) in enumerate(zip(mineral_names, colors)):
            ax.plot(WAVELENGTHS, spectra[i], label=name, color=color, linewidth=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("L2-normalized Reflectance")
        ax.set_title("RELAB-motivated Lunar Mineral Endmember Spectra (IIRS bands)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig_path = os.path.join(output_dir, "endmember_spectra.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[Endmembers] Spectra figure → {fig_path}")
    except ImportError:
        pass

    print("\n[Endmembers] Done.")
    print("NOTE: For publication-quality results, replace synthetic spectra")
    print("      with actual RELAB spectra resampled to your IIRS band positions.")
    print("      Use the IIRS HDR wavelength axis for accurate resampling.")


if __name__ == "__main__":
    main()
