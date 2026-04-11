"""
Align TMC-2 DEM and Clementine FeO maps to the IIRS spatial grid. Run once
before training.

Usage:
    python scripts/preprocess_data.py --iirs_hdr data/raw/data.hdr \
        --iirs_qub data/raw/data.qub --tmc_dem data/raw/ch2_tmc_*.tif \
        --feo_map data/raw/Lunar_Clementine_*.tif --output_dir data/processed/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import spectral.io.envi as envi


# Lunar sphere radius (IAU 2015)
LUNAR_RADIUS_M = 1_737_400
LUNAR_CRS = f"+proj=longlat +R={LUNAR_RADIUS_M} +no_defs"


def get_iirs_metadata(hdr_path: str, qub_path: str) -> dict:
    """Extract spatial metadata from IIRS header."""
    img = envi.open(hdr_path, image=qub_path)
    metadata = img.metadata

    n_rows = img.nrows
    n_cols = img.ncols
    n_bands = img.nbands

    # Parse map info from ENVI header if present
    map_info = metadata.get("map info", None)
    if map_info is not None and isinstance(map_info, list):
        # ENVI map info format:
        # [projection, ref_x, ref_y, x_start, y_start, x_pixel_size, y_pixel_size, ...]
        try:
            x_start    = float(map_info[3])
            y_start    = float(map_info[4])
            x_pix_size = float(map_info[5])
            y_pix_size = float(map_info[6])
            west  = x_start
            north = y_start
            east  = west  + n_cols * x_pix_size
            south = north - n_rows * y_pix_size
        except (IndexError, ValueError):
            # Fallback: use hardcoded bounds from original code
            print("[WARN] Could not parse map info; using default IIRS bounds.")
            north, south, east, west = -24.790135, -39.641186, 16.993043, 15.965546
    else:
        north, south, east, west = -24.790135, -39.641186, 16.993043, 15.965546

    transform = rasterio.transform.from_bounds(west, south, east, north, n_cols, n_rows)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_bands": n_bands,
        "transform": transform,
        "north": north, "south": south, "east": east, "west": west,
    }


def align_raster_to_iirs(
    source_path: str,
    iirs_meta: dict,
    output_path: str,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Reprojects and resamples a GeoTIFF to match the IIRS spatial grid.
    """
    n_rows    = iirs_meta["n_rows"]
    n_cols    = iirs_meta["n_cols"]
    transform = iirs_meta["transform"]

    with rasterio.open(source_path) as src:
        dest = np.zeros((n_rows, n_cols), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=LUNAR_CRS,
            resampling=resampling,
        )

        meta = src.meta.copy()
        meta.update({
            "crs":       LUNAR_CRS,
            "transform": transform,
            "width":     n_cols,
            "height":    n_rows,
            "dtype":     "float32",
            "count":     1,
        })
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(dest, 1)

    print(f"[Preprocess] Aligned → {output_path} | shape: {dest.shape}")
    return dest


def compute_quality_stats(data: np.ndarray, name: str):
    valid = data[np.isfinite(data)]
    n_nan = np.sum(~np.isfinite(data))
    print(f"[Stats] {name}: min={valid.min():.4f}, max={valid.max():.4f}, "
          f"mean={valid.mean():.4f}, nan%={100*n_nan/data.size:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iirs_hdr",    required=True)
    parser.add_argument("--iirs_qub",    required=True)
    parser.add_argument("--tmc_dem",     required=True)
    parser.add_argument("--feo_map",     required=True)
    parser.add_argument("--output_dir",  default="data/processed/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[Preprocess] Extracting IIRS metadata...")
    iirs_meta = get_iirs_metadata(args.iirs_hdr, args.iirs_qub)
    print(f"[Preprocess] IIRS grid: {iirs_meta['n_rows']}×{iirs_meta['n_cols']}×{iirs_meta['n_bands']}")
    print(f"[Preprocess] Bounds: N={iirs_meta['north']:.4f}, S={iirs_meta['south']:.4f}, "
          f"E={iirs_meta['east']:.4f}, W={iirs_meta['west']:.4f}")

    # Save metadata as JSON for reference
    import json
    meta_path = os.path.join(args.output_dir, "iirs_metadata.json")
    with open(meta_path, "w") as f:
        # transform is not JSON-serializable; exclude it
        json.dump({k: v for k, v in iirs_meta.items() if k != "transform"}, f, indent=2)

    # Align DEM
    dem_out = os.path.join(args.output_dir, "aligned_tmc2_dem.tif")
    if not os.path.exists(dem_out):
        dem_data = align_raster_to_iirs(args.tmc_dem, iirs_meta, dem_out)
        compute_quality_stats(dem_data, "DEM")
    else:
        print(f"[Preprocess] DEM already exists: {dem_out}")

    # Align FeO
    feo_out = os.path.join(args.output_dir, "aligned_elemental_map.tif")
    if not os.path.exists(feo_out):
        feo_data = align_raster_to_iirs(
            args.feo_map, iirs_meta, feo_out, resampling=Resampling.bilinear
        )
        compute_quality_stats(feo_data, "FeO")
    else:
        print(f"[Preprocess] FeO already exists: {feo_out}")

    print("\n[Preprocess] Done. Files ready for training:")
    print(f"  DEM:  {dem_out}")
    print(f"  FeO:  {feo_out}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/prepare_endmembers.py")
    print(f"  2. python scripts/train.py --config configs/gcsuae_default.yaml")


if __name__ == "__main__":
    main()
