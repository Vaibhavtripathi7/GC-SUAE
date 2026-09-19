"""
Register auxiliary rasters (TMC-2 DEM, Kaguya MI FeO) onto the IIRS pixel grid
using the per-pixel selenographic coordinates shipped with the IIRS PDS4
bundle (*_d_loc_*_ard.img: Longitude, Latitude, Radius, Height).

Each IIRS pixel centre is projected into the source raster's CRS and the
raster is sampled bilinearly there. Output rasters have the IIRS shape
(lines x samples), float32, nodata = -9999, and carry no map transform: their
pixel grid *is* the IIRS grid. This replaces preprocess_data.py, which fitted
the strip to an axis-aligned lon/lat rectangle and misplaced pixels by up to
~30 px along the strip.

Usage:
    python scripts/register_aux.py \
        --loc  path/to/ch2_iir_..._d_loc_d18_ard.img \
        --dem  path/to/ch2_tmc_..._d_dtm_d18.tif \
        --feo  path/to/kaguya_feo_scene.tif \
        --output_dir data/processed/
"""

import argparse
import os
import re

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import map_coordinates

LUNAR_LONLAT = "+proj=longlat +R=1737400 +no_defs"
OUT_NODATA = -9999.0


def read_loc(img_path: str) -> dict:
    """Read the IIRS location cube; returns dict of (lines, samples) float32 bands."""
    hdr_path = os.path.splitext(img_path)[0] + ".hdr"
    hdr = open(hdr_path, errors="ignore").read()
    samples = int(re.search(r"samples\s*=\s*(\d+)", hdr).group(1))
    lines   = int(re.search(r"lines\s*=\s*(\d+)", hdr).group(1))
    bands   = int(re.search(r"bands\s*=\s*(\d+)", hdr).group(1))
    order   = "<" if re.search(r"byte order\s*=\s*0", hdr) else ">"
    names   = re.search(r"band names\s*=\s*\{([^}]*)\}", hdr).group(1)
    names   = [n.strip() for n in names.split(",")]
    cube = np.fromfile(img_path, dtype=order + "f4").reshape(bands, lines, samples)
    return {name: cube[i] for i, name in enumerate(names)}


def sample_raster(path: str, lon: np.ndarray, lat: np.ndarray, band: int = 1):
    """
    Bilinearly sample `path` at the given lon/lat arrays (any shape).
    Returns (values, valid) with OUT_NODATA where the source is nodata/outside.
    """
    shape = lon.shape
    with rasterio.open(path) as src:
        xs, ys = warp_transform(LUNAR_LONLAT, src.crs, lon.ravel().tolist(), lat.ravel().tolist())
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys, op=lambda v: v)
        rows, cols = np.asarray(rows, dtype=np.float64), np.asarray(cols, dtype=np.float64)
        # rowcol gives the containing cell index in continuous coords; pixel
        # centres sit at +0.5, so shift to centre-based coordinates for interpolation
        rows -= 0.5
        cols -= 0.5

        inside = (rows >= 0) & (rows <= src.height - 1) & (cols >= 0) & (cols <= src.width - 1)
        values = np.full(rows.size, OUT_NODATA, dtype=np.float32)
        if not inside.any():
            return values.reshape(shape), inside.reshape(shape)

        r0, r1 = int(np.floor(rows[inside].min())), int(np.ceil(rows[inside].max())) + 1
        c0, c1 = int(np.floor(cols[inside].min())), int(np.ceil(cols[inside].max())) + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, src.height), min(c1, src.width)
        data = src.read(band, window=((r0, r1), (c0, c1))).astype(np.float32)
        nodata = src.nodata
        valid_src = np.isfinite(data)
        if nodata is not None and np.isfinite(nodata):
            valid_src &= data != np.float32(nodata)
        if np.issubdtype(src.dtypes[band - 1], np.integer):
            valid_src &= data > -30000
        data = np.where(valid_src, data, 0.0)

        coords = np.vstack([rows[inside] - r0, cols[inside] - c0])
        interp = map_coordinates(data, coords, order=1, mode="nearest")
        # a sample is valid only if every contributing source pixel is valid
        wsum = map_coordinates(valid_src.astype(np.float32), coords, order=1, mode="nearest")
        ok = wsum > 0.999
        out = np.where(ok, interp, OUT_NODATA).astype(np.float32)
        values[inside] = out
        valid = np.zeros(rows.size, dtype=bool)
        valid[inside] = ok
        return values.reshape(shape), valid.reshape(shape)


def write_grid(path: str, arr: np.ndarray):
    """Write an IIRS-grid raster (no georeference; the grid is the IIRS grid)."""
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", nodata=OUT_NODATA, compress="deflate") as dst:
        dst.write(arr.astype(np.float32), 1)


def report(name: str, values: np.ndarray, valid: np.ndarray):
    v = values[valid]
    print(f"[Register] {name}: valid {100*valid.mean():.1f}% | "
          f"min {v.min():.3f} | 2nd {np.percentile(v, 2):.3f} | median {np.median(v):.3f} | "
          f"98th {np.percentile(v, 98):.3f} | max {v.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True, help="IIRS *_d_loc_*_ard.img (hdr alongside)")
    ap.add_argument("--dem", required=True, help="TMC-2 DTM GeoTIFF")
    ap.add_argument("--feo", required=False, help="Kaguya MI FeO wt%% GeoTIFF (scene crop)")
    ap.add_argument("--output_dir", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loc = read_loc(args.loc)
    lon, lat = loc["Longitude"], loc["Latitude"]
    print(f"[Register] IIRS grid {lon.shape} | lon {lon.min():.4f}..{lon.max():.4f} | "
          f"lat {lat.min():.4f}..{lat.max():.4f}")

    dem, dem_valid = sample_raster(args.dem, lon, lat)
    report("TMC-2 DEM", dem, dem_valid)
    write_grid(os.path.join(args.output_dir, "aligned_tmc2_dem.tif"), dem)

    # Registration check: the bundle's own per-pixel Height should track the
    # sampled DEM closely if the geometry is right (offset is allowed, the
    # datums may differ; correlation should be ~0.99).
    if "Height" in loc:
        h = loc["Height"][dem_valid]; d = dem[dem_valid]
        r = np.corrcoef(h, d)[0, 1]
        print(f"[Register] DEM vs IIRS-bundle Height: corr {r:.4f} | "
              f"median offset {np.median(d - h):+.1f} m | MAD {np.median(np.abs((d - h) - np.median(d - h))):.1f} m")

    if args.feo:
        feo, feo_valid = sample_raster(args.feo, lon, lat)
        report("Kaguya FeO", feo, feo_valid)
        write_grid(os.path.join(args.output_dir, "aligned_elemental_map.tif"), feo)
        both = dem_valid & feo_valid
        print(f"[Register] pixels valid in both DEM and FeO: {100*both.mean():.1f}%")

    print(f"[Register] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
