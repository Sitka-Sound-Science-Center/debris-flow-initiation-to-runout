# Growth Zones
#
# Step A: exaggerated runout. Grfin INUNDATION from the initiation points with
#         exaggerated volume-area coefficients (custom alpha1, alpha2), so the
#         runout reaches every channel a debris flow could plausibly use.
#         Volumes are used as given in the points file unless a volume_factor
#         is passed (used for the modeled points; inventory volumes are
#         prepared outside the code).
# Step B: growth zones = Grfin stream cells inside the exaggerated runout where
#         cell slope >= each transition slope angle (1 = growth zone, 0 = not).
#         One raster per transition slope angle. Slope is computed cell by cell
#         with Horn's method, matching the QGIS Slope tool.
#
# Author: Nicolas Mathews

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from model.grfin import run_grfin


def run_exaggerated(cfg, points_csv, name, volume_factor=1.0):
    """Run the exaggerated inundation scenario from a set of initiation points.

    points_csv     x, y, volume CSV (no header), e.g. initiation_points.csv
    name           short label for this point set, e.g. "modeled" or "inventory"
    volume_factor  multiplier on the volumes for this run only (1.0 = as given);
                   a scaled copy of the points file is written to the run folder
    Returns the path to the exaggerated inundation raster.
    """
    g = cfg["growth_zones"]
    dem = Path(cfg["paths"]["dem"]).resolve()
    run_dir = Path(cfg["paths"]["output_directory"]) / f"exaggerated_{name}"
    inundation = run_dir.resolve() / f"{dem.stem}_inundation.tif"

    if inundation.exists() and not g["recompute"]:
        print(f"Exaggerated runout ({name}): using existing {inundation.name}")
        return inundation

    points_csv = Path(points_csv).resolve()
    if not points_csv.exists():
        raise FileNotFoundError(f"Initiation points file not found: {points_csv}")

    # Optional volume scaling for this run (written as a copy, original untouched)
    if volume_factor != 1.0:
        run_dir.mkdir(parents=True, exist_ok=True)
        pts = pd.read_csv(points_csv, header=None)
        pts[2] = pts[2] * volume_factor
        scaled_csv = run_dir.resolve() / f"{name}_volumes_x{volume_factor:g}.csv"
        pts.to_csv(scaled_csv, index=False, header=False)
        print(f"Exaggerated runout ({name}): volumes x {volume_factor:g} -> {scaled_csv.name}")
        points_csv = scaled_csv

    settings = {
        "setup": {
            "title": f"Exaggerated runout from {name} initiation points",
            "demfilename": dem,
            "num_procs": cfg["streams"]["num_procs"],
        },
        "options": {
            "inundation": True,
            "hl": False,
            "growth": False,
            "standalone_drainage_network": False,
        },
        "inundation_options": {
            "flowtype": "custom",
            "xsec_alpha1": float(g["exaggerated_alpha1"]),
            "plan_alpha2": float(g["exaggerated_alpha2"]),
            "hbar": bool(g["hbar"]),
            "user_specified_volumes_file": points_csv,
        },
    }
    run_grfin(settings, run_dir, expect=[inundation])
    print(f"Exaggerated runout ({name}): wrote {inundation}")
    return inundation


def slope_degrees(dem_path, out_path):
    """Cell-by-cell slope (degrees) using Horn's 3x3 method, as in QGIS Slope.

    Missing neighbors (DEM edge or NoData) take the center cell's elevation,
    as QGIS does. Saved once to out_path and reused afterwards.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return out_path

    with rasterio.open(dem_path) as src:
        z = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = src.profile.copy()
        dx, dy = abs(src.transform.a), abs(src.transform.e)

    zp = np.pad(z, 1, constant_values=np.nan)
    rows, cols = z.shape

    def nb(i, j):
        """Neighbor offset (i, j) with missing values replaced by the center."""
        a = zp[1 + i:1 + i + rows, 1 + j:1 + j + cols]
        return np.where(np.isnan(a), z, a)

    a, b, c = nb(-1, -1), nb(-1, 0), nb(-1, 1)
    d, f = nb(0, -1), nb(0, 1)
    g, h, i = nb(1, -1), nb(1, 0), nb(1, 1)
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    slope[np.isnan(z)] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(slope), -9999.0, slope).astype("float32"), 1)
    print(f"Slope: wrote {out_path}")
    return out_path


def make_growth_zones(cfg, inundation_path, stream_path, slope_path, name):
    """Growth zones for each transition slope angle.

    Growth zone = Grfin stream cell AND inside the exaggerated runout AND
    cell slope >= transition slope angle.

    inundation_path  exaggerated inundation raster from run_exaggerated()
    stream_path      Grfin stream raster (1 = stream)
    slope_path       slope raster in degrees, from slope_degrees()
    name             label for this point set, e.g. "modeled" or "inventory"
    Returns {angle: path} for growth_zone_<name>_<angle>deg.tif files
    (saved in the exaggerated run folder).
    """
    inundation_path = Path(inundation_path)
    angles = cfg["scenarios"]["transition_slope_angles"]

    with rasterio.open(inundation_path) as inu, \
         rasterio.open(stream_path) as stm, \
         rasterio.open(slope_path) as slp:
        for r in (stm, slp):
            if r.shape != inu.shape or not np.allclose(tuple(r.transform),
                                                       tuple(inu.transform)):
                raise ValueError(f"{Path(r.name).name} is not on the same grid "
                                 f"as {inundation_path.name}.")
        # Grfin inundation: 10, 20, -10, -20 = inundated; 0 or NoData = not
        inundated = inu.read(1, masked=True).filled(0) != 0
        stream = stm.read(1, masked=True).filled(0) == 1
        slope = slp.read(1, masked=True).filled(-1.0)
        profile = stm.profile.copy()

    base = inundated & stream
    profile.update(dtype="uint8", nodata=None, count=1, compress="lzw")
    zones = {}
    for angle in angles:
        zone = (base & (slope >= angle)).astype("uint8")
        out = inundation_path.parent / f"growth_zone_{name}_{angle:g}deg.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(zone, 1)
        zones[angle] = out
        print(f"Growth zone ({name}, {angle:g} deg): {zone.sum():,} cells")
    return zones
