# Landslide Initiation Model
# Description: Computes factor of safety using the infinite slope method with
# modeled colluvium depth, TWI-based saturation, and soil series strength.
# Groups unstable cells into discrete landslides and exports initiation points
# (x, y, volume) for Grfin Tools.
#
# Author: Nicolas Mathews

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy import ndimage
from scipy.optimize import curve_fit

from model.slope_and_aspect import slope_aspect
from model.infinite_slope import infinite_slope
from model.raster_io import save_raster


def logistic(x, L, k, x0):
    """Logistic relationship between TWI and saturation ratio."""
    return L / (1 + np.exp(-k * (x - x0)))


def load_initiation_inputs(cfg):
    """Load and prepare all initiation model inputs (shared with validation).

    Returns a dictionary with the DEM, slope, colluvium depth, saturation ratio,
    soil strength rasters, georeferencing, and inventory initiation points.
    """

    paths = cfg["paths"]
    init_cfg = cfg["initiation"]

    ## Import DEM
    print("Loading DEM...", end="")
    with rasterio.open(paths["dem"]) as src:
        z = src.read(1).astype(float)   # elevations (m)
        R = src.transform               # georeferencing data
        profile = src.profile           # raster settings: size, CRS, transform, data type
        x_cellsize = src.res[0]         # size of each raster cell, x-direction (m)
        y_cellsize = src.res[1]         # size of each raster cell, y-direction (m)

    z[z < 0] = np.nan   # make negative elevations NaN
    print(" Complete.")

    ## DEM Derivatives

    # # World Coordinates
    nrows, ncols = z.shape
    x_world = R.c + x_cellsize * (np.arange(ncols) + 0.5)   # column centers
    y_world = R.f - y_cellsize * (np.arange(nrows) + 0.5)   # row centers, top to bottom

    # Calculate Slope
    slope = slope_aspect(z, x_cellsize, y_cellsize)   # slope (degrees)

    ## Load Colluvium Depth Raster
    soil_depth_path = Path(paths["soil_depth"])
    if not soil_depth_path.exists():
        raise FileNotFoundError(f"Soil depth raster not found: {soil_depth_path}")

    print("Loading Soil Depth Raster...", end="")
    with rasterio.open(soil_depth_path) as src:
        depth = src.read(1).astype(float)
    print(" Complete.")

    # Set Upper Limit for Colluvium Depth
    depth_max = init_cfg["colluvium_depth_max"]
    depth[depth > depth_max] = depth_max

    ## Load Topographic Wetness Index (TWI) Raster
    with rasterio.open(paths["twi"]) as src:
        twi = src.read(1).astype(float)   # TWI values

    ## Load TWI-Saturation Relationship (from Rinderer et al., 2014)
    twi_table = pd.read_csv(paths["twi_saturation_relationship"])
    twi_data = twi_table["TWI [ln(m)]"].to_numpy()
    gw_depth_data = twi_table["Median Groundwater Level Relative to Soil Depth"].to_numpy()

    # Limit GW Depth to Realistic Values
    gw_depth_data = np.clip(gw_depth_data, 0, 1)

    ## Load Inventory Shapefile (Initiation Points)
    initiation_pts = gpd.read_file(paths["inventory_points"])
    x_init = initiation_pts.geometry.x.to_numpy()
    y_init = initiation_pts.geometry.y.to_numpy()

    # Convert to Row and Column (Like worldToIntrinsic in MATLAB)
    row_init, col_init = rasterio.transform.rowcol(R, x_init, y_init)
    row_init = np.array(row_init)
    col_init = np.array(col_init)

    ## Import Soil ID Map, Import Soil Strength Lookup Table, and Build Rasters
    c_roots = init_cfg["root_cohesion"]   # root cohesion (kPa)

    if init_cfg["phi_uniform"] is not None:   # Uniform Soil Parameters

        print("Assigning Uniform Soil Strength...", end="")

        # Build Soil Parameter Rasters
        phi = np.full_like(z, init_cfg["phi_uniform"])
        c = np.full_like(z, init_cfg["coh_uniform"] + c_roots)
        gamma = np.full_like(z, init_cfg["gamma_uniform"])

    else:   # Soil Lookup Table

        print("Assigning Soil Strength Using Soil Series...", end="")

        # Import Soil ID Raster
        with rasterio.open(paths["soil_id"]) as src:
            soil_id = src.read(1).astype(float)
            if src.nodata is not None:
                soil_id[soil_id == src.nodata] = np.nan
        soil_id[np.isnan(soil_id)] = 0
        soil_id = soil_id.astype(int)

        # Import Lookup Table
        soil_lookup = pd.read_csv(paths["soil_lookup_table"], encoding="cp1252")

        # Sample Soil Series at Inventory Points
        inside = (row_init >= 0) & (row_init < nrows) & (col_init >= 0) & (col_init < ncols)
        soil_series_init = soil_id[row_init[inside], col_init[inside]]

        # Find Most Prevalent Landslide Soil Series and Assign its Parameters to Undefined Soils
        soil_series_mode = np.bincount(soil_series_init[soil_series_init > 0]).argmax()

        phi_undefined = soil_lookup.iloc[soil_series_mode - 1, 3]
        c_undefined = soil_lookup.iloc[soil_series_mode - 1, 4]
        gamma_undefined = soil_lookup.iloc[soil_series_mode - 1, 9]

        # Build Soil Parameter Rasters
        phi = np.full_like(z, phi_undefined)
        c = np.full_like(z, c_undefined + c_roots)
        gamma = np.full_like(z, gamma_undefined)

        for i in range(1, len(soil_lookup) + 1):   # loop through soil series (IDs 1 to n)
            phi[soil_id == i] = soil_lookup.iloc[i - 1, 3]
            c[soil_id == i] = soil_lookup.iloc[i - 1, 4] + c_roots
            gamma[soil_id == i] = soil_lookup.iloc[i - 1, 9]   # mean bulk unit weight (kN/m3)

    # Make NaNs Match DEM
    c[np.isnan(z)] = np.nan
    phi[np.isnan(z)] = np.nan
    gamma[np.isnan(z)] = np.nan

    print(" Complete.")

    ## Fit Logistic Curve to TWI-Saturation Relationship
    p0 = [1, 1, np.median(twi_data)]   # initial guess [L, k, x0]
    p_fit, _ = curve_fit(logistic, twi_data, gw_depth_data, p0=p0)

    ## Compute Saturation Ratio Using TWI Raster and Curve Fit
    mw = logistic(twi, *p_fit)
    mw[mw > 1] = 1   # limit saturation ratio to 1

    return {
        "z": z, "R": R, "profile": profile,
        "x_cellsize": x_cellsize, "y_cellsize": y_cellsize,
        "x_world": x_world, "y_world": y_world,
        "slope": slope, "depth": depth, "mw": mw,
        "c": c, "phi": phi, "gamma": gamma,
        "x_init": x_init, "y_init": y_init,
    }


def initiation(cfg):
    """Run the landslide initiation model and return the path to the initiation points CSV."""

    paths = cfg["paths"]
    init_cfg = cfg["initiation"]

    # Output Directory (Create if Nonexistent)
    out_dir = Path(paths["output_directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ## Load Inputs (DEM, Slope, Colluvium Depth, Saturation, Soil Strength)
    inp = load_initiation_inputs(cfg)
    z, R, profile = inp["z"], inp["R"], inp["profile"]
    x_cellsize, y_cellsize = inp["x_cellsize"], inp["y_cellsize"]
    x_world, y_world = inp["x_world"], inp["y_world"]
    slope, depth, mw = inp["slope"], inp["depth"], inp["mw"]
    c, phi, gamma = inp["c"], inp["phi"], inp["gamma"]

    ## Compute Factor of Safety Using Infinite Slope
    print("Computing Factor of Safety...", end="")
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = infinite_slope(c, gamma, phi, slope, depth, mw, init_cfg["gamma_w"], z=z)
    print(" Complete.")

    ## Create Binary Landslide Map (Source Areas)
    lsio = fs <= 1   # NaN and inf cells are automatically False

    ## Group Unstable Clusters into Discrete Landslides
    labels, n_slides = ndimage.label(lsio, structure=np.ones((3, 3)))   # 8-connected, like bwconncomp
    ids = np.arange(1, n_slides + 1)
    print(f"{n_slides} modeled landslides")

    ## Find Volume of Each Landslide
    cell_area = x_cellsize * y_cellsize
    ls_vol = ndimage.sum(depth * cell_area, labels, ids)   # landslide volume (m^3)

    ## Find Location of Maximum Elevation of Each Landslide (Debris-Flow Initiation Point)
    cell_idx = np.flatnonzero(labels)        # indices of all unstable cells only
    cell_lab = labels.flat[cell_idx]         # which landslide each cell belongs to
    cell_z = z.flat[cell_idx]                # elevation of each cell

    order = np.lexsort((cell_z, cell_lab))   # sort by landslide, then by elevation
    cell_idx = cell_idx[order]
    cell_lab = cell_lab[order]

    # Last Cell of Each Landslide After Sorting = Highest Elevation
    is_last = np.r_[cell_lab[1:] != cell_lab[:-1], True]
    top_idx = cell_idx[is_last]

    rows, cols = np.unravel_index(top_idx, labels.shape)

    ls_init_x = x_world[cols]
    ls_init_y = y_world[rows]

    ## Create and Export Table of Debris-Flow Initiation Points as CSV (x, y, volume)
    init_points_path = out_dir / "initiation_points.csv"

    if cfg["export"]["initiation_table"]:
        init_table = pd.DataFrame({
            "x": ls_init_x,
            "y": ls_init_y,
            "volume": ls_vol,
        })
        init_table.to_csv(init_points_path, index=False, header=False)
        print("Saved:", init_points_path)

    ## Export FS Raster
    if cfg["export"]["fs_map"]:
        save_raster(out_dir / "fs.tif", fs, profile)
        print("Saved:", out_dir / "fs.tif")

    ## Export Initiation Area Raster (Source Areas)
    if cfg["export"]["source_areas"]:
        save_raster(out_dir / "source_areas.tif", lsio, profile, dtype="uint8", nodata=255)
        print("Saved:", out_dir / "source_areas.tif")

    return init_points_path