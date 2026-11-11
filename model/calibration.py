# Runout Calibration Against Inventory and Calibrated Maps
#
# Port of runout_calibration_against_inventory_and_build_model_maps.m.
#
# Watersheds: a polygon file (shapefile, GeoPackage, GeoJSON) rasterized onto
# the DEM grid (cell centers), or a raster of watershed IDs on the DEM grid.
# Inventory areas: landslide polygons rasterized the same way (1 = landslide),
# or a raster (1 = landslide) on the DEM grid.
#
# Runout per scenario: Grfin inundation > 0 (negative boundary values and
# NoData count as not inundated), plus any cell inundated by the no-growth run.
#
# Calibration (inventory scenarios), per watershed:
#   sim = inventory + 2 * runout   ->  1 = FN, 2 = FP, 3 = TP
#   similarity index = (TP - FN - FP) / (TP + FN + FP)
#   (Gorr et al., 2022; Heiser et al., 2017)
#   Best scenario = highest similarity index. Ties go to the first scenario in
#   MATLAB column order (lowest growth factor first, then lowest slope).
#   Watersheds without inventory landslides are not calibrated; they use the
#   default transition slope and growth factor from the settings file
#   (calibration: default_transition_slope, default_growth_factor).
#
# Maps (for any point set, using the inventory-calibrated parameters):
#   optimized runout:         runout of the best scenario in each watershed (1 = inundated)
#   optimized runout growth:  at the best transition slope, the lowest growth
#                             factor that inundates each cell (0 = not inundated)
#
# Author: Nicolas Mathews

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# -- Raster Helpers -- #

def _same_grid(src, ref_shape, ref_transform, label):
    if src.shape != ref_shape or not np.allclose(tuple(src.transform), tuple(ref_transform)):
        raise ValueError(f"{label} is not on the same grid as the DEM.")


VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}


def load_watersheds(path, id_field, shape, transform, crs, out_dir):
    """Watershed IDs on the DEM grid, from polygons or a raster.

    Polygons: rasterized at cell centers. Each polygon (or each unique value of
    id_field, if given) gets an integer ID 1..n; the rasterized grid is saved
    as watersheds.tif for checking. Raster: read as is (must match the DEM grid).
    Returns (ID array as float64 with 0 = no watershed, NoData value,
    {ID: name} or None, and for polygons the original polygons with their IDs
    (for exporting results) or None for rasters).
    """
    path = Path(path)
    if path.suffix.lower() in VECTOR_SUFFIXES:
        import geopandas as gpd
        from rasterio.features import rasterize

        gdf_orig = gpd.read_file(path)
        gdf = gdf_orig
        if gdf.crs is not None and crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        if id_field:
            if id_field not in gdf.columns:
                raise KeyError(f"watershed_id_field '{id_field}' not in {path.name}; "
                               f"fields: {[c for c in gdf.columns if c != 'geometry']}")
            values = gdf[id_field].astype(str)
            names = {i + 1: v for i, v in enumerate(sorted(values.unique()))}
            code = {v: i for i, v in names.items()}
            ids = values.map(code).to_numpy()
        else:
            ids = np.arange(1, len(gdf) + 1)
            names = {int(i): str(i) for i in ids}
        ws = rasterize(((geom, int(i)) for geom, i in zip(gdf.geometry, ids) if geom is not None),
                       out_shape=shape, transform=transform, fill=0, dtype="int32")
        profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
                   "dtype": "int32", "crs": crs, "transform": transform, "nodata": 0,
                   "compress": "lzw"}
        with rasterio.open(Path(out_dir) / "watersheds.tif", "w", **profile) as dst:
            dst.write(ws, 1)
        print(f"Calibration: rasterized {len(gdf)} watershed polygons from {path.name}")
        polygons = {"gdf": gdf_orig, "ids": np.asarray(ids), "source": path}
        return ws.astype("float64"), 0, names, polygons

    with rasterio.open(path) as src:
        _same_grid(src, shape, transform, "Watershed raster")
        return src.read(1).astype("float64"), src.nodata, None, None


def load_inventory_areas(path, shape, transform, crs, out_dir):
    """Inventory landslide areas on the DEM grid (1 = landslide, 0 = not).

    Polygons: rasterized at cell centers and saved as inventory_areas.tif for
    checking. Raster: read as is (must match the DEM grid; NaN counts as 0).
    """
    path = Path(path)
    if path.suffix.lower() in VECTOR_SUFFIXES:
        import geopandas as gpd
        from rasterio.features import rasterize

        gdf = gpd.read_file(path)
        if gdf.crs is not None and crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        inv = rasterize(((geom, 1) for geom in gdf.geometry if geom is not None),
                        out_shape=shape, transform=transform, fill=0, dtype="uint8")
        profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1,
                   "dtype": "uint8", "crs": crs, "transform": transform, "nodata": None,
                   "compress": "lzw"}
        with rasterio.open(Path(out_dir) / "inventory_areas.tif", "w", **profile) as dst:
            dst.write(inv, 1)
        print(f"Calibration: rasterized {len(gdf)} inventory landslide polygons from {path.name}")
        return inv.astype("float64")

    with rasterio.open(path) as src:
        _same_grid(src, shape, transform, "Inventory raster")
        inv = src.read(1).astype("float64")
    inv[np.isnan(inv)] = 0
    return inv


def export_watershed_results(polygons, table, out_dir):
    """Copy of the watershed polygons with calibration results added.

    Fields: WSID (watershed ID 1..n, as in the calibration tables),
    similarity (highest similarity index; null where not calibrated),
    slope and growth (transition slope angle and growth factor used in the
    calibrated maps), source ('calibrated' or 'default').
    """
    gdf = polygons["gdf"].copy()
    rows = {int(w): r for w, r in zip(table["WSID"], table.itertuples(index=False))}
    get = lambda i, f, empty: getattr(rows[i], f) if i in rows else empty
    ids = [int(i) for i in polygons["ids"]]
    gdf["WSID"] = ids
    gdf["similarity"] = [get(i, "similarity_index", np.nan) for i in ids]
    gdf["slope"] = [get(i, "transition_slope", np.nan) for i in ids]
    gdf["growth"] = [get(i, "growth_factor", np.nan) for i in ids]
    gdf["source"] = [("calibrated" if get(i, "parameter_source", "") == "calibrated"
                      else "default" if i in rows else "") for i in ids]
    out = Path(out_dir) / f"{Path(polygons['source']).stem}_calibrated.shp"
    gdf.to_file(out)
    print(f"Calibration: wrote {out.name}")
    return out


def load_runout(path, nogrowth=None, valid=None):
    """Runout mask: Grfin inundation > 0, OR'd with the no-growth mask.

    Returns only the cells in `valid` (1-D) when valid is given.
    """
    with rasterio.open(path) as src:
        r = src.read(1)
    runout = r > 0
    if valid is not None:
        runout = runout[valid]
    if nogrowth is not None:
        runout |= nogrowth
    return runout


# -- Core Calculations (pure NumPy) -- #

def count_per_watershed(inv, runout, ws_idx, n_ws):
    """FN, FP, TP cell counts per watershed (sim = inventory + 2 * runout)."""
    sim = inv + 2.0 * runout
    fn = np.bincount(ws_idx[sim == 1], minlength=n_ws)
    fp = np.bincount(ws_idx[sim == 2], minlength=n_ws)
    tp = np.bincount(ws_idx[sim == 3], minlength=n_ws)
    return fn, fp, tp


def similarity(tp, fp, fn):
    """Similarity index and its components (NaN where TP + FP + FN = 0)."""
    tot = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b, g = tp / tot, fn / tot, fp / tot
    return a - b - g, a, b, g


def best_scenarios(si):
    """Best (slope index, growth index) per watershed, MATLAB tie order.

    si has shape (n_watersheds, n_slopes, n_growth). Returns i, j, si_max;
    i and j are -1 where every value is NaN.
    """
    n, ns, ng = si.shape
    flat = si.transpose(0, 2, 1).reshape(n, ns * ng)     # column-major order
    best_i = np.full(n, -1)
    best_j = np.full(n, -1)
    si_max = np.full(n, np.nan)
    for k in range(n):
        if np.all(np.isnan(flat[k])):
            continue
        idx = int(np.nanargmax(flat[k]))                  # first occurrence of max
        best_j[k], best_i[k] = divmod(idx, ns)
        si_max[k] = flat[k, idx]
    return best_i, best_j, si_max


# -- Calibration -- #

def calibrate(cfg, scenarios, nogrowth_path):
    """Calibrate scenarios (inventory points) against the inventory areas.

    scenarios      {(slope, growth): inundation raster}, from run_scenarios()
    nogrowth_path  no-growth inundation raster, from run_nogrowth()
    Returns a dictionary used by build_maps(); also writes calibration tables.
    """
    s = cfg["scenarios"]
    c = cfg["calibration"]
    slopes = list(s["transition_slope_angles"])
    growths = list(s["growth_factors"])
    ns, ng = len(slopes), len(growths)

    # Defaults for watersheds without inventory landslides (must be a scenario)
    d_slope, d_growth = c["default_transition_slope"], c["default_growth_factor"]
    if d_slope not in slopes or d_growth not in growths:
        raise ValueError(
            f"calibration defaults (slope {d_slope}, growth {d_growth}) must be in "
            f"transition_slope_angles {slopes} and growth_factors {growths}.")
    d_i, d_j = slopes.index(d_slope), growths.index(d_growth)
    out_dir = Path(cfg["paths"]["output_directory"]) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference grid (DEM)
    with rasterio.open(cfg["paths"]["dem"]) as src:
        shape, transform, crs = src.shape, src.transform, src.crs

    # Watersheds (IDs >= 1 and not NoData)
    print("Calibration: loading watersheds and inventory")
    ws, ws_nodata, ws_names, ws_polygons = load_watersheds(
        cfg["paths"]["watershed"], c.get("watershed_id_field"), shape, transform, crs, out_dir)
    valid = np.isfinite(ws) & (ws >= 1)
    if ws_nodata is not None:
        valid &= ws != ws_nodata
    del ws_nodata
    ids, ws_idx = np.unique(ws[valid], return_inverse=True)
    n = len(ids)
    print(f"Calibration: {n} watersheds on the DEM grid (IDs {ids.min():g} to {ids.max():g})")

    # Inventory landslide areas (polygons or raster; NaN as 0, as in the MATLAB)
    inv = load_inventory_areas(cfg["paths"]["inventory_areas"], shape, transform, crs,
                               out_dir)[valid]

    # No-growth runout
    with rasterio.open(nogrowth_path) as src:
        _same_grid(src, shape, transform, "No-growth raster")
    nogrowth = load_runout(nogrowth_path, valid=valid)

    # Counts for every scenario
    FN = np.zeros((n, ns, ng)); FP = np.zeros((n, ns, ng)); TP = np.zeros((n, ns, ng))
    for i, a in enumerate(slopes):
        for j, g in enumerate(growths):
            print(f"Calibration: scenario slope {a:g}, growth {g:g}")
            runout = load_runout(scenarios[(a, g)], nogrowth, valid)
            FN[:, i, j], FP[:, i, j], TP[:, i, j] = count_per_watershed(inv, runout, ws_idx, n)

    si, at, bt, gt = similarity(TP, FP, FN)
    best_i, best_j, si_max = best_scenarios(si)

    # Watersheds without inventory landslides use the defaults
    has_inventory = (TP[:, 0, 0] + FN[:, 0, 0]) > 0      # inventory cells, any scenario
    map_i = np.where(has_inventory, best_i, d_i)
    map_j = np.where(has_inventory, best_j, d_j)

    # Table of parameters per watershed
    rows = []
    for k in range(n):
        i, j = map_i[k], map_j[k]
        cal_k = bool(has_inventory[k])
        rows.append({
            "WSID": int(ids[k]),
            "watershed_name": ws_names.get(int(ids[k]), "") if ws_names else "",
            "parameter_source": "calibrated" if cal_k else "default (no inventory)",
            "transition_slope": slopes[i],
            "growth_factor": growths[j],
            "similarity_index": si_max[k] if cal_k else np.nan,
            "alpha_t": at[k, i, j] if cal_k else np.nan,
            "beta_t": bt[k, i, j] if cal_k else np.nan,
            "gamma_t": gt[k, i, j] if cal_k else np.nan,
            "TP": TP[k, i, j], "FP": FP[k, i, j], "FN": FN[k, i, j],
        })
    best_table = pd.DataFrame(rows)
    best_table.to_csv(out_dir / "best_parameters_by_watershed.csv", index=False)
    if ws_polygons is not None:
        export_watershed_results(ws_polygons, best_table, out_dir)

    # Every watershed x scenario (for figures and checking)
    long = pd.DataFrame([
        {"WSID": int(ids[k]),
         "watershed_name": ws_names.get(int(ids[k]), "") if ws_names else "",
         "transition_slope": slopes[i], "growth_factor": growths[j],
         "similarity_index": si[k, i, j], "TP": TP[k, i, j], "FP": FP[k, i, j], "FN": FN[k, i, j]}
        for k in range(n) for i in range(ns) for j in range(ng)])
    long.to_csv(out_dir / "similarity_all_scenarios.csv", index=False)

    print(f"Calibration: {has_inventory.sum()} watersheds calibrated, "
          f"{(~has_inventory).sum()} without inventory landslides set to the defaults "
          f"(slope {d_slope:g}, growth {d_growth:g}); tables in {out_dir}")
    return {"valid": valid, "ws_idx": ws_idx, "map_i": map_i, "map_j": map_j,
            "shape": shape, "out_dir": out_dir, "slopes": slopes, "growths": growths}


# -- Calibrated Maps -- #

def build_maps(cfg, cal, scenarios, nogrowth_path, name):
    """Optimized runout and growth-range rasters for one point set.

    Uses the inventory-calibrated parameters in `cal` (from calibrate()).
    Writes <name>_optimized_runout.tif and <name>_optimized_runout_growth.tif.
    """
    valid, ws_idx = cal["valid"], cal["ws_idx"]
    slopes, growths = cal["slopes"], cal["growths"]
    cell_i, cell_j = cal["map_i"][ws_idx], cal["map_j"][ws_idx]   # per valid cell
    nogrowth = load_runout(nogrowth_path, valid=valid)

    opt = np.zeros(valid.sum(), dtype="uint8")
    opt_growth = np.zeros(valid.sum(), dtype="int16")
    for i in np.unique(cell_i):
        for j in reversed(range(len(growths))):           # highest growth first
            sel_slope = cell_i == i
            sel_best = sel_slope & (cell_j == j)
            if not sel_slope.any():
                continue
            runout = load_runout(scenarios[(slopes[i], growths[j])], nogrowth, valid)
            opt[sel_best] = runout[sel_best]
            opt_growth[sel_slope & runout] = growths[j]   # lower growth overwrites
    print(f"Maps ({name}): built from {len(np.unique(cell_i))} transition slope(s)")

    with rasterio.open(nogrowth_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, compress="lzw", nodata=None)

    paths = {}
    for label, arr, dtype in [("optimized_runout", opt, "uint8"),
                              ("optimized_runout_growth", opt_growth, "int16")]:
        full = np.zeros(cal["shape"], dtype=dtype)
        full[valid] = arr
        path = cal["out_dir"] / f"{name}_{label}.tif"
        profile.update(dtype=dtype)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(full, 1)
        paths[label] = path
        print(f"Maps ({name}): wrote {path.name}")
    return paths
