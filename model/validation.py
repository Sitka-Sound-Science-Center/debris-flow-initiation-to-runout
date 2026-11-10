# Initiation Validation (ROC / AUC)
#
# Port of the MATLAB initiation validation. Compares versions of the infinite
# slope model (every combination of colluvium depth and saturation option)
# using the same inventory and background points. Background points are drawn
# only from steep terrain (slope >= slope_threshold).
#
# For each model:
#   - FS from the infinite slope model (infinite FS replaced by fs_cap)
#   - minimum FS within a circular buffer around each inventory point (and
#     around each background point if buffer_background is true)
#   - hit rate (inventory points with FS <= 1), steep area with FS <= 1, and the
#     steep area needed to match the reference model's hit rate
#   - ROC curve and AUC for each background draw (lower FS = more unstable)
#
# Outputs (validation/ in the output directory):
#   summary.csv              one row per model (AUC, TPR/FPR at FS = 1, areas)
#   auc_by_draw.csv          AUC for every model and background draw
#   roc_curves_draw1.csv     ROC curves from the first draw (for plotting)
#   background_points.csv    background points used (row, col, x, y, draw)
#
# Author: Nicolas Mathews

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve

from model.infinite_slope import infinite_slope
from model.initiation import load_initiation_inputs


# -- MATLAB-Equivalent Helpers -- #

def matlab_round(x):
    """Round half away from zero (MATLAB round), not to even (NumPy)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def disk_offsets(r_pix):
    """Row and column offsets of a circular window (dr^2 + dc^2 <= r^2)."""
    dr, dc = np.meshgrid(np.arange(-r_pix, r_pix + 1), np.arange(-r_pix, r_pix + 1),
                         indexing="ij")
    mask = dr ** 2 + dc ** 2 <= r_pix ** 2
    return dr[mask], dc[mask], mask


def buffer_min(A, rows, cols, r_pix):
    """Minimum of A within a circular buffer around each point (MATLAB buffer_min).

    NaN for points whose buffer has no valid values. r_pix = 0 returns the
    value at the single pixel.
    """
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    nr, nc = A.shape
    dr, dc, _ = disk_offsets(r_pix)
    out = np.full(rows.shape, np.inf)
    any_valid = np.zeros(rows.shape, dtype=bool)
    for a, b in zip(dr, dc):
        r, c = rows + a, cols + b
        inside = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
        vals = np.full(rows.shape, np.nan)
        vals[inside] = A[r[inside], c[inside]]
        ok = ~np.isnan(vals)
        out[ok] = np.minimum(out[ok], vals[ok])
        any_valid |= ok
    out[~any_valid] = np.nan
    return out


def matlab_quantile(x, p):
    """MATLAB quantile for a vector (NaNs ignored; positions (i - 0.5)/n)."""
    x = np.sort(np.asarray(x, dtype=float)[~np.isnan(x)])
    n = len(x)
    if n == 0:
        return np.nan
    pos = n * p + 0.5                       # 1-based position of p
    if pos <= 1:
        return x[0]
    if pos >= n:
        return x[-1]
    lo = int(np.floor(pos))
    return x[lo - 1] + (pos - lo) * (x[lo] - x[lo - 1])


# -- Validation -- #

def _num(x):
    """Number as text with at least one decimal (1 -> 1.0, 0.25 -> 0.25)."""
    t = f"{float(x):g}"
    return t if "." in t or "e" in t else t + ".0"


def _labels(value, kind):
    if value == "modeled":
        return "Modeled" if kind == "depth" else "TWI-based"
    return f"{_num(value)} m" if kind == "depth" else f"Uniform {_num(value)}"


def validate_initiation(cfg):
    """Run the ROC/AUC validation for all depth x saturation models."""
    v = cfg["validation"]
    out_dir = Path(cfg["paths"]["output_directory"]) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Validation: loading initiation inputs")
    inp = load_initiation_inputs(cfg)
    z, slope, depth, mw = inp["z"], inp["slope"], inp["depth"], inp["mw"]
    c, phi, gamma, R = inp["c"], inp["phi"], inp["gamma"], inp["R"]
    gamma_w = cfg["initiation"]["gamma_w"]
    nrows, ncols = z.shape

    # Options
    slope_thresh = v["slope_threshold"]
    fs_cap = v["fs_cap"]
    n_bg, n_draws = int(v["background_points"]), int(v["draws"])
    depth_opts = list(v["depth_options"])
    sat_opts = list(v["saturation_options"])
    ref_d = depth_opts.index(v["reference"][0])
    ref_s = sat_opts.index(v["reference"][1])
    n_depth, n_sat = len(depth_opts), len(sat_opts)
    n_models = n_depth * n_sat
    ref_k = ref_s * n_depth + ref_d          # model index k = s * n_depth + d

    # Distances to Pixels
    cell_m = np.mean([inp["x_cellsize"], inp["y_cellsize"]])
    r_pix = int(matlab_round(v["buffer_radius_m"] / cell_m))
    r_pix_bg = r_pix if v["buffer_background"] else 0
    ex_pix = int(matlab_round(v["exclusion_distance_m"] / cell_m))

    # Reference FS (as in the initiation model, infinite values kept)
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = infinite_slope(c, gamma, phi, slope, depth, mw, gamma_w, z=z)

    # Steep Terrain Mask
    steep = (slope >= slope_thresh) & ~np.isnan(z) & ~np.isnan(fs)

    # Inventory Points -> Rows and Columns (MATLAB worldToIntrinsic + round)
    col_f = (inp["x_init"] - R.c) / R.a + 0.5          # 1-based intrinsic
    row_f = (inp["y_init"] - R.f) / R.e + 0.5
    col_i = matlab_round(col_f).astype(int) - 1        # to 0-based
    row_i = matlab_round(row_f).astype(int) - 1
    in_dem = (row_i >= 0) & (row_i < nrows) & (col_i >= 0) & (col_i < ncols)
    row_i, col_i = row_i[in_dem], col_i[in_dem]
    row_all, col_all = row_i.copy(), col_i.copy()      # all in-DEM points

    on_steep = steep[row_i, col_i]
    print(f"Validation: initiation points on slopes >= {slope_thresh:g} deg: "
          f"{on_steep.sum()} of {on_steep.size} ({100 * on_steep.mean():.0f}%)")
    keep = on_steep if v["filter_inventory"] else ~np.isnan(fs[row_i, col_i])
    row_i, col_i = row_i[keep], col_i[keep]
    n_init = row_i.size
    print(f"Validation: using {n_init} inventory points")

    # Background Points (Steep Cells Away from Inventory Points)
    near = np.zeros(z.shape, dtype=bool)
    near[row_all, col_all] = True
    near = ndimage.binary_dilation(near, structure=disk_offsets(ex_pix)[2])
    pool_mask = steep & ~near
    del near

    # Pool in MATLAB (column-major) order, sampled without replacement per draw
    pool = np.flatnonzero(pool_mask.T.reshape(-1))      # column-major linear index
    rng = np.random.default_rng(v["random_seed"])
    idx = np.concatenate([rng.choice(pool, n_bg, replace=False) for _ in range(n_draws)])
    row_bg, col_bg = idx % nrows, idx // nrows
    print(f"Validation: background pool {pool.size:,} steep cells; "
          f"{n_draws} draws x {n_bg} points")
    del pool_mask
    draw_id = np.repeat(np.arange(1, n_draws + 1), n_bg)

    # Evaluate Each Model
    fs_min_init = np.full((n_init, n_models), np.nan)
    fs_min_bg = np.full((n_bg * n_draws, n_models), np.nan)
    hit_rate = np.full(n_models, np.nan)
    area_flag = np.full(n_models, np.nan)
    fs_steep_all = {}
    labels_d, labels_s = [""] * n_models, [""] * n_models

    order = [(ref_s, ref_d)] + [(s, d) for s in range(n_sat) for d in range(n_depth)
                                if (s, d) != (ref_s, ref_d)]
    for n, (s, d) in enumerate(order, start=1):
        k = s * n_depth + d
        labels_d[k], labels_s[k] = _labels(depth_opts[d], "depth"), _labels(sat_opts[s], "sat")
        print(f"Validation: model {n} of {n_models} "
              f"(depth: {labels_d[k]}, saturation: {labels_s[k]})")

        dep = depth if depth_opts[d] == "modeled" else float(depth_opts[d])
        sat = mw if sat_opts[s] == "modeled" else float(sat_opts[s])
        with np.errstate(divide="ignore", invalid="ignore"):
            fs_model = infinite_slope(c, gamma, phi, slope, dep, sat, gamma_w, z=z, fs_cap=fs_cap)

        if k == ref_k:
            fs_check = fs.copy()
            fs_check[np.isinf(fs_check)] = fs_cap
            print(f"  max difference from initiation FS: "
                  f"{np.nanmax(np.abs(fs_model[steep] - fs_check[steep])):.2e}")
            del fs_check

        fs_min_init[:, k] = buffer_min(fs_model, row_i, col_i, r_pix)
        fs_min_bg[:, k] = buffer_min(fs_model, row_bg, col_bg, r_pix_bg)
        fs_steep_all[k] = fs_model[steep]
        hit_rate[k] = np.mean(fs_min_init[:, k] <= 1)
        area_flag[k] = np.mean(fs_steep_all[k] <= 1)
        del fs_model

    # Area Needed to Match the Reference Hit Rate
    t_match = np.array([matlab_quantile(fs_min_init[:, k], hit_rate[ref_k])
                        for k in range(n_models)])
    area_match = np.array([np.mean(fs_steep_all[k] <= t_match[k]) for k in range(n_models)])
    del fs_steep_all

    # ROC / AUC for Each Draw
    auc = np.full((n_draws, n_models), np.nan)
    tpr1 = np.full((n_draws, n_models), np.nan)
    fpr1 = np.full((n_draws, n_models), np.nan)
    roc_rows = []
    ok_init = np.all(~np.isnan(fs_min_init), axis=1)
    lab_all = np.r_[np.ones(n_init), np.zeros(n_bg)]
    for i in range(n_draws):
        sl = slice(i * n_bg, (i + 1) * n_bg)
        ok = np.r_[ok_init, np.all(~np.isnan(fs_min_bg[sl]), axis=1)]
        lab = lab_all[ok]
        for k in range(n_models):
            scores = np.r_[fs_min_init[:, k], fs_min_bg[sl, k]][ok]
            auc[i, k] = roc_auc_score(lab, -scores)       # lower FS = more unstable
            tpr1[i, k] = np.mean(scores[lab == 1] <= 1)
            fpr1[i, k] = np.mean(scores[lab == 0] <= 1)
            if i == 0:
                f, t, th = roc_curve(lab, -scores, drop_intermediate=False)
                roc_rows.append(pd.DataFrame({
                    "depth": labels_d[k], "saturation": labels_s[k],
                    "fpr": f, "tpr": t, "fs_threshold": -th}))

    # Summary Table
    summary = pd.DataFrame({
        "depth": labels_d, "saturation": labels_s,
        "depth_option": [depth_opts[k % n_depth] for k in range(n_models)],
        "saturation_option": [sat_opts[k // n_depth] for k in range(n_models)],
        "reference": [k == ref_k for k in range(n_models)],
        "AUC_mean": auc.mean(axis=0), "AUC_min": auc.min(axis=0), "AUC_max": auc.max(axis=0),
        "TPR_FS1_pct": 100 * tpr1.mean(axis=0),
        "FPR_FS1_pct": 100 * fpr1.mean(axis=0),
        "FPR_FS1_min": 100 * fpr1.min(axis=0), "FPR_FS1_max": 100 * fpr1.max(axis=0),
        "SteepAreaFS1_pct": 100 * area_flag,
        "AreaAtRefHitRate_pct": 100 * area_match,
    })
    summary.to_csv(out_dir / "summary.csv", index=False)

    auc_table = pd.DataFrame(auc, columns=[f"{labels_d[k]} | {labels_s[k]}" for k in range(n_models)])
    auc_table.insert(0, "draw", np.arange(1, n_draws + 1))
    auc_table.to_csv(out_dir / "auc_by_draw.csv", index=False)
    pd.concat(roc_rows).to_csv(out_dir / "roc_curves_draw1.csv", index=False)

    x_world, y_world = inp["x_world"], inp["y_world"]
    pd.DataFrame({"draw": draw_id, "row": row_bg, "col": col_bg,
                  "x": x_world[col_bg], "y": y_world[row_bg]}).to_csv(
        out_dir / "background_points.csv", index=False)

    # Print Summary
    print(f"\nValidation (slope >= {slope_thresh:g} deg, {n_init} inventory points, "
          f"{n_draws} draws x {n_bg} background points)")
    print(f"Reference hit rate: {100 * hit_rate[ref_k]:.1f}%; inventory points with no "
          f"modeled colluvium within buffer: {100 * np.mean(fs_min_init[:, ref_k] >= fs_cap):.1f}%")
    cols = ["depth", "saturation", "AUC_mean", "AUC_min", "AUC_max", "TPR_FS1_pct",
            "FPR_FS1_pct", "SteepAreaFS1_pct", "AreaAtRefHitRate_pct"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nValidation: results in {out_dir}")
    return out_dir
