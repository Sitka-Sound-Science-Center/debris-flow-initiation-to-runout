# Runout Scenarios
#
# One Grfin Tools run (GROWTH + INUNDATION) for every combination of
# transition slope angle and growth factor:
#   - growth zones: the growth zone raster for that transition slope angle
#   - growth volume: length growth (growth factor, m3 per m of growth zone)
#     plus the initiation points as point sources
#   - optional maximum flow volume (max_volume), which caps volumes summed at
#     tributary junctions (Reid et al., 2025, sec. 4.2.2.6.2)
#   - inundation: custom volume-area coefficients (alpha, beta), Hbar
# The drainage network from the stream step is reused (not recomputed).
#
# Each run gets its own folder: scenarios_<name>/slope<angle>_growth<factor>/
# (with _vmax<volume> added when max_volume is set, so capped and uncapped runs
# are kept separately)
# Runs go one at a time (Grfin shares files next to the DEM between runs).
#
# Author: Nicolas Mathews

from pathlib import Path

import pandas as pd

from model.grfin import run_grfin


# -- Scenarios -- #

def _max_volume(cfg):
    """Maximum flow volume (m3) from the settings, or None for no cap."""
    v = cfg["scenarios"].get("max_volume")
    return None if v is None else float(v)


def _scenario_settings(cfg, dem, points_csv, zone_raster, angle, gf, name):
    """Grfin settings for one scenario (dem = the DEM path this run uses)."""
    s = cfg["scenarios"]
    return {
        "setup": {
            "title": f"{name} initiation, transition slope {angle:g} deg, "
                     f"growth factor {gf:g} m3/m",
            "demfilename": dem,
            "num_procs": s["num_procs"],
        },
        "options": {
            "inundation": True,
            "hl": False,
            "growth": True,
        },
        "drainage_options": {
            "compute_new_drainage": False,   # reuse the stream network
        },
        "growth_options": {
            "growth_zones": {
                "growth_zone_raster": Path(zone_raster).resolve(),
            },
            "growth_volumes": {
                "max_volume": _max_volume(cfg),
                "point_source_volumes_file": points_csv,
                "length_growth": {
                    "length_growth_factor": float(gf),
                },
            },
        },
        "inundation_options": {
            "flowtype": "custom",
            "xsec_alpha1": float(s["alpha"]),
            "plan_alpha2": float(s["beta"]),
            "hbar": bool(s["hbar"]),
        },
    }


def run_scenarios(cfg, points_csv, growth_zones, name):
    """Run all transition slope x growth factor scenarios for one point set.

    points_csv    x, y, initiation volume CSV (no header), used as point sources
    growth_zones  {angle: growth zone raster path}, from make_growth_zones()
    name          label for this point set, e.g. "modeled" or "inventory"
    Returns {(angle, growth_factor): inundation raster path}.
    """
    s = cfg["scenarios"]
    dem = Path(cfg["paths"]["dem"]).resolve()
    points_csv = Path(points_csv).resolve()
    out_dir = Path(cfg["paths"]["output_directory"]).resolve()
    base_dir = out_dir / f"scenarios_{name}"

    results = {}
    combos = [(a, g) for a in s["transition_slope_angles"] for g in s["growth_factors"]]
    for n, (angle, gf) in enumerate(combos, start=1):
        vmax = _max_volume(cfg)
        run_dir = base_dir / (f"slope{angle:g}_growth{gf:g}"
                              + (f"_vmax{vmax:g}" if vmax is not None else ""))
        inundation = run_dir / f"{dem.stem}_inundation.tif"
        results[(angle, gf)] = inundation
        label = f"{name}, slope {angle:g}, growth {gf:g}"

        if inundation.exists() and not s["recompute"]:
            print(f"Scenario {n}/{len(combos)} ({label}): using existing")
            continue

        print(f"Scenario {n}/{len(combos)} ({label})")
        settings = _scenario_settings(cfg, dem, points_csv, growth_zones[angle],
                                      angle, gf, name)
        run_grfin(settings, run_dir, expect=[inundation])

    return results


# -- No-Growth Runout -- #

def run_nogrowth(cfg, points_csv, name, volume_factor=1.0):
    """Inundation-only run without growth, using the scenario alpha and beta.

    Calibration adds this runout to every scenario (cells inundated here count
    as inundated in all scenarios), as in the MATLAB calibration workflow.
    volume_factor scales the points file for this run only, e.g.
    1 / initiation_fraction to turn initiation volumes into full volumes.
    Returns the path to the no-growth inundation raster.
    """
    s = cfg["scenarios"]
    dem = Path(cfg["paths"]["dem"]).resolve()
    run_dir = Path(cfg["paths"]["output_directory"]).resolve() / f"scenarios_{name}" / "nogrowth"
    inundation = run_dir / f"{dem.stem}_inundation.tif"

    if inundation.exists() and not s["recompute"]:
        print(f"No-growth runout ({name}): using existing")
        return inundation

    points_csv = Path(points_csv).resolve()
    if not points_csv.exists():
        raise FileNotFoundError(f"Initiation points file not found: {points_csv}")

    if volume_factor != 1.0:
        run_dir.mkdir(parents=True, exist_ok=True)
        pts = pd.read_csv(points_csv, header=None)
        pts[2] = pts[2] * volume_factor
        scaled_csv = run_dir / f"{name}_volumes_x{volume_factor:g}.csv"
        pts.to_csv(scaled_csv, index=False, header=False)
        print(f"No-growth runout ({name}): volumes x {volume_factor:g} -> {scaled_csv.name}")
        points_csv = scaled_csv

    settings = {
        "setup": {
            "title": f"No-growth runout from {name} initiation points",
            "demfilename": dem,
            "num_procs": s["num_procs"],
        },
        "options": {
            "inundation": True,
            "hl": False,
            "growth": False,
            "standalone_drainage_network": False,
        },
        "inundation_options": {
            "flowtype": "custom",
            "xsec_alpha1": float(s["alpha"]),
            "plan_alpha2": float(s["beta"]),
            "hbar": bool(s["hbar"]),
            "user_specified_volumes_file": points_csv,
        },
    }
    run_grfin(settings, run_dir, expect=[inundation])
    print(f"No-growth runout ({name}): wrote {inundation}")
    return inundation
