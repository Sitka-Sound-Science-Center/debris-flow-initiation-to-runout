# Grfin Drainage Network (Standalone DRAINAGE Tool Run)
#
# Runs Grfin Tools once in drainage-only mode to build the curvature-based
# stream network with TauDEM, exactly as Grfin does (Reid et al., 2025,
# sec. 3.4 and Example C). The DRAINAGE tool:
#   1. smooths the DEM with a circular mean window (dem_neighborhood radius)
#   2. computes planform curvature (Zevenbergen and Thorne, 1987), 3x3 mean
#   3. keeps concave cells (curvature >= min_curvature_drainage)
#   4. drops concave patches smaller than min_concav_area
#   5. accumulates contributing area of the remaining concave cells
#   6. starts channels where that area >= min_drainage_area, then follows D8
#
# Grfin writes its outputs next to the DEM, in its standard layout, so later
# Grfin scenario runs (compute_new_drainage: no) reuse the same files:
#   <DEM folder>/<DEM>_drainage/<DEM>_stream.tif       (1 = stream, 0 = not)
#   <DEM folder>/<DEM>_drainage/<DEM>_smoothplancurv.tif
#   <DEM folder>/<DEM>_taudem/<DEM>_felp.tif            (TauDEM D8 directions)
# work_dir only holds the settings file and Grfin's run log and summary.
#
# Author: Nicolas Mathews

import subprocess
from pathlib import Path

import numpy as np
import rasterio

from model import ORIGINAL_ENV


def grfin_paths(dem_path):
    """Paths of the Grfin drainage and TauDEM outputs for a DEM."""
    dem = Path(dem_path).resolve()
    stem = dem.stem
    return {
        "stream": dem.parent / f"{stem}_drainage" / f"{stem}_stream.tif",
        "plancurv": dem.parent / f"{stem}_drainage" / f"{stem}_smoothplancurv.tif",
        "settings": dem.parent / f"{stem}_drainage" / f"{stem}_drain_settings.txt",
        "fel": dem.parent / f"{stem}_taudem" / f"{stem}_fel.tif",
        "felp": dem.parent / f"{stem}_taudem" / f"{stem}_felp.tif",
        "ad8": dem.parent / f"{stem}_taudem" / f"{stem}_ad8.tif",
    }


def grfin_stream_network(dem_path, work_dir,
                         dem_neighborhood=10.0, min_curvature=0.02,
                         min_concav_area=200.0, min_drainage_area=200.0,
                         num_procs=8, recompute=False, grfin_cmd="grfintools"):
    """Build the Grfin stream network for a DEM (skipped if it already exists).

    Drainage and TauDEM outputs go next to the DEM (Grfin's standard layout);
    work_dir holds the settings file, run log and Grfin's summary output.
    Returns the dictionary from grfin_paths().
    """
    dem = Path(dem_path).resolve()
    work = Path(work_dir).resolve()
    paths = grfin_paths(dem)

    if paths["stream"].exists() and not recompute:
        print(f"Grfin drainage: using existing {paths['stream']}")
        return paths

    work.mkdir(parents=True, exist_ok=True)

    # Settings file (single quotes keep spaces and backslashes safe in YAML)
    yaml_file = work / f"{dem.stem}_drainage.yaml"
    yaml_file.write_text(
        "setup:\n"
        f"  title: Standalone drainage network for {dem.stem}\n"
        f"  demfilename: '{dem.as_posix()}'\n"
        f"  dest_dir: '{work.as_posix()}'\n"
        f"  num_procs: {int(num_procs)}\n"
        "\n"
        "options:\n"
        "  inundation: no\n"
        "  growth: no\n"
        "  hl: no\n"
        "  standalone_drainage_network: yes\n"
        "\n"
        "drainage_options:\n"
        "  compute_new_drainage: yes\n"
        f"  dem_neighborhood: {dem_neighborhood:g}\n"
        f"  min_curvature_drainage: {min_curvature:g}\n"
        f"  min_concav_area: {min_concav_area:g}\n"
        f"  min_drainage_area: {min_drainage_area:g}\n"
    )

    # Run Grfin with the original environment, so TauDEM finds its own GDAL
    print(f"Grfin drainage: running {grfin_cmd} {yaml_file.name}")
    result = subprocess.run([grfin_cmd, yaml_file.name], cwd=work,
                            env=ORIGINAL_ENV, capture_output=True, text=True,
                            shell=True)  # shell=True finds grfintools like a terminal does
    log_file = work / f"{dem.stem}_drainage_log.txt"
    log_file.write_text(result.stdout + "\n" + result.stderr)

    if result.returncode != 0 or not paths["stream"].exists():
        raise RuntimeError(
            f"Grfin drainage run failed (exit code {result.returncode}).\n"
            f"See {log_file}\n\nLast lines:\n"
            + "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:]))

    print(f"Grfin drainage: wrote {paths['stream']}")
    return paths


# -- Flow Directions Matching the Grfin Network -- #

# TauDEM D8:    1=E 2=NE 3=N 4=NW 5=W 6=SW 7=S 8=SE
# pyflwdir D8:  1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE, 0=pit, 247=nodata
_TAUDEM_TO_PYFLWDIR = {1: 1, 2: 128, 3: 64, 4: 32, 5: 16, 6: 8, 7: 4, 8: 2}


def load_grfin_flwdir(dem_path):
    """pyflwdir flow-direction object built from Grfin's TauDEM D8 grid.

    Use this for routing in the runout code so flow paths follow the same
    D8 directions as the Grfin stream network.
    """
    import pyflwdir

    felp = grfin_paths(dem_path)["felp"]
    with rasterio.open(felp) as src:
        td = src.read(1, masked=True)
        transform = src.transform
    d8 = np.full(td.shape, 247, dtype="uint8")          # NoData by default
    valid = ~np.ma.getmaskarray(td)
    vals = td.filled(0)
    d8[valid] = 0                                        # pits/outlets
    for t, p in _TAUDEM_TO_PYFLWDIR.items():
        d8[valid & (vals == t)] = p
    return pyflwdir.from_array(d8, ftype="d8", transform=transform, latlon=False)
