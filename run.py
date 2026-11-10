# Landslide Initiation and Runout Workflow
# Usage: python run.py settings.yaml
# Author: Nicolas Mathews

import sys
from pathlib import Path

import yaml

import model  # noqa: F401  (hides the TauDEM GDAL before rasterio loads)
from model.initiation import initiation
from model.grfin_drainage import grfin_stream_network
from model.growth_zones import run_exaggerated
from model.growth_zones import run_exaggerated, slope_degrees, make_growth_zones
from model.scenarios import run_scenarios, run_nogrowth
from model.calibration import calibrate, build_maps
from model.validation import validate_initiation

# Step 1: Load Settings
if len(sys.argv) < 2:
    sys.exit("Usage: python run.py <settings file>   e.g. python run.py settings/example_POW.yaml")
settings_file = sys.argv[1]
with open(settings_file) as f:
    cfg = yaml.safe_load(f)

# Step 2: Landslide Initiation (Initiation Points and Volumes)
init_points_path = Path(cfg["paths"]["output_directory"]) / "initiation_points.csv"
if cfg["initiation"]["recompute"] or not init_points_path.exists():
    init_points_path = initiation(cfg)
else:
    print(f"Initiation: using existing {init_points_path}")

# Step 2b: Initiation Validation (ROC / AUC)
if cfg["validation"]["run"]:
    validate_initiation(cfg)

# Optional: Use Other Modeled Initiation Points (e.g. MATLAB, for verification)
modeled_points = cfg["paths"].get("modeled_points_override") or init_points_path
if modeled_points != init_points_path:
    print(f"Using modeled points override: {modeled_points}")

# Step 3: Stream Network (Grfin DRAINAGE Tool, Run Once per DEM)
s = cfg["streams"]
grfin = grfin_stream_network(
    cfg["paths"]["dem"],
    Path(cfg["paths"]["output_directory"]) / "grfin_drainage",
    dem_neighborhood=s["dem_neighborhood_m"],
    min_curvature=s["min_curvature"],
    min_concav_area=s["min_concav_area_m2"],
    min_drainage_area=s["min_drainage_area_m2"],
    num_procs=s["num_procs"],
    recompute=s["recompute"],
)
stream_file = grfin["stream"]   # 1 = stream, 0 = not

# Step 4: Exaggerated Runout (Modeled and Inventory Points)
# Initiation volume -> full debris-flow volume (/ fraction), then exaggerate (x factor)
g = cfg["growth_zones"]
exaggerated_factor = g["exaggeration_factor"] / g["initiation_fraction"]
exaggerated_modeled = run_exaggerated(cfg, modeled_points, "modeled",
                                      volume_factor=exaggerated_factor)
if cfg["options"]["calibrate_runout_to_inventory"]:
    exaggerated_inventory = run_exaggerated(cfg, cfg["paths"]["inventory_volumes_grfin"], "inventory",
                                            volume_factor=exaggerated_factor)

# Step 5: Growth Zones (Stream Within Exaggerated Runout, Slope >= Transition Angle)
slope = slope_degrees(cfg["paths"]["dem"],
                      Path(cfg["paths"]["output_directory"]) / "slope_deg.tif")
growth_zones_modeled = make_growth_zones(cfg, exaggerated_modeled, grfin["stream"],
                                         slope, "modeled")
if cfg["options"]["calibrate_runout_to_inventory"]:
    growth_zones_inventory = make_growth_zones(cfg, exaggerated_inventory, grfin["stream"],
                                               slope, "inventory")

# Step 6: Runout Scenarios (All Transition Slopes x Growth Factors)
scenarios_modeled = run_scenarios(cfg, modeled_points, growth_zones_modeled, "modeled")
if cfg["options"]["calibrate_runout_to_inventory"]:
    scenarios_inventory = run_scenarios(cfg, cfg["paths"]["inventory_volumes_grfin"],
                                        growth_zones_inventory, "inventory")

# Step 7: No-Growth Runout (Full Debris-Flow Volumes, No Growth)
full_factor = 1 / cfg["growth_zones"]["initiation_fraction"]
nogrowth_modeled = run_nogrowth(cfg, modeled_points, "modeled", volume_factor=full_factor)
if cfg["options"]["calibrate_runout_to_inventory"]:
    nogrowth_inventory = run_nogrowth(cfg, cfg["paths"]["inventory_volumes_grfin"], "inventory",
                                      volume_factor=full_factor)

# Step 8: Calibrate to Inventory and Build Calibrated Maps
if cfg["options"]["calibrate_runout_to_inventory"]:
    cal = calibrate(cfg, scenarios_inventory, nogrowth_inventory)
    build_maps(cfg, cal, scenarios_inventory, nogrowth_inventory, "inventory")
    build_maps(cfg, cal, scenarios_modeled, nogrowth_modeled, "modeled")

print("Workflow complete.")