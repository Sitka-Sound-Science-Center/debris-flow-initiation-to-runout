# Grfin Tools Runner
#
# One function to run any Grfin Tools analysis from Python: settings are given
# as a nested dictionary (same sections and field names as the Grfin manual),
# written to a Settings YAML file, and run with grfintools.
#
# Each run gets its own folder (used as dest_dir), because Grfin names its
# primary outputs after the DEM (e.g. <DEM>_inundation.tif) and would
# otherwise overwrite earlier runs.
#
# Reference: Reid et al. (2025), Grfin Tools user guide, USGS TM 14-A3, sec. 4.2.2
#
# Author: Nicolas Mathews

import subprocess
from pathlib import Path

from model import ORIGINAL_ENV


# -- Settings File -- #

def _yaml_value(v):
    """Format one value the way Grfin's example Settings files do."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:g}" if isinstance(v, float) else str(v)
    if isinstance(v, Path):
        v = v.resolve().as_posix()
    s = str(v)
    plain = s.replace("_", "").replace("-", "").replace(".", "").replace("/", "")
    return s if plain.isalnum() else "'" + s.replace("'", "''") + "'"


def to_grfin_yaml(settings, indent=0):
    """Nested dictionary -> Grfin Settings text (two spaces per level).

    Fields set to None are left out, so Grfin uses its defaults.
    """
    lines = []
    pad = "  " * indent
    for key, val in settings.items():
        if val is None:
            continue
        if isinstance(val, dict):
            lines.append(f"{pad}{key}:")
            lines.extend(to_grfin_yaml(val, indent + 1).splitlines())
            if indent == 0:
                lines.append("")          # blank line between sections
        else:
            lines.append(f"{pad}{key}: {_yaml_value(val)}")
    return "\n".join(lines) + ("\n" if indent == 0 else "")


# -- Run -- #

def run_grfin(settings, run_dir, expect=(), grfin_cmd="grfintools"):
    """Run Grfin Tools with the given settings in its own folder.

    settings   nested dict with Grfin sections (setup, options, ...).
               setup: dest_dir is set to run_dir automatically.
    run_dir    folder for this run: Settings file, Grfin outputs and run log.
    expect     file names (in run_dir) or full paths that must exist after the
               run; a missing file raises an error even if Grfin exits cleanly.
    Returns run_dir as a Path.
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    name = run_dir.name

    settings = {**settings, "setup": {**settings["setup"], "dest_dir": run_dir}}
    yaml_file = run_dir / f"{name}.yaml"
    yaml_file.write_text(to_grfin_yaml(settings))

    # Run with the original environment, so TauDEM finds its own GDAL
    print(f"Grfin: running {name}")
    result = subprocess.run(f'{grfin_cmd} "{yaml_file.name}"', cwd=run_dir,
                            env=ORIGINAL_ENV, capture_output=True, text=True,
                            shell=True)
    log_file = run_dir / f"{name}_log.txt"
    log_file.write_text(result.stdout + "\n" + result.stderr)

    missing = [str(f) for f in expect
               if not (Path(f) if Path(f).is_absolute() else run_dir / f).exists()]
    if result.returncode != 0 or missing:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
        raise RuntimeError(
            f"Grfin run '{name}' failed (exit code {result.returncode})."
            + (f"\nMissing outputs: {', '.join(missing)}" if missing else "")
            + f"\nFull log: {log_file}\n\nLast lines:\n{tail}")
    return run_dir
