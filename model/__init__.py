# model/__init__.py
# Runs automatically the first time anything is imported from the model folder.

# Hide Standalone GDAL (Installed with TauDEM) from Python
import os
import sys

# Keep the untouched environment for external programs (Grfin, TauDEM),
# which may need the standalone GDAL that Python must not see
ORIGINAL_ENV = dict(os.environ)


def _is_outside_gdal(path):
    """True for GDAL/OSGeo4W folders that are not part of this conda env."""
    p = os.path.normpath(path.strip().strip('"')).rstrip("\\/").upper()
    if not p or p.startswith(os.path.normpath(sys.prefix).upper()):
        return False
    last = os.path.basename(p)
    return "GDAL" in last or "OSGEO4W" in p


os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep)
    if p and not _is_outside_gdal(p)
)

# Settings left by the standalone GDAL also break rasterio; drop them if they
# point outside this environment
for _var in ("GDAL_DATA", "GDAL_DRIVER_PATH", "PROJ_LIB", "PROJ_DATA"):
    _val = os.environ.get(_var)
    if _val and not os.path.normpath(_val).upper().startswith(
            os.path.normpath(sys.prefix).upper()):
        del os.environ[_var]