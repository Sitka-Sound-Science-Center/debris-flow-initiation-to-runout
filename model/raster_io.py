import numpy as np
import rasterio

def save_raster(path, array, profile, dtype="float32", nodata=np.nan):
    """Save an array as a GeoTIFF using the georeferencing in profile."""
    profile = profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)