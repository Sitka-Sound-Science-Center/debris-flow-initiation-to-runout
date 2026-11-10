import numpy as np

def infinite_slope(c, gamma, phi, slope, depth, mw, gamma_w, z=None, fs_cap=None):
    """Infinite slope factor of safety.

    c       cohesion, soil + roots (kPa)
    gamma   unit weight of soil (kN/m3)
    phi     friction angle (degrees)
    slope   slope angle (degrees)
    depth   colluvium depth (m)
    mw      saturation ratio
    gamma_w unit weight of water (kN/m3)
    z       DEM (optional), used to set NaNs outside the DEM
    fs_cap  value to replace infinite FS (optional, e.g. zero-depth cells)
    """

    # Compute Stability Using Infinite Slope
    fs = (c / (gamma * depth * np.sin(np.radians(slope)))
          + (np.cos(np.radians(slope)) * np.tan(np.radians(phi)) / np.sin(np.radians(slope)))
          * (1 - mw * gamma_w / gamma))

    # Make Sure that NaNs are Reflected in FS Raster
    if z is not None:
        fs[np.isnan(z)] = np.nan

    # Replace Infinite FS
    if fs_cap is not None:
        fs[np.isinf(fs)] = fs_cap

    return fs