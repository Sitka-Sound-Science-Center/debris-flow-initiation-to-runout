def slope_aspect(z, x_cellsize, y_cellsize):
    """Slope and aspect using Horn's method"""

    # Import Libraries
    import numpy as np

    # Preallocate (edges stay 0, as in the MATLAB version)
    slope = np.zeros_like(z)
    aspect = np.zeros_like(z)
    dx = np.zeros_like(z)
    dy = np.zeros_like(z)

    # Gradient Using Weighted 3x3 Neighborhood (All Interior Cells at Once)
    dz_dx = ((z[:-2, 2:] + 2*z[1:-1, 2:] + z[2:, 2:])
             - (z[:-2, :-2] + 2*z[1:-1, :-2] + z[2:, :-2])) / (8*x_cellsize)
    dz_dy = ((z[2:, :-2] + 2*z[2:, 1:-1] + z[2:, 2:])
             - (z[:-2, :-2] + 2*z[:-2, 1:-1] + z[:-2, 2:])) / (8*y_cellsize)

    # Compute Slope
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Compute Aspect (Clockwise from North, ArcGIS Convention)
    asp = np.degrees(np.arctan2(dz_dy, -dz_dx))
    cell = np.where(asp < 0, 90 - asp,
           np.where(asp > 90, 360 - asp + 90, 90 - asp))

    # Fill Interior Cells
    slope[1:-1, 1:-1] = slope_deg
    aspect[1:-1, 1:-1] = np.round(cell)
    dx[1:-1, 1:-1] = dz_dx
    dy[1:-1, 1:-1] = dz_dy

    return slope