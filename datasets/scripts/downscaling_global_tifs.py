import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label


# Hyperparameters
FLOOD_VOLUME_TIF = "flood_volume.tif"
HIGH_RES_DEM_TIF = "high_res_dem.tif"


def fill_volume_to_depth(dem_patch, volume, cell_area, dz=0.1):
    dem = dem_patch.copy()
    flooded = np.zeros_like(dem, dtype=float)
    mask = ~np.isnan(dem)
    water_depth = np.zeros_like(dem)

    h = 0.0
    filled_volume = 0.0

    while filled_volume < volume:
        h += dz
        depth = np.maximum(h - dem, 0)
        depth[~mask] = 0
        flooded_volume = np.sum(depth * cell_area)
        
        if flooded_volume >= volume:
            excess = flooded_volume - volume
            depth_adjust = excess / (np.sum(depth > 0) * cell_area)
            depth -= depth_adjust
            break

        water_depth = depth

    return water_depth
with rasterio.open(FLOOD_VOLUME_TIF) as vol_src, \
     rasterio.open(HIGH_RES_DEM_TIF) as dem_src:

    output = np.zeros(dem_src.shape, dtype=float)

    for i in range(0, vol_src.height):
        for j in range(0, vol_src.width):
            vol = vol_src.read(1)[i, j]
            if vol <= 0:
                continue

            # Get bounds & read DEM patch
            bounds = vol_src.window_bounds(Window(j, i, 1, 1))
            window = dem_src.window(*bounds)
            dem_patch = dem_src.read(1, window=window)

            cell_area = 1e6  # Assuming 1 km² based on www.hydrol-earth-syst-sci.net/17/1871/2013/
            depth_patch = fill_volume_to_depth(dem_patch, vol, cell_area)
            output[window.row_off:window.row_off+window.height,
                   window.col_off:window.col_off+window.width] += depth_patch