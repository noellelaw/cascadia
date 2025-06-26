import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol, xy
from affine import Affine

def raster_to_aligned_grid(depth_path, nlcd_path, grid_size_m=100):
    """Read depth and NLCD rasters, resample to a fixed grid, and overlay depths + land cover.
    
    # Example usage:
        depth_tif = "path/to/flood_depth.tif"
        nlcd_tif = "path/to/nlcd.tif"
        grid_df = raster_to_aligned_grid(depth_tif, nlcd_tif, grid_size_m=100)

    # Preview
        import ace_tools as tools; tools.display_dataframe_to_user(name="Flood + NLCD Grid", dataframe=grid_df)
    
    """
    
    with rasterio.open(depth_path) as depth_src, rasterio.open(nlcd_path) as nlcd_src:
        # Sanity check CRS match
        assert depth_src.crs == nlcd_src.crs, "Depth and NLCD rasters must share the same CRS"

        # Read arrays
        depth = depth_src.read(1).astype(float)
        nlcd = nlcd_src.read(1).astype(int)
        
        # Mask no-data
        if depth_src.nodata is not None:
            depth[depth == depth_src.nodata] = np.nan
        if nlcd_src.nodata is not None:
            nlcd[nlcd == nlcd_src.nodata] = -9999
        
        # Extract bounds and resolution from depth raster
        bounds = depth_src.bounds
        transform = depth_src.transform
        resolution = transform.a  # assume square pixels

        # Generate grid coordinates
        x_coords = np.arange(bounds.left, bounds.right, grid_size_m)
        y_coords = np.arange(bounds.top, bounds.bottom, -grid_size_m)

        grid = []
        for i, y in enumerate(y_coords[:-1]):
            for j, x in enumerate(x_coords[:-1]):
                # Get window in pixel coordinates
                row_start, col_start = rowcol(transform, x, y)
                row_end, col_end = rowcol(transform, x + grid_size_m, y - grid_size_m)

                # Clip windows to array bounds
                row_start = max(0, min(depth.shape[0], row_start))
                row_end = max(0, min(depth.shape[0], row_end))
                col_start = max(0, min(depth.shape[1], col_start))
                col_end = max(0, min(depth.shape[1], col_end))

                # Extract values from each window
                depth_patch = depth[row_start:row_end, col_start:col_end]
                nlcd_patch = nlcd[row_start:row_end, col_start:col_end]

                if depth_patch.size == 0:
                    continue

                mean_depth = np.nanmean(depth_patch)
                dominant_landcover = np.bincount(nlcd_patch[nlcd_patch != -9999].flatten()).argmax() if np.any(nlcd_patch != -9999) else -9999

                grid.append({
                    "x": x + grid_size_m / 2,
                    "y": y - grid_size_m / 2,
                    "mean_depth": mean_depth,
                    "land_cover": dominant_landcover
                })

    return pd.DataFrame(grid)


