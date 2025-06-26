import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from tqdm import tqdm

def process_raster_row(depth_path, landcover_path, slr, grid_size_m=100):
    """
    Processes a single row: overlays depth + landcover + SLR to a common grid.
    

    # Example usage:
        csv_path = "path/to/inputs.csv"
        grid_df = process_csv(csv_path, grid_size_m=100)

        import ace_tools as tools; tools.display_dataframe_to_user(name="SLR Grid Output", dataframe=grid_df)
    """
    
    with rasterio.open(depth_path) as depth_src, rasterio.open(landcover_path) as nlcd_src:
        # Confirm CRS match
        assert depth_src.crs == nlcd_src.crs, f"CRS mismatch between {depth_path} and {landcover_path}"

        # Read arrays
        depth = depth_src.read(1).astype(float)
        landcover = nlcd_src.read(1).astype(int)

        # Mask nodata
        depth[depth == depth_src.nodata] = np.nan
        landcover[landcover == nlcd_src.nodata] = -9999

        # Adjust depth by adding SLR
        depth += slr

        # Get bounds and transform
        bounds = depth_src.bounds
        transform = depth_src.transform

        # Define grid
        x_coords = np.arange(bounds.left, bounds.right, grid_size_m)
        y_coords = np.arange(bounds.top, bounds.bottom, -grid_size_m)

        records = []
        for y in y_coords[:-1]:
            for x in x_coords[:-1]:
                row_start, col_start = rowcol(transform, x, y)
                row_end, col_end = rowcol(transform, x + grid_size_m, y - grid_size_m)

                row_start = max(0, min(depth.shape[0], row_start))
                row_end = max(0, min(depth.shape[0], row_end))
                col_start = max(0, min(depth.shape[1], col_start))
                col_end = max(0, min(depth.shape[1], col_end))

                depth_patch = depth[row_start:row_end, col_start:col_end]
                lc_patch = landcover[row_start:row_end, col_start:col_end]

                if depth_patch.size == 0:
                    continue

                mean_depth = np.nanmean(depth_patch)
                dominant_landcover = (
                    np.bincount(lc_patch[lc_patch != -9999].flatten()).argmax()
                    if np.any(lc_patch != -9999) else -9999
                )

                records.append({
                    "x": x + grid_size_m / 2,
                    "y": y - grid_size_m / 2,
                    "mean_depth": mean_depth,
                    "land_cover": dominant_landcover,
                    "slr_added": slr
                })

    return pd.DataFrame(records)

def process_csv(csv_path, grid_size_m=100):
    df_inputs = pd.read_csv(csv_path)
    all_results = []

    for _, row in tqdm(df_inputs.iterrows(), total=len(df_inputs)):
        out_df = process_raster_row(
            row["depth_tif"],
            row["landcover_tif"],
            row["slr"],
            grid_size_m=grid_size_m
        )
        out_df["source_depth_tif"] = row["depth_tif"]
        out_df["source_landcover_tif"] = row["landcover_tif"]
        all_results.append(out_df)

    return pd.concat(all_results, ignore_index=True)

