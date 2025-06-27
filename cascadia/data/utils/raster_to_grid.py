import rasterio
import numpy as np
import pandas as pd
import os
from rasterio.transform import rowcol
from tqdm import tqdm
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio

def reproject_to_match(src_path, dst_path, target_crs):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest  # good for landcover
                )

def process_raster_row(depth_path, landcover_path, slr, grid_size=512):
    """
    Processes a large raster into smaller flood grid grids (e.g., 256x256),
    only retaining grids with valid depth data.

    Args:
        depth_path (str): Path to the flood depth raster.
        landcover_path (str): Path to the land cover raster.
        slr (float): Scalar sea level rise to add to flood depths.
        grid_size (int): Size of the square grid to extract (in raster pixels).

    Returns:
        pd.DataFrame: Table of retained grid center coordinates and features.
    """

    with rasterio.open(depth_path) as depth_src, rasterio.open(landcover_path) as nlcd_src:
        print("Depth CRS:", depth_src.crs)
        print("Landcover CRS:", nlcd_src.crs)
        if depth_src.crs != nlcd_src.crs:
            # Reproject landcover to match depth CRS
            if nlcd_src.crs is None:
                raise ValueError(f"Landcover raster {landcover_path} has no CRS defined.")
            if depth_src.crs is None:
                raise ValueError(f"Depth raster {depth_path} has no CRS defined.")

            # Reproject landcover to match depth CRS
            if depth_src.crs.to_string() != nlcd_src.crs.to_string():
                print(f"CRS mismatch between {depth_path} and {landcover_path}, reprojecting landcover to match depth CRS")
                target_crs = depth_src.crs
                reprojected_path = landcover_path + "_reprojected.tif"
                reproject_to_match(landcover_path, reprojected_path, target_crs)
                nlcd_src.close()  # Close original
                nlcd_src = rasterio.open(reprojected_path)  # Reopen reprojected raster

        depth = depth_src.read(1).astype(float)
        landcover = nlcd_src.read(1).astype(int)

        depth[depth == depth_src.nodata] = np.nan
        landcover[landcover == nlcd_src.nodata] = -9999

        depth += slr

        transform = depth_src.transform
        height, width = depth.shape

        records = []

        for row_start in range(0, height, grid_size):
            for col_start in range(0, width, grid_size):
                row_end = min(row_start + grid_size, height)
                col_end = min(col_start + grid_size, width)

                depth_patch = depth[row_start:row_end, col_start:col_end]
                lc_patch = landcover[row_start:row_end, col_start:col_end]

                if np.isnan(depth_patch).all():
                    continue  # Skip empty grids

                # Get approximate center coordinates of the grid
                center_row = (row_start + row_end) // 2
                center_col = (col_start + col_end) // 2
                center_x, center_y = rasterio.transform.xy(transform, center_row, center_col)

                mean_depth = np.nanmean(depth_patch)
                dominant_landcover = (
                    np.bincount(lc_patch[lc_patch != -9999].flatten()).argmax()
                    if np.any(lc_patch != -9999) else -9999
                )

                records.append({
                    "x": center_x,
                    "y": center_y,
                    "mean_depth": mean_depth,
                    "land_cover": dominant_landcover,
                    "slr_added": slr
                })
    return pd.DataFrame(records)

def process_csv(csv_path, grid_size_m=512):
    df_inputs = pd.read_csv(csv_path)
    root_dir = os.path.dirname(os.path.abspath(csv_path))  # Determine root directory of the CSV
    all_results = []

    for _, row in tqdm(df_inputs.iterrows(), total=len(df_inputs)):
        # Resolve absolute paths to TIF files
        depth_path =  os.path.normpath(os.path.join(root_dir,  row["depth_tif"]))
        landcover_path = os.path.normpath(os.path.join(root_dir, row["landcover_tif"]))

        out_df = process_raster_row(
            depth_path,
            landcover_path,
            row["slr"],
            grid_size=grid_size_m
        )
        out_df["source_depth_tif"] = row["depth_tif"]
        out_df["source_landcover_tif"] = row["landcover_tif"]
        all_results.append(out_df)

    return pd.concat(all_results, ignore_index=True)
