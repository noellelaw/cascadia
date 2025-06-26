
# Maybe right now I need to simplify (</3) and assume I am reading in a CSV that maps to a:
# 1. path to a raster file with flood depth data
# 2. path to a raster file with land use mask data
# 3. A column for SLR data (relative to the 1900 baseline, meters please)
# 4. A column for meteorlogical data (e.g., surge data)
import torch
import pandas as pd
from typing import Optional
from cascadia.data.utils.raster_to_grid import process_csv  # <- your raster grid processing function

class Flood:
    """
    Flood scenario representation based on DDPM-sampled grid data and optional infrastructure context.

    Will eventually take in precipitation, wind pressure, surge data if a good enough corpus is available. 
    (Selfishly hating on amsterdam for not being in a surge zone)
    
    # Example usage
        if __name__ == "__main__":
            flood = Flood.from_csv("validation.csv", grid_size_m=100)
            print("Flood tensor shapes:")
            print("Depth:", flood.depth.shape)
            print("Land mask:", flood.land_mask.shape)
            if flood.meta is not None:
                print("Meta:", flood.meta.shape)
    """

    def __init__(
            self, depth: torch.Tensor, land_mask: torch.Tensor, 
            meta: Optional[torch.Tensor] = None, 
            slr: Optional[torch.Tensor] = None
        ):
        """
        Args:
            depth (Tensor): (B, H, W) tensor of flood depths or elevation perturbations.
            land_mask (Tensor): (B, H, W) binary or categorical land use mask.
            meta (Tensor): (B, H, W, D) optional metadata (e.g., surge, pressure, velocity).
        """
        self.depth = depth
        self.land_mask = land_mask
        self.meta = meta
        self.slr = slr  # scalar value (float)

    @staticmethod
    def from_XCD(X: torch.Tensor, C: torch.Tensor, D: Optional[torch.Tensor]) -> "Flood":
        return Flood(depth=X, land_mask=C, meta=D)

    def to_XCD(self):
        return self.depth, self.land_mask, self.meta

    def canonicalize(self):
        self.depth = torch.clamp(self.depth, min=0.0)
        return self

    @staticmethod
    def from_csv(csv_path: str, grid_size_m: int = 100) -> "Flood":
        """
        Load flood scenario tensors from a CSV describing (depth + landcover) rasters
        + SLR mean value since 1900.
        
        """
        df = process_csv(csv_path, grid_size_m=grid_size_m)

        # Pivot into (B, H, W) shaped tensors (assume batch=1 for now)
        # You may later want to group by input filenames

        x_coords = sorted(df["x"].unique())
        y_coords = sorted(df["y"].unique(), reverse=True)  # top-to-bottom

        H, W = len(y_coords), len(x_coords)
        x_map = {x: i for i, x in enumerate(x_coords)}
        y_map = {y: i for i, y in enumerate(y_coords)}

        depth_tensor = torch.zeros((1, H, W), dtype=torch.float32)
        land_tensor = torch.zeros((1, H, W), dtype=torch.long)
        has_meta = "mean_surge" in df.columns
        meta_tensor = torch.zeros((1, H, W, 1), dtype=torch.float32) if has_meta else None

        for _, row in df.iterrows():
            i = y_map[row["y"]]
            j = x_map[row["x"]]
            depth_tensor[0, i, j] = row["mean_depth"]
            land_tensor[0, i, j] = row["land_cover"]
            if has_meta:
                meta_tensor[0, i, j, 0] = row["mean_surge"]

        return Flood(depth=depth_tensor, land_mask=land_tensor, meta=meta_tensor)


