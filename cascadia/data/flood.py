
# Maybe right now I need to simplify (</3) and assume I am reading in a CSV that maps to a:
# 1. path to a raster file with flood depth data
# 2. path to a raster file with land use mask data
# 3. A column for SLR data (relative to the 1900 baseline, meters please)
# 4. A column for meteorlogical data (e.g., surge data)
import torch
import pandas as pd
from typing import Optional, Tuple, Generator
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

    
    def to_XCD(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert Flood object into flattened X (flood input), C (conditioning), and D_meta (e.g., land cover).

        Returns:
            X (torch.Tensor): Flood input features — e.g., depth, shape (B, H*W, 1)
            C (torch.Tensor): Conditioning features — e.g., SLR, surge, shape (B, H*W, K)
            D_meta (torch.Tensor): Metadata — e.g., land cover class, shape (B, H*W, 1)
        """
        B, H, W = self.depth.shape

        # X: flood depth field
        X = self.depth.unsqueeze(-1)          # (B, H, W, 1)
        X = X.view(B, H * W, 1)               # (B, H*W, 1)

        # C: conditioning features
        if self.meta is not None:
            Bm, Hm, Wm, K = self.meta.shape
            assert B == Bm and H == Hm and W == Wm, "Meta tensor shape mismatch."
            C = self.meta.view(B, H * W, K)   # (B, H*W, K)
        else:
            raise ValueError("Expected 'meta' to contain SLR and meteorological data for conditioning.")

        # D_meta: land cover
        D_meta = self.land_mask.unsqueeze(-1).float()  # (B, H, W, 1)
        D_meta = D_meta.view(B, H * W, 1)              # (B, H*W, 1)
        return X, C, D_meta

    def to_XCD_tiled(
        self,
        tile_H: int = 32,
        tile_W: int = 32,
        overlap: int = 8  # you can adjust overlap as needed
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Convert Flood object into tiled flattened X (depth input), C (conditioning), and D_meta (e.g. land cover).

        Yields:
            Tuple of (X_tile, C_tile, D_meta_tile)
        """
        B, H, W = self.depth.shape

        if self.meta is None:
            raise ValueError("Expected 'meta' to contain SLR and meteorological data for conditioning.")

        for y0 in range(0, H, tile_H - overlap):
            y1 = min(y0 + tile_H, H)
            for x0 in range(0, W, tile_W - overlap):
                x1 = min(x0 + tile_W, W)

                # Ensure tile has valid size (in case at edges)
                h = y1 - y0
                w = x1 - x0

                # Slice tiles
                depth_tile = self.depth[:, y0:y1, x0:x1]  # (B, h, w)
                meta_tile = self.meta[:, y0:y1, x0:x1, :]  # (B, h, w, K)
                land_mask_tile = self.land_mask[:, y0:y1, x0:x1]  # (B, h, w)

                # Flatten for GNN
                B_tile = B
                K = meta_tile.shape[-1]

                X_tile = depth_tile.unsqueeze(-1).view(B_tile, h * w, 1)  # (B, h*w, 1)
                C_tile = meta_tile.view(B_tile, h * w, K)                 # (B, h*w, K)
                D_meta_tile = land_mask_tile.unsqueeze(-1).float().view(B_tile, h * w, 1)  # (B, h*w, 1)

                yield X_tile, C_tile, D_meta_tile
    
    def canonicalize(self):
        self.depth = torch.clamp(self.depth, min=0.0)
        return self

    @staticmethod
    def from_csv(csv_path: str) -> "Flood":
        """
        Load flood scenario tensors from a CSV describing (depth + landcover) rasters
        and SLR/meteorological conditioning metadata.

        Returns:
            Flood instance with batched tensors of shape:
                depth:     (B, H, W)
                land_mask: (B, H, W)
                meta:      (B, H, W, K)
        """
        df = process_csv(csv_path)

        # Group by source files (or scenario ID) to build each batch slice
        group_key = "source_depth_tif"  # or a "scenario_id" column if available
        grouped = df.groupby(group_key)
        batch_size = len(grouped)

        # Extract universal x/y coordinates from the first group (assumes aligned grids)
        x_coords = sorted(df["x"].unique())
        y_coords = sorted(df["y"].unique(), reverse=True)
        H, W = len(y_coords), len(x_coords)
        x_map = {x: i for i, x in enumerate(x_coords)}
        y_map = {y: i for i, y in enumerate(y_coords)}

        # Allocate batched tensors
        depth_tensor = torch.zeros((batch_size, H, W), dtype=torch.float32)
        land_tensor = torch.zeros((batch_size, H, W), dtype=torch.long)
        has_meta = "slr_added" in df.columns
        meta_tensor = torch.zeros((batch_size, H, W, 1), dtype=torch.float32) if has_meta else None

        for b, (_, group_df) in enumerate(grouped):
            for _, row in group_df.iterrows():
                i = y_map[row["y"]]
                j = x_map[row["x"]]
                depth_tensor[b, i, j] = row["mean_depth"]
                land_tensor[b, i, j] = row["land_cover"]
                if has_meta:
                    meta_tensor[b, i, j, 0] = row["slr_added"]
        return Flood(depth=depth_tensor, land_mask=land_tensor, meta=meta_tensor)


