import torch

# Maybe right now I need to simplify (</3) and assume I am reading in a CSV that maps to a:
# 1. path to a raster file with flood depth data
# 2. path to a raster file with land use mask data
# 3. A column for SLR data (relative to the 1900 baseline, meters please)
# 4. A column for meteorlogical data (e.g., surge data)
class Flood:
    """Flood scenario representation based on DDPM-sampled grid data and optional infrastructure context."""

    def __init__(self, depth: torch.Tensor, land_mask: torch.Tensor, meta: torch.Tensor):
        """
        Args:
            depth (Tensor): (B, H, W) tensor of flood depths or elevation perturbations.
            land_mask (Tensor): (B, H, W) binary or categorical land use mask.
            meta (Tensor): (B, H, W, D) optional metadata (e.g., velocity, flow direction, damage levels).
        """
        self.depth = depth
        self.land_mask = land_mask
        self.meta = meta

    @staticmethod
    def from_XCD(X: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> "Flood":
        """Construct Flood object from tensor representations."""
        return Flood(depth=X, land_mask=C, meta=D)

    def to_XCD(self):
        """Convert Flood object to tensors."""
        return self.depth, self.land_mask, self.meta

    def canonicalize(self):
        """Placeholder for normalization, projection, or mask cleanup."""
        self.depth = torch.clamp(self.depth, min=0.0)
        return self