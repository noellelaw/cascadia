import torch
from typing import Tuple, Optional
import matplotlib

# Example Usage: 
#   depth = torch.rand(8, 1, 128, 128)         # 8 flood maps
#   condition = torch.rand(8, 3, 128, 128)     # SLR, wind, DEM
#   label = torch.randint(0, 10, (8, 128, 128))  # land cover class
#   flood_batch = Flood.from_XCD(depth, condition, label)
#   print(flood_batch)
#   flood_batch.visualize()


class Flood:
    """
    Flood: A utility class for managing flood tensors for generative modeling.

    Attributes:
        depth (torch.Tensor): Flood depth map, shape (B, 1, H, W) or (B, H, W).
        condition (torch.Tensor): Conditioning tensor (e.g., multi-channel forcing or SLR), shape (B, C, H, W).
        label (torch.Tensor): Discrete land cover or categorical map, shape (B, H, W).
        device (str): Torch device used for tensor operations.
    """

    def __init__(
        self,
        depth: torch.Tensor,        # Flood depth
        condition: Optional[torch.Tensor] = None,  # Conditioning like SLR, met forcing, topo
        label: Optional[torch.Tensor] = None,      # Discrete land cover or infrastructure mask
        device: str = "cpu"
    ):
        self.depth = depth.to(device)
        self.condition = condition.to(device) if condition is not None else None
        self.label = label.to(device) if label is not None else None
        self.device = device

    @classmethod
    def from_XCD(
        cls,
        X: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ) -> "Flood":
        return cls(X, C, D, device=device)

    def to_XCD(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.depth, self.condition, self.label

    def clone(self) -> "Flood":
        return Flood(
            self.depth.clone(),
            self.condition.clone() if self.condition is not None else None,
            self.label.clone() if self.label is not None else None,
            device=self.device
        )

    def shape(self) -> Tuple[int, int, int]:
        return self.depth.shape

    def apply_mask(self, mask: torch.Tensor) -> "Flood":
        masked_depth = self.depth * mask
        masked_condition = self.condition * mask if self.condition is not None else None
        masked_label = self.label * mask if self.label is not None else None
        return Flood(masked_depth, masked_condition, masked_label, device=self.device)

    def __str__(self):
        cond_shape = self.condition.shape if self.condition is not None else None
        label_shape = self.label.shape if self.label is not None else None
        return f"Flood(depth={self.depth.shape}, condition={cond_shape}, label={label_shape}, device={self.device})"

    def visualize(self, cmap="Blues", index: int = 0):
        import matplotlib.pyplot as plt
        plt.imshow(self.depth[index].squeeze().cpu(), cmap=cmap)
        plt.title("Flood Depth")
        plt.colorbar()
        plt.show()
