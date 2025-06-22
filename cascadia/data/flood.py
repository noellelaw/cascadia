class Flood:
    """
    Flood: A utility class for managing flood maps.

    Loading, transforming, and exporting flood data in tensor format for training and inference.

    Attributes:
        depth (torch.Tensor): Flood depth map, shape (B, H, W) or (B, 1, H, W).
        condition (torch.Tensor): Conditioning feature map (e.g., elevation), shape (B, H, W).
        label (torch.Tensor): Discrete land cover class label map, shape (B, H, W).
        device (str): Torch device used for tensor operations.
    """

    def __init__(self, X: torch.Tensor, C: torch.Tensor, D: torch.Tensor, device: str = "cpu"):
        self.depth = X.to(device)
        self.condition = C.to(device)
        self.label = D.to(device)
        self.device = device

    @classmethod
    def from_XCD(cls, X: torch.Tensor, C: torch.Tensor, D: torch.Tensor, device: str = "cpu") -> Flood:
        return cls(X, C, D, device=device)

    def to_XCD(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.depth, self.condition, self.label

    def clone(self) -> Flood:
        return Flood(self.depth.clone(), self.condition.clone(), self.label.clone(), device=self.device)

    def shape(self) -> Tuple[int, int, int]:
        return self.depth.shape

    def apply_mask(self, mask: torch.Tensor) -> Flood:
        masked_X = self.depth * mask
        masked_C = self.condition * mask
        masked_D = self.label * mask
        return Flood(masked_X, masked_C, masked_D, device=self.device)

    def __str__(self):
        return f"Flood(depth={self.depth.shape}, condition={self.condition.shape}, label={self.label.shape}, device={self.device})"

    def visualize(self, cmap="Blues"):
        import matplotlib.pyplot as plt
        plt.imshow(self.depth[0].squeeze().cpu(), cmap=cmap)
        plt.title("Flood Depth")
        plt.colorbar()
        plt.show()
