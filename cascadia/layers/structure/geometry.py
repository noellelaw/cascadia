import torch
import torch.nn as nn

class GridDistances(nn.Module):
    """Compute Euclidean distances between grid cells."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None):
        # X: (B, H, W, 2) = coordinates or centroids
        B, H, W, _ = X.shape
        X_flat = X.view(B, H * W, 2)
        dist = torch.cdist(X_flat, X_flat, p=2).view(B, H, W, H, W)
        if mask is not None:
            dist = dist * mask.unsqueeze(1).unsqueeze(1)
        return dist


class FlowAngles(nn.Module):
    """Compute angles between flow gradients."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, grad_x: torch.Tensor, grad_y: torch.Tensor):
        # Compute angles (in radians) between horizontal and vertical slopes
        dot = (grad_x * grad_y).sum(dim=-1)
        norm_x = grad_x.norm(dim=-1)
        norm_y = grad_y.norm(dim=-1)
        cos_angle = dot / (norm_x * norm_y + self.eps)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        return angle

class VirtualNodes(nn.Module):
    # Something done in chroma that could be used for inferred locations
    # for cascading infrastructure 
    def __init__(self, offset=(1.0, 0.0)):
        super().__init__()
        self.offset = torch.tensor(offset).view(1, 1, 1, 2)

    def forward(self, X: torch.Tensor):
        # X: (B, H, W, 2)=position of each grid cell
        return X + self.offset.to(X.device)